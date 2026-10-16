"""Step 2: MinerU PDF parsing.

Runs mineru inside conda env.
Output goes to TMP_DIR.
After finding _content_list_v2.json, deletes intermediate files
not needed by later steps to reduce disk I/O.
Uses pipeline backend (GPU-accelerated via device-mode: cuda in config).
"""

import os
import subprocess
import logging
from pathlib import Path

from ..config import MINERU_CONDA_ENV, MINERU_CMD, MINERU_MODEL_SOURCE, MINERU_METHOD, MINERU_TIMEOUT, MINERU_LANG, MINERU_MODELSCOPE_CACHE, TMP_DIR, MAX_RETRIES

logger = logging.getLogger(__name__)


def _clean_intermediate_output(out_dir: Path):
    """Delete MinerU intermediate files that are not needed by later steps.
    
    Keeps: _content_list_v2.json, images/, origin.pdf, and anything else.
    Deletes: _middle.json, _model.json, *.md, _layout.pdf, _span.pdf.
    """
    patterns = [
        "*_middle.json", "*_model.json",
        "*.md", "*_layout.pdf", "*_span.pdf",
    ]
    for pattern in patterns:
        for f in out_dir.glob(pattern):
            try:
                f.unlink()
                logger.debug("Cleaned: %s", f.name)
            except Exception as e:
                logger.warning("Failed to delete %s: %s", f.name, e)


def run_mineru(pdf_path: Path, text_lang: str | None = None) -> Path:
    """Parse a PDF with MinerU.

    Uses pipeline backend (GPU-accelerated via device-mode: cuda config).
    Only _content_list_v2.json and images/ are kept;
    intermediate files are cleaned immediately.

    Args:
        pdf_path: Path to the PDF file
        text_lang: MinerU 语言提示；None → config.MINERU_LANG（默认 ch）

    Returns:
        Path to the output directory containing _content_list_v2.json etc.

    Raises:
        RuntimeError: If MinerU fails after retries
    """
    import shlex

    text_lang = text_lang or MINERU_LANG

    output_dir = TMP_DIR / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # pipeline backend is ~1.85x faster for research PDFs vs default
    # hybrid-auto-engine, while device-mode: cuda in mineru.json
    # already enables GPU acceleration for model inference.
    cmd = (
        f"conda run -n {MINERU_CONDA_ENV} {MINERU_CMD} "
        f"-p {shlex.quote(str(pdf_path))} "
        f"-o {shlex.quote(str(output_dir))} "
        f"-l {text_lang} "
        f"-b pipeline "
        f"-m {MINERU_METHOD} "
        f"-f true "
        f"-t true"
    )

    logger.info("Running MinerU: %s", cmd)
    if MINERU_MODELSCOPE_CACHE.is_dir():
        logger.info("MinerU 模型缓存: %s", MINERU_MODELSCOPE_CACHE)
    else:
        logger.warning(
            "MinerU 模型缓存目录不存在：%s —— MinerU 会尝试重新下载（务必确认它落在 NAS）",
            MINERU_MODELSCOPE_CACHE,
        )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # initial + retries
        try:
            # Force LOCAL MinerU models (ModelScope cache); never hit HuggingFace.
            _env = dict(os.environ)
            _env["MINERU_MODEL_SOURCE"] = MINERU_MODEL_SOURCE
            # 显式钉住模型缓存位置：这里走 subprocess（非交互非登录 shell），
            # 读不到 ~/.bashrc / ~/.profile 里的 MODELSCOPE_CACHE，不设就会回落到
            # ~/.cache/modelscope 并重新下载 4.6 G（2026-10-16 实测确认）。
            _env.setdefault("MODELSCOPE_CACHE", str(MINERU_MODELSCOPE_CACHE))
            _env.setdefault("HF_HUB_OFFLINE", "1")
            _env.setdefault("TRANSFORMERS_OFFLINE", "1")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=MINERU_TIMEOUT,  # configurable (RAG_MINERU_TIMEOUT)
                env=_env,
            )
            if result.returncode == 0:
                # Find the actual output directory
                # MinerU v3 creates: output_dir/pdf_stem/<version>/
                candidate = None
                for item in output_dir.iterdir():
                    if item.is_dir():
                        for sub in sorted(item.iterdir(), reverse=True):
                            if sub.is_dir() and any(
                                f.name.endswith("_content_list.json")
                                or f.name.endswith("_content_list_v2.json")
                                for f in sub.iterdir()
                            ):
                                candidate = sub
                                break
                            if sub.is_dir():
                                for subsub in sorted(sub.iterdir(), reverse=True):
                                    if subsub.is_dir() and any(
                                        f.name.endswith("_content_list_v2.json")
                                        for f in subsub.iterdir()
                                    ):
                                        candidate = subsub
                                        break
                        if candidate:
                            break
                        # Check single-file output
                        if any(
                            f.name.endswith("_content_list.json")
                            or f.name.endswith("_content_list_v2.json")
                            for f in item.iterdir()
                        ):
                            candidate = item
                            break

                if candidate and any(f.name.endswith("_content_list_v2.json") or f.name.endswith("_content_list.json") for f in candidate.iterdir()):
                    logger.info("MinerU output: %s", candidate)
                    _clean_intermediate_output(candidate)
                    return candidate

                # Fallback: check output_dir directly
                if any(f.name.endswith("_content_list_v2.json") or f.name.endswith("_content_list.json") for f in output_dir.iterdir()):
                    _clean_intermediate_output(output_dir)
                    return output_dir

                # List what we got
                found = list(output_dir.rglob("*_content_list*"))
                logger.warning(
                    "MinerU completed but content list not found at expected location. Found: %s",
                    found,
                )
                if found:
                    fdir = found[0].parent
                    _clean_intermediate_output(fdir)
                    return fdir
                raise RuntimeError(f"MinerU output not found in {output_dir}")

            stderr_tail = result.stderr.strip().split("\n")[-10:]
            logger.error(
                "MinerU failed (attempt %d, rc=%d): %s",
                attempt, result.returncode, "\n".join(stderr_tail),
            )
            last_error = RuntimeError(
                f"MinerU returned {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            logger.error("MinerU timed out (attempt %d)", attempt)
            last_error = TimeoutError("MinerU timed out after 600s")

        if attempt <= MAX_RETRIES:
            logger.info("Retrying MinerU (attempt %d)...", attempt + 1)
        else:
            raise last_error  # type: ignore

    raise last_error  # type: ignore
