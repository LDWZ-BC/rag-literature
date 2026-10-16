"""Pipeline orchestrator — ties together steps 1-7 for PDF processing."""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import INPUT_DIR, TMP_DIR, RAG_JSON_DIR, RUN_DB_DIR, TRASH_DIR

from ..pipeline.step2_mineru import run_mineru
from ..pipeline.step3_metadata import rename_and_move
from ..pipeline.step4_clean import clean_v2_json
from ..pipeline.step6_rag_update import incremental_update
from ..pipeline.step7_cleanup import cleanup_tmp
from ..refs import refresh_references_bib
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil

# ── Pipeline state (checkpoint / resume) ──────────────────────
STATE_FILE = RUN_DB_DIR / "pipeline_state.json"
_state_lock = threading.Lock()


def _load_hashes() -> set:
    """Load the SHA256 dedup registry."""
    f = RUN_DB_DIR / "dedup_hashes.json"
    if f.exists():
        try:
            return set(json.loads(f.read_text()))
        except Exception:
            return set()
    return set()


def _register_hash(h: str):
    """Register a PDF hash as successfully processed."""
    if not h:
        return
    f = RUN_DB_DIR / "dedup_hashes.json"
    try:
        s = _load_hashes()
        s.add(h)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(s)))
    except Exception as e:
        logger.warning("Could not register hash: %s", e)


def _load_state() -> dict:
    """Load pipeline state. Returns empty dict if file does not exist."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"papers": {}, "last_run": ""}


def _save_state(state: dict):
    """Atomically write state file (write .tmp then rename)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.rename(STATE_FILE)


def _mark_step(pdf_basename: str, step: str,
               pdf_hash: str = "", base_name: str = "", error: str = ""):
    """Mark a paper's current processing step in the state file."""
    with _state_lock:
        s = _load_state()
        now = datetime.datetime.now().isoformat()
        if pdf_basename not in s["papers"]:
            s["papers"][pdf_basename] = {"started_at": now}
        e = s["papers"][pdf_basename]
        e["step"] = step
        e["updated_at"] = now
        if pdf_hash:
            e["pdf_hash"] = pdf_hash
        if base_name:
            e["base_name"] = base_name
        if error:
            e["error"] = error
        _save_state(s)


# Serialize GPU access
_mineru_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _archive_input(pdf_path: Path) -> None:
    """成功后把 `.input/` 原件移入 `._trash/`。

    2026-10-14 修：step3 搬进 `DonePDF/` 的是 **MinerU 输出目录里的副本**，`.input/` 原件
    从来不动 → 每跑完一轮 `.input/` 看着「还有内容」，用户反复以为没处理完。原先靠人工
    workflow 步骤清理，现改为自动归档（走 `._trash/`，可恢复）。
    """
    try:
        if pdf_path.parent.resolve() != INPUT_DIR.resolve():
            return                      # 不是从 .input/ 进来的（如单篇指定路径），不动
        if not pdf_path.exists():
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = TRASH_DIR / f"input_done_{ts}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / pdf_path.name
        if dest.exists():
            dest = dest_dir / f"{pdf_path.stem}_{ts}{pdf_path.suffix}"
        shutil.move(str(pdf_path), str(dest))
        logger.info("Input archived: %s → %s", pdf_path.name, dest)
    except Exception as e:
        logger.warning("Input archive failed（原件仍留在 .input/）: %s", e)


def process_pdf(pdf_path: Path, title_override: str | None = None, year_override: str | None = None) -> Optional[dict]:
    """Run the full processing pipeline for a single PDF.

    1. MinerU parse
    2. Extract metadata + rename + move files
    3. Clean JSON（翻译已移除）
    4. (RAG update done separately for batching)

    语言不再是一个独立步骤：MinerU 的语言提示取 config.MINERU_LANG（默认 ch）；
    中/英判定改为从已抽文本的 CJK 占比得出（step3 抽标题、step4 选分句器各算一次）。

    Args:
        pdf_path: Path to the PDF file
        title_override: Optional title from md file (takes priority over MinerU extraction)
        year_override: Optional year from md file

    Returns:
        dict with processing results, or None on failure
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        return None

    pdf_basename = pdf_path.stem

    # Checkpoint: skip already-done papers
    with _state_lock:
        prev_state = _load_state()
        if pdf_basename in prev_state.get("papers", {}):
            prev = prev_state["papers"][pdf_basename]
            if prev.get("step") == "done":
                logger.info("Skipping (checkpoint done): %s", pdf_path.name)
                return {"skipped": True, "reason": "checkpoint_done",
                        "base_name": prev.get("base_name", "")}

    # Dedup by SHA256 hash.
    # NOTE: the hash is registered ONLY after the paper finishes cleanly
    # (see _register_hash at the end). Writing it up front meant a failed run
    # left a stale hash behind, and every later run skipped the paper as
    # "already processed" — this bit us twice (9 UNKN entries, Yoo 2002).
    pdf_hash = ""  # default, set below
    try:
        pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if pdf_hash in _load_hashes():
            logger.info("Skipping (already processed): %s", pdf_path.name)
            return {"skipped": True, "reason": "hash_match", "hash": pdf_hash}
    except Exception as e:
        logger.warning("Hash dedup failed (proceeding anyway): %s", e)

    logger.info("=" * 60)
    logger.info("Processing: %s", pdf_path.name)
    logger.info("=" * 60)

    errors = []

    # Step 2: MinerU parsing (serialized)
    with _mineru_lock:
        logger.info("Step 2: Acquired MinerU lock...")
        try:
            mineru_out = run_mineru(pdf_path)
            logger.info("Step 2: MinerU output → %s", mineru_out)
            _mark_step(pdf_basename, "step2", pdf_hash)
        except Exception as e:
            logger.error("Step 2 failed: %s", e)
            _mark_step(pdf_basename, "failed", pdf_hash, error=str(e))
            return None

    # Step 3: Metadata + rename + move
    try:
        naming_info = rename_and_move(mineru_out, pdf_path, title_override=title_override, year_override=year_override)
        logger.info("Step 3: Named as %s", naming_info["base_name"])
        _mark_step(pdf_basename, "step3", pdf_hash, base_name=naming_info["base_name"])
    except Exception as e:
        logger.error("Step 3 failed: %s", e)
        _mark_step(pdf_basename, "failed", pdf_hash, error=str(e))
        return None

    # Step 4: Clean JSON（翻译已移除，不再需要串行锁）
    json_path = RAG_JSON_DIR / f"{naming_info['base_name']}_v2.json"
    raw_path = RAG_JSON_DIR / f"{naming_info['base_name']}_raw.json"
    jcr_zone = naming_info.get("jcr_zone")
    venue = naming_info.get("venue", "")
    doi = naming_info.get("doi", "")
    structured = {}
    try:
        structured = clean_v2_json(raw_path, jcr_zone=jcr_zone, venue=venue, doi=doi)
        logger.info("Step 4: %d blocks processed", len(structured.get("blocks", [])))
        _mark_step(pdf_basename, "step4", pdf_hash, base_name=naming_info.get("base_name", ""))
        # Delete raw.json after successful cleaning
        if raw_path.exists():
            raw_path.unlink()
    except Exception as e:
        logger.error("Step 4 failed: %s", e)
        errors.append(f"step4: {e}")
        _mark_step(pdf_basename, "failed", pdf_hash,
                   base_name=naming_info.get("base_name", ""), error=str(e))

    # Step 5 已移除：cnRead 不再随 pipeline 生成（需要时另行实现，不属核心流程）

    # Count sentences for summary
    sentence_count = sum(
        len(b["sentences"])
        for b in structured.get("blocks", [])
        if b.get("type") == "text"
    )

    if errors:
        # 有步骤失败：标 failed 且**不登记 hash**，下次运行会重试（而不是被静默跳过）
        logger.warning("Completed with errors: %s -> %s", pdf_path.name, errors)
        _mark_step(pdf_basename, "failed", pdf_hash,
                   base_name=naming_info.get("base_name", ""), error="; ".join(str(e) for e in errors))
    else:
        _mark_step(pdf_basename, "done", pdf_hash, base_name=naming_info.get("base_name", ""))
        _register_hash(pdf_hash)
        _archive_input(pdf_path)

    return {
        "pdf": pdf_path.name,
        "base_name": naming_info["base_name"],
        "sentences": sentence_count,
        "errors": errors,
    }


def process_all(update_rag: bool = True):
    """Process all PDFs in input/ directory.

    Args:
        update_rag: Whether to update vector DB after processing
    """
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        logger.info("No PDFs found in %s", INPUT_DIR)
        return

    # Filter out @eaDir and hidden files
    pdfs = [p for p in pdfs if not p.name.startswith(".") and "@eaDir" not in str(p)]

    logger.info("Found %d PDFs to process", len(pdfs))
    results = []

    max_workers = int(os.environ.get("RAG_PIPELINE_WORKERS", "2"))
    logger.info("Starting parallel processing with %d workers", max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_pdf, pdf_path): pdf_path
            for pdf_path in pdfs
        }
        for future in as_completed(future_map):
            pdf_path = future_map[future]
            try:
                result = future.result(timeout=7200)
                if result:
                    results.append(result)
                    logger.info("Completed: %s", pdf_path.name)
            except Exception as e:
                logger.error("Fatal error processing %s: %s", pdf_path.name, e)
                results.append({"pdf": pdf_path.name, "errors": [str(e)]})

    # Step 6: RAG incremental update
    if update_rag:
        logger.info("Step 6: Updating RAG vector database...")
        try:
            rag_stats = incremental_update()
            logger.info("RAG stats: %s", rag_stats)
        except Exception as e:
            logger.error("RAG update failed: %s", e)

    # Step 6b: Refresh the unified references.bib (cheap re-merge; incremental-safe)
    logger.info("Step 6b: Refreshing unified bibliography (references.bib)...")
    try:
        n_refs = refresh_references_bib()
        logger.info("references.bib: %d entries", n_refs)
    except Exception as e:
        logger.warning("Bibliography refresh failed: %s", e)

    # （原有末尾「全库产物名补扫」已移除：命名在 step3 完成；
    #   utils.files.sync_all_artifacts 保留为手动维护工具，改年份后单独跑一次）

    # Step 6c: 刷新只读检索缓存（preset/问答的检索走它，不刷新就会静默停在旧语料）
    logger.info("Step 6c: Refreshing read-only retrieval index...")
    try:
        from ..tools import export_index
        if export_index.check(export_index.DEFAULT_OUT) != 0:
            n = export_index.export(export_index.DEFAULT_OUT)
            logger.info("Retrieval index refreshed: %d records", n)
        else:
            logger.info("Retrieval index already up to date")
    except Exception as e:
        logger.warning("Retrieval index refresh failed (查询层会停在旧语料): %s", e)


    # Step 8: Cleanup tmp/
    logger.info("Step 8: Cleaning up tmp/")
    try:
        cleanup_tmp()
    except Exception as e:
        logger.error("Cleanup failed: %s", e)

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline complete. Results:")
    for r in results:
        status = "OK" if not r.get("errors") else f"ERRORS: {r['errors']}"
        logger.info("  %s → %s (%d sentences) %s",
                 r.get("pdf", "?"), r.get("base_name", "?"),
                 r.get("sentences", 0), status)

    return results
if __name__ == "__main__":
    process_all(update_rag=True)
