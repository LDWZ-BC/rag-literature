"""llama-cli process wrapper for one-shot inference + HTTP server fallback."""

import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

from ..config import LLAMA_CLI, LLAMA_CLI_ARGS_COMMON, LLAMA_GENERATION_SERVER_URL, LLAMA_SERVER, MODEL_GENERATION

logger = logging.getLogger(__name__)

# Patterns to strip from llama-cli output
_PROGRESS_ARTIFACT_RE = re.compile(r'[\|\-\/\\]\x08')
_BACKSPACE_RE = re.compile(r'[\x08]')
_MEMORY_BREAKDOWN_RE = re.compile(r'common_memory_breakdown_print:.*$', re.MULTILINE)

# Lines that should be entirely removed
_META_LINE_RE = re.compile(
    r'^\s*\[?\s*(Prompt\s*:|Generation\s*:|'
    r'Exiting\.\.\.|'
    r'>\s).*',
    re.IGNORECASE,
)


def llama_generate(
    model_path: Path,
    prompt: str,
    n_predict: int = 512,
    ctx_size: int = 2048,
    temperature: float = 0.1,
    extra_args: Optional[list[str]] = None,
    timeout: int = 300,
) -> str:
    """Run llama-cli one-shot and return generated text.

    Args:
        model_path: Path to GGUF model
        prompt: Input prompt
        n_predict: Max tokens to generate
        ctx_size: Context size
        temperature: Sampling temperature (0.1 for translation)
        extra_args: Additional CLI arguments
        timeout: Subprocess timeout in seconds

    Returns:
        Generated text, stripped of all diagnostic artifacts.
    """
    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx_size),
        "--temp", str(temperature),
        *LLAMA_CLI_ARGS_COMMON,
    ]
    if extra_args:
        cmd.extend(extra_args)

    import os as _os

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        logger.error("llama-cli timed out after %ds", timeout)
        raise

    if result.returncode != 0:
        stderr_tail = result.stderr.strip().split("\n")[-5:]
        logger.error(
            "llama-cli failed (rc=%d): %s",
            result.returncode,
            "\n".join(stderr_tail),
        )
        raise RuntimeError(f"llama-cli exited with code {result.returncode}")

    return _parse_output(result.stdout, prompt)


def _parse_output(raw: str, prompt: str) -> str:
    """Extract model-generated text from llama-cli stdout.

    Strategy: find the 'Chinese/English translations:' marker in the echoed
    prompt and take everything after it (excluding meta lines).
    This is much more robust than line-by-line prompt matching.
    """
    import re as _re

    text = _PROGRESS_ARTIFACT_RE.sub('', raw)
    text = _BACKSPACE_RE.sub('', text)
    text = _MEMORY_BREAKDOWN_RE.sub('', text)

    # Find the translations header and split
    marker_match = _re.search(
        r'(?:Chinese|English)\s*(?:translations\s*)?\s*:\s*',
        text, _re.IGNORECASE
    )
    if marker_match:
        # Take everything after the marker
        after = text[marker_match.end():]
    else:
        # Fallback: use full text
        after = text

    # Filter meta/diagnostic lines
    result_lines = []
    for line in after.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if _META_LINE_RE.match(stripped):
            continue
        result_lines.append(stripped)

    return '\n'.join(result_lines).strip()


def llama_generate_via_server(
    prompt: str,
    n_predict: int = 512,
    temperature: float = 0.1,
    server_url: str = LLAMA_GENERATION_SERVER_URL,
    timeout: int = 300,
) -> str:
    """Run generation via llama-server HTTP API.

    Args:
        prompt: Input prompt
        n_predict: Max tokens to generate
        temperature: Sampling temperature
        server_url: Full URL to /completion endpoint
        timeout: Request timeout in seconds

    Returns:
        Generated text, stripped of diagnostic artifacts.

    Raises:
        RuntimeError: If server request fails
    """
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json.loads(resp.read()).get("content", "")
    except Exception as e:
        raise RuntimeError(f"llama-server request failed: {e}") from e

    return _parse_output(result, prompt)
