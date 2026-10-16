"""Translation client using HY-MT models via llama-cli or HTTP server.

Supports efficient batch translation: multiple sentences in one call
to avoid repeated model loading overhead.
"""

import re
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

from ..models.llama import llama_generate, llama_generate_via_server
from ..models.manager import server_manager
from ..config import (
    MODEL_TRANSLATE_7B,
    MODEL_TRANSLATE_1_5B,
    TRANSLATE_CTX,
    TRANSLATE_N_PREDICT,
    LLAMA_SERVER_URL,
)

logger = logging.getLogger(__name__)

# Batch size: how many sentences to pack into one llama-cli call
BATCH_SIZE = 10
SERVER_BATCH_SIZE = 50
BATCH_N_PREDICT_PER_SENTENCE = 384  # tokens per sentence output
SERVER_N_PREDICT_PER_SENT = 120
SERVER_MAX_CTX = 3900
SERVER_CTX_LIMIT = 4096
SERVER_URL = "http://127.0.0.1:8081/completion"
SERVER_TIMEOUT = 300


def _est_tokens(text: str) -> int:
    return len(text) // 3 + 1


def _make_prompt(batch: list[str]) -> str:
    lines_str = "\n".join(f"[{i}] {t.strip()}" for i, t in enumerate(batch, 1))
    return (
        "Translate each of the following English sentences to Chinese. "
        "Output one translation per line, keeping the [N] numbering. "
        "Do not add any extra text.\n\n"
        f"{lines_str}\n\nChinese translations:"
    )


def _make_safe_batches(
    texts: list[str],
    max_batch: int = SERVER_BATCH_SIZE,
) -> list[list[str]]:
    """Split texts into token-aware batches avoiding n_predict starvation.

    Each batch guarantees enough generation tokens for all its sentences.
    """
    batches = []
    i = 0
    while i < len(texts):
        # Start with max_batch, shrink if needed
        hi = min(i + max_batch, len(texts))
        while hi > i:
            batch = texts[i:hi]
            prompt = _make_prompt(batch)
            prompt_toks = _est_tokens(prompt)
            available = SERVER_CTX_LIMIT - prompt_toks - 200
            needed = len(batch) * SERVER_N_PREDICT_PER_SENT
            if available >= needed:
                break
            hi -= 1
        batch = texts[i:hi]
        batches.append(batch)
        i = hi
    if batches:
        total_safe = sum(len(b) for b in batches)
        if total_safe != len(texts):
            logger.warning(
                "Safe batching: %d/%d sentences assigned (%d batches, avg %.1f/batch)",
                total_safe, len(texts), len(batches), total_safe / len(batches),
            )
    return batches


def translate_batch_server(
    texts: list[str],
    batch_size: int = SERVER_BATCH_SIZE,
) -> list[str]:
    """Translate using llama-server HTTP API (model stays loaded in GPU).

    Auto-starts server if not running; auto-shuts down after 10 min idle.
    """
    if not texts:
        return []
    # Ensure server is running
    if not server_manager.ensure_running():
        logger.warning("llama-server unavailable")
        return []

    all_results = []
    safe_batches = _make_safe_batches(texts, batch_size)
    for batch in safe_batches:
        prompt = _make_prompt(batch)
        prompt_toks = _est_tokens(prompt)
        max_gen = max(SERVER_N_PREDICT_PER_SENT, SERVER_MAX_CTX - prompt_toks - 200)
        n_predict = min(len(batch) * SERVER_N_PREDICT_PER_SENT, max_gen)
        if len(batch) * SERVER_N_PREDICT_PER_SENT > max_gen:
            logger.warning(
                "Batch of %d sents: n_predict=%d, needed=%d — may truncate",
                len(batch), n_predict, len(batch) * SERVER_N_PREDICT_PER_SENT,
            )
        results = None
        for attempt in range(3):
            try:
                body = json.dumps({
                    "prompt": prompt, "n_predict": n_predict, "temperature": 0.1
                }).encode()
                req = urllib.request.Request(
                    SERVER_URL, data=body,
                    headers={"Content-Type":"application/json"}
                )
                resp = urllib.request.urlopen(req, timeout=SERVER_TIMEOUT)
                content = json.loads(resp.read()).get("content", "")
                results = _parse_numbered_response(content, len(batch), batch)
                server_manager.mark_activity()
                break
            except Exception as e:
                logger.warning(f"Server translate attempt {attempt+1}/3: {e}")
                import time
                time.sleep(2)

        if results is None:
            logger.warning(
                "Server failed for batch of %d sents",
                len(batch),
            )
            results = translate_batch(batch, source_lang="en")
        all_results.extend(results)
    return all_results

def translate_en2zh(text: str, model_path: Path = MODEL_TRANSLATE_1_5B) -> str:
    """Translate a single English text to Chinese using server API."""
    if not text.strip():
        return ""
    prompt = (
        "Translate the following English to Chinese. "
        "Output ONLY the Chinese translation, nothing else.\n\n"
        f"English: {text.strip()}\n\nChinese:"
    )
    ensure_server = server_manager.ensure_running()
    if ensure_server:
        try:
            result = llama_generate_via_server(
                prompt=prompt,
                n_predict=TRANSLATE_N_PREDICT,
                temperature=0.1,
                server_url=LLAMA_SERVER_URL,
            )
            server_manager.mark_activity()
            return _clean_translation(result)
        except Exception:
            logger.warning("Server request failed")
    else:
        logger.warning("Translation server unavailable")
    result = llama_generate(
        model_path=model_path,
        prompt=prompt,
        n_predict=TRANSLATE_N_PREDICT,
        ctx_size=TRANSLATE_CTX,
        temperature=0.1,
    )
    return _clean_translation(result)


def translate_zh2en(text: str, model_path: Path = MODEL_TRANSLATE_1_5B) -> str:
    """Translate a single Chinese text to English using server API."""
    if not text.strip():
        return ""
    prompt = (
        "Translate the following Chinese to English. "
        "Output ONLY the English translation, nothing else.\n\n"
        f"Chinese: {text.strip()}\n\nEnglish:"
    )
    ensure_server = server_manager.ensure_running()
    if ensure_server:
        try:
            result = llama_generate_via_server(
                prompt=prompt,
                n_predict=TRANSLATE_N_PREDICT,
                temperature=0.1,
                server_url=LLAMA_SERVER_URL,
            )
            server_manager.mark_activity()
            return _clean_translation(result)
        except Exception:
            logger.warning("Server request failed")
    else:
        logger.warning("Translation server unavailable")
    result = llama_generate(
        model_path=model_path,
        prompt=prompt,
        n_predict=TRANSLATE_N_PREDICT,
        ctx_size=TRANSLATE_CTX,
        temperature=0.1,
    )
    return _clean_translation(result)


def translate_batch(
    texts: list[str],
    source_lang: str = "en",
    model_path: Path = MODEL_TRANSLATE_7B,
) -> list[str]:
    """Translate a batch of texts efficiently.

    Groups sentences into sub-batches and translates each sub-batch
    in a single llama-cli call, avoiding per-sentence model reload.

    Args:
        texts: List of sentences to translate
        source_lang: 'en' or 'zh'
        model_path: Model to use

    Returns:
        List of translated sentences in the same order
    """
    if not texts:
        return []

    all_results = []
    total = len(texts)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = texts[batch_start:batch_end]

        logger.info(
            "Translating batch %d-%d/%d (%d sentences)...",
            batch_start + 1, batch_end, total, len(batch),
        )

        try:
            batch_results = _translate_sub_batch(batch, source_lang, model_path)
            all_results.extend(batch_results)
        except Exception as e:
            logger.error("Batch translation failed: %s", e)
            # Fallback: individual translation for this batch
            logger.info("Falling back to individual translation...")
            for text in batch:
                try:
                    if source_lang == "en":
                        r = translate_en2zh(text, model_path)
                    else:
                        r = translate_zh2en(text, model_path)
                    all_results.append(r)
                except Exception as e2:
                    all_results.append(f"[TRANSLATION_ERROR: {e2}]")

    return all_results


def _translate_sub_batch(
    texts: list[str],
    source_lang: str,
    model_path: Path,
) -> list[str]:
    """Translate a sub-batch of sentences in one llama-cli call.

    Uses [N] numbering for reliable round-trip parsing.
    """
    sl = "English" if source_lang == "en" else "Chinese"
    tl = "Chinese" if source_lang == "en" else "English"

    lines = []
    for i, text in enumerate(texts, 1):
        lines.append(f"[{i}] {text.strip()}")

    prompt = (
        f"Translate each of the following {sl} sentences to {tl}. "
        f"Output one translation per line, keeping the [N] numbering. "
        f"Do not add any extra text.\n\n"
        + "\n".join(lines)
        + f"\n\n{tl} translations:"
    )

    n_predict = len(texts) * BATCH_N_PREDICT_PER_SENTENCE
    ctx = max(TRANSLATE_CTX, len(prompt.split()) * 2)

    result = llama_generate(
        model_path=model_path,
        prompt=prompt,
        n_predict=n_predict,
        ctx_size=ctx,
        temperature=0.1,
        timeout=300,
    )

    return _parse_numbered_response(result, len(texts), texts)


def _parse_numbered_response(response: str, expected_count: int, originals: list[str] | None = None) -> list[str]:
    """Parse a numbered translation response like '[1] text\n[2] text'.

    Uses numbered regex matching with line-by-line fallback.
    """
    import re as _re

    # Strip artifacts (no DOTALL — memory breakdown is on a single line)
    response = _re.sub(r'common_memory_breakdown_print:.*', '', response)
    response = _re.sub(r'\[?\s*Prompt\s*:[\d.]+\s*t/s\s*\|\s*Generation\s*:[\d.]+\s*t/s\s*\]?\s*', '', response)
    response = _re.sub(r'^\s*Exiting\.\.\.\s*$', '', response, flags=_re.MULTILINE)
    response = response.strip()

    results = [""] * expected_count
    overwrites = []

    # Strategy 1: numbered [N] regex, anchored to line start (prevents [N]
    # inside translation content — e.g. "Ref. [3]" — from being mis-parsed)
    pattern = _re.compile(r"^\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", _re.DOTALL | _re.MULTILINE)
    matches = pattern.findall(response)
    for num_str, text in matches:
        try:
            idx = int(num_str) - 1
            if 0 <= idx < expected_count:
                text = text.strip()
                text = _re.sub(r'[\|\-\/\\]+\s*$', '', text)
                text = _re.sub(r'\s*common_memory_breakdown.*', '', text, flags=_re.DOTALL)
                if text:
                    if results[idx]:
                        overwrites.append(idx)
                    results[idx] = text
        except ValueError:
            continue

    # Strategy 2: line-by-line fill for any remaining gaps
    # Use actual [N] numbering from lines, not positional index
    raw_lines = [l.strip() for l in response.split("\n") if l.strip() and not _re.match(r'^\s*$', l)]
    numbered_lines = [
        (int(m.group(1)), _re.sub(r"^\[\d+\]\s*", "", l).strip())
        for l in raw_lines
        if (m := _re.match(r"^\[(\d+)\]\s+", l))
    ]

    for num, text in numbered_lines:
        idx = num - 1
        if 0 <= idx < expected_count:
            text = _re.sub(r'\s*common_memory_breakdown.*', '', text, flags=_re.DOTALL)
            if text and not results[idx]:
                if results[idx]:
                    overwrites.append(idx)
                results[idx] = text

    if overwrites:
        logger.warning(
            "Duplicate [N] numbering: %d overwrites in %d-elt response. "
            "Indices: %s. Model likely restarted numbering mid-batch.",
            len(overwrites), expected_count,
            ",".join(str(i) for i in sorted(set(overwrites))[:20]),
        )

    # Validate results have Chinese content; reset to original if missing
    # Also guard against length-ratio hallucinations (short EN → long CN)
    cleared = []
    length_cleared = []
    for i in range(expected_count):
        r = results[i]
        if r and not any('\u4e00' <= c <= '\u9fff' for c in r):
            cleared.append(i)
            results[i] = originals[i] if originals else ""
        elif originals and len(r) > len(originals[i]) * 5 and len(r) > 80:
            length_cleared.append(i)
            results[i] = originals[i]
    if cleared:
        logger.warning(
            "Falling back %d/%d results without Chinese chars (indices: %s)",
            len(cleared), expected_count,
            ",".join(str(i) for i in cleared[:20]),
        )
    if length_cleared:
        logger.warning(
            "Length-ratio fallback for %d/%d results (indices: %s) — short EN fragment expanded to long hallucination",
            len(length_cleared), expected_count,
            ",".join(str(i) for i in length_cleared[:20]),
        )

    return results


def _clean_translation(text: str) -> str:
    """Clean up model output artifacts."""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    for prefix in ["Translation:", "Chinese:", "English:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text
