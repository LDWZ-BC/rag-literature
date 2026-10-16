"""Step 4: Clean MinerU JSON, translate (EN only), preserve block structure.

Key improvements:
- Discards header/footer/footnote page decoration blocks
- Uses page_idx from MinerU for page tracking
- Normalizes \\\\ → \\ and Greek chars → LaTeX
- Sentence splitting respects $$ and $ math boundaries
- Math regions are replaced with placeholders during translation,
  then restored afterward for better translation quality.
"""

import json
import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional

from ..config import RAG_JSON_DIR
from ..utils.textlang import looks_chinese

logger = logging.getLogger(__name__)

# ── Block types to discard ────────────────────────────────────────────────────
_SKIP_BLOCK_TYPES = {
    "header", "footer", "page_footnote", "page_number", "footnote",
    "marginal_note", "note",
}

# ── Greek and math symbol fixes ───────────────────────────────────────────────

MATH_SYMBOL_MAP = {
    "μ": r"$\mu$", "σ": r"$\sigma$", "Σ": r"$\Sigma$",
    "π": r"$\pi$", "Π": r"$\Pi$", "θ": r"$\theta$",
    "Θ": r"$\Theta$", "α": r"$\alpha$", "β": r"$\beta$",
    "γ": r"$\gamma$", "Γ": r"$\Gamma$", "δ": r"$\delta$",
    "Δ": r"$\Delta$", "ε": r"$\varepsilon$", "λ": r"$\lambda$",
    "Λ": r"$\Lambda$", "ω": r"$\omega$", "Ω": r"$\Omega$",
    "φ": r"$\varphi$", "Φ": r"$\Phi$", "ψ": r"$\psi$",
    "Ψ": r"$\Psi$", "τ": r"$\tau$", "η": r"$\eta$",
    "ν": r"$\nu$", "ζ": r"$\zeta$", "ξ": r"$\xi$",
    "ρ": r"$\rho$", "χ": r"$\chi$", "κ": r"$\kappa$",
    "∞": r"$\infty$", "±": r"$\pm$", "×": r"$\times$",
    "÷": r"$\div$", "≈": r"$\approx$", "≠": r"$\neq$",
    "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\to$",
    "←": r"$\leftarrow$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "∂": r"$\partial$", "∇": r"$\nabla$", "∫": r"$\int$",
    "∑": r"$\sum$", "∏": r"$\prod$", "√": r"$\sqrt{}$",
    "⟨": r"$\langle$", "⟩": r"$\rangle$", "ħ": r"$\hbar$",
    "⊥": r"$\perp$", "⊕": r"$\oplus$", "⊗": r"$\otimes$",
}


def fix_math_symbols(text: str) -> str:
    """Replace isolated Unicode math symbols with LaTeX, normalize backslashes."""
    # Fix double backslashes → single
    text = text.replace("\\\\", "\\")
    # Replace symbols not inside existing $...$ boundaries
    for symbol, latex in MATH_SYMBOL_MAP.items():
        text = _replace_outside_latex(text, symbol, latex)
    return text


def _replace_outside_latex(text: str, old: str, new: str) -> str:
    """Replace old with new only when not inside $ delimiters."""
    result = []
    in_math = False
    i = 0
    while i < len(text):
        if text[i : i + 2] == "$$":
            in_math = not in_math
            result.append("$$")
            i += 2
            continue
        if text[i] == "$" and not (i > 0 and text[i - 1] == "\\"):
            in_math = not in_math
            result.append("$")
            i += 1
            continue
        if not in_math and text[i : i + len(old)] == old:
            result.append(new)
            i += len(old)
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


# ── Math protection for translation ───────────────────────────────────────────

def _protect_math(text: str) -> tuple[str, list[str]]:
    """Replace all $...$ and $$...$$ regions with ASCII placeholders.

    Uses [[MATH0]], [[MATH1]] etc. — double-bracket ASCII tokens that
    models reliably preserve and won't trigger UTF-8 decode errors.
    """
    placeholders = []

    def _repl(m):
        placeholders.append(m.group(0))
        return f"[[MATH{len(placeholders) - 1}]]"

    # Match $$...$$ (non-greedy across lines) and $...$ (single line)
    result = re.sub(r'\$\$(.+?)\$\$', _repl, text, flags=re.DOTALL)
    result = re.sub(r'\$(.+?)\$', _repl, result)
    return result, placeholders


def _restore_math(text: str, placeholders: list[str]) -> str:
    """Restore [[MATHN]] placeholders back to original math expressions."""
    import re as _re
    for i, ph in enumerate(placeholders):
        marker = f"[[MATH{i}]]"
        text = text.replace(marker, ph)
        # Handle possible spacing variants from model (using lambda to avoid \m escape issues)
        text = _re.sub(
            rf"\[\[\s*MATH\s*{i}\s*\]\]",
            lambda m, p=ph: p,
            text,
        )
    return text


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Split English text into sentences without breaking inside $...$ math.

    Strategy: protect math regions, split on sentence boundaries, restore math.
    """
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # Protect math regions
    clean, math_phs = _protect_math(text)

    # Protect common abbreviations before sentence splitting
    clean = re.sub(r"\b([A-Z][a-z]?)\.", r"\1<DOT>", clean)
    for abbr in ["et al", "fig", "eq", "sec", "ref", "cf", "vs", "no",
                 "vol", "suppl", "approx", "dept"]:
        clean = re.sub(rf"\b{re.escape(abbr)}\.", rf"{abbr}<DOT>", clean, flags=re.IGNORECASE)
    for abbr in ["ie", "eg"]:
        clean = re.sub(rf"\b{abbr}\.", rf"{abbr}<DOT>", clean)
    # Common journal abbreviations that should NOT cause sentence breaks
    for abbr in ["rev", "phys", "sci", "lett", "chem", "appl", "acad",
                 "natl", "proc", "trans", "adv", "int", "conf", "symp",
                 "soc", "j", "ser", "ieee", "nature", "springer",
                 "elsevier", "wiley", "am", "ann", "curr", "eur",
                 "front", "ieee", "inst", "mater", "math", "med",
                 "opt", "phil", "plos", "pnas", "polym", "prog",
                 "pure", "stat", "surf", "syst", "tech", "theor",
                 "acta", "bio", "cell", "eng", "gen", "genom",
                 "immunol", "mol", "nanotechnol", "neurosci",
                 "pharmacol", "physiol", "plant"]:
        clean = re.sub(rf"\b{re.escape(abbr)}\.", rf"{abbr}<DOT>", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bi\.e\.", r"i<DOT>e<DOT>", clean)
    clean = re.sub(r"\be\.g\.", r"e<DOT>g<DOT>", clean)

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)

    # Handle trailing leftover
    result = []
    for s in sentences:
        s = s.replace("<DOT>", ".").strip()
        if len(s) > 1:
            result.append(_restore_math(s, math_phs))

    return result


def split_chinese_sentences(text: str) -> list[str]:
    """Split Chinese text on 。！？without breaking inside math."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    clean, math_phs = _protect_math(text)
    parts = re.split(r'(?<=[。！？；])\s*', clean)
    result = []
    for p in parts:
        p = p.strip()
        if len(p) > 1:
            result.append(_restore_math(p, math_phs))
    return result


# ── Main processing ───────────────────────────────────────────────────────────

def clean_v2_json(
    json_path: Path,
    jcr_zone: str | None = None,
    venue: str | None = None,
    doi: str | None = None,
) -> dict:
    """Clean a MinerU v3 JSON and return structured blocks.

    2026-10-13：翻译已移除（`zh` 一律留空）。语言改为在每个文本块上按 CJK 占比判定，
    只用于**选分句器**；文档级 `meta.is_chinese` 也由同一占比在末尾得出。

    Args:
        json_path: Path to raw content list JSON

    Returns:
        dict with 'meta' and 'blocks'
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_path = Path(json_path)
    json_path = Path(json_path)
    json_path = Path(json_path)
    base_name = json_path.stem.replace("_raw", "").replace("_v2", "")
    book_name = base_name.rsplit("_", 1)[0] if "_" in base_name else base_name
    year = base_name.rsplit("_", 1)[1] if "_" in base_name else ""

    blocks = data if isinstance(data, list) else []

    structured_blocks = []
    text_block_indices = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        btype = str(block.get("type", "")).lower()

        # Skip decorative blocks
        if btype in _SKIP_BLOCK_TYPES:
            continue

        page = block.get("page_idx", 0)

        if btype in ("text", "title", "paragraph", "header"):
            text = block.get("text", "") or ""
            if not text:
                spans = block.get("spans", [])
                text = "".join(
                    s.get("content", "") or s.get("text", "")
                    for s in spans if isinstance(s, dict)
                )
            if not text.strip():
                continue

            text = unicodedata.normalize("NFC", text)
            text = fix_math_symbols(text)

            _zh = looks_chinese(text)
            if _zh:
                sents = split_chinese_sentences(text)
            else:
                sents = split_into_sentences(text)

            sent_entries = []
            for s in sents:
                s = s.strip()
                if (_zh and len(s) < 2) or (not _zh and len(s) < 3):
                    continue
                sent_entries.append({"en": s, "zh": ""})

            if sent_entries:
                structured_blocks.append({
                    "type": "text",
                    "page": page,
                    "sentences": sent_entries,
                })
                text_block_indices.append(len(structured_blocks) - 1)

        elif btype == "image":
            structured_blocks.append({
                "type": "image",
                "page": page,
                "path": block.get("img_path", ""),
                "caption": block.get("image_caption", block.get("caption", [])),
            })

        elif btype == "equation":
            structured_blocks.append({
                "type": "equation",
                "page": page,
                "latex": block.get("text", block.get("text_format", "")),
            })

        elif btype == "list":
            items = block.get("list_items", [])
            if items:
                combined = " ".join(items)
                if looks_chinese(combined):
                    sents = split_chinese_sentences(combined)
                else:
                    sents = split_into_sentences(combined)
                sent_entries = [
                    {"en": s.strip(), "zh": ""}
                    for s in sents if len(s.strip()) > 2
                ]
                if sent_entries:
                    structured_blocks.append({
                        "type": "text",
                        "page": page,
                        "sentences": sent_entries,
                    })
                    text_block_indices.append(len(structured_blocks) - 1)
        else:
            logger.debug("Skipping unrecognized block type: %s", btype)

    logger.info(
        "Extracted %d blocks (%d text) from %s",
        len(structured_blocks), len(text_block_indices), json_path.name,
    )

    # 文档级语言标志：由已抽文本的 CJK 占比得出（无独立检测步骤、无额外 I/O）
    _all_text = " ".join(
        _s.get("en", "")
        for _b in structured_blocks if _b.get("type") == "text"
        for _s in _b.get("sentences", [])
    )
    is_chinese = looks_chinese(_all_text)

    # ── Save structured output to _v2.json ─────────────────────────────────
    v2_path = Path(str(json_path).replace("_raw.json", "_v2.json"))

    output = {
        "meta": {
            "book": book_name,
            "year": year,
            "is_chinese": is_chinese,
            "jcr_zone": jcr_zone or "",
            "venue": venue or "",
            "doi": doi or "",
        },
        "blocks": structured_blocks,
    }

    from ..utils.files import write_json as _write_json
    _write_json(v2_path, output)
    logger.info(
        "Saved structured JSON: %s (%d blocks)", v2_path.name, len(structured_blocks)
    )

    return output
