"""Step 3: Extract metadata from MinerU JSON and rename files.

Key improvements:
- Year lookup via Semantic Scholar + EasyScholar APIs (with API keys)
- Filename sanitization (strip forbidden chars)
- Deletes _raw.json after step4 succeeds (done in orchestrator)
"""

import shutil
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests

from ..config import (
    RAG_JSON_DIR,
    DONE_PDF_DIR,
    IMAGES_DIR,
    KEEP_IMAGES,
    SEMANTIC_SCHOLAR_API,
    SEMANTIC_SCHOLAR_KEY,
    EASYSCHOLAR_API,
    EASYSCHOLAR_KEY,
    API_CALL_INTERVAL,
    MAX_RETRIES,
    JCR_ENABLED,
)
from ..models.journal_tier import lookup_jcr_zone
from ..utils.naming import sanitize_name, extract_year, build_filename
from ..utils.textlang import looks_chinese
from ..utils.files import safe_move

logger = logging.getLogger(__name__)

# Filename forbidden characters (Windows + cross-platform)
_FORBIDDEN_CHARS_RE = re.compile(r'[\\/:*?"<>|]')
# Track last API call time for rate limiting
_last_api_call = 0.0


def _rate_limit():
    """Ensure minimum interval between API calls."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < API_CALL_INTERVAL:
        time.sleep(API_CALL_INTERVAL - elapsed)
    _last_api_call = time.time()


def lookup_year_apis(title: str) -> str:
    """Look up publication year via Semantic Scholar and EasyScholar APIs.

    Returns year string, or empty string if not found.
    """
    # ── Semantic Scholar ─────────────────────────────────────────────────
    _rate_limit()
    try:
        url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
        params = {
            "query": title,
            "limit": 1,
            "fields": "title,year",
            "api_key": SEMANTIC_SCHOLAR_KEY,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            papers = data.get("data", [])
            if papers and papers[0].get("year"):
                year = str(papers[0]["year"])
                logger.info("Semantic Scholar found year: %s", year)
                return year
        elif resp.status_code != 429:
            logger.debug("Semantic Scholar returned status %d", resp.status_code)
    except Exception as e:
        logger.debug("Semantic Scholar year lookup error: %s", e)

    # ── EasyScholar ───────────────────────────────────────────────────────
    _rate_limit()
    try:
        url = EASYSCHOLAR_API
        params = {
            "secretKey": EASYSCHOLAR_KEY,
            "publicationName": title,
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 200:
                rank_data = data.get("data", {})
                # EasyScholar gives journal ranking, not paper year
                # But the API might also work with paper titles
                logger.debug("EasyScholar returned ranking data for: %s", title)
    except Exception as e:
        logger.debug("EasyScholar year lookup error: %s", e)

    return ""


def sanitize_filename_part(name: str) -> str:
    """Clean a filename part for filesystem safety."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"\s+", " ", name).strip()


# Known section headings that are NOT real paper titles
_SECTION_HEADING_BLACKLIST = frozenset({
    "annotation", "comment", "abstract", "introduction", "conclusion",
    "results", "discussion", "methods", "method", "materials and methods",
    "references", "acknowledgments", "acknowledgements", "supplementary",
    "supplementary material", "supporting information",
    "research report", "report", "review", "overview", "summary",
    "editorial", "letter", "communication", "article", "note",
    "introduction.", "discussion.", "results.", "methods.", "method.",
})


def _is_likely_section_heading(title: str) -> bool:
    """Check if extracted title is a generic section heading rather than a real paper title."""
    t = title.strip().lower().rstrip(".")
    if t in _SECTION_HEADING_BLACKLIST:
        return True
    if len(t) < 5:
        return True
    if len(t.split()) == 1 and len(t) < 10:
        return True
    return False


def _content_looks_chinese(json_path: Path, cap: int = 20_000) -> bool:
    """从 MinerU 的 content list 抽一段文本样本判定语言。

    2026-10-13：原先由 step1「语言检测」步骤提供（且只看**文件名**）；该步骤已移除，
    改为在需要处从**已抽出的内容**判定 —— 不看文件名、不额外读 PDF。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if isinstance(data, list):
        blocks = data
    elif isinstance(data, dict):
        blocks = [b for v in data.values() if isinstance(v, list) for b in v]
    else:
        blocks = []
    buf: list[str] = []
    n = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        txt = b.get("text") or ""
        if not txt:
            txt = "".join(
                (s.get("content") or s.get("text") or "")
                for s in (b.get("spans") or []) if isinstance(s, dict)
            )
        if txt:
            buf.append(txt)
            n += len(txt)
            if n >= cap:
                break
    return looks_chinese(" ".join(buf))


def extract_title_from_mineru_json(json_path: Path, is_chinese: bool | None = None) -> str | None:
    """Extract likely title from MinerU's content list JSON.

    For Chinese PDFs: looks for "题 目" pattern (OCR may split chars with spaces).
    For English PDFs: uses text_level=1 or first title/text block.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = []
    if isinstance(data, list):
        blocks = data
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                blocks.extend(val)

    if not blocks:
        return None

    if is_chinese is None:
        # 语言从抽出的内容判定（不再有独立的语言检测步骤）
        _sample = ""
        for _b in blocks[:200]:
            if isinstance(_b, dict):
                _sample += (_b.get("text") or "") + "\n"
            if len(_sample) >= 20_000:
                break
        is_chinese = looks_chinese(_sample)

    if is_chinese:
        first_page_text = ""
        for block in blocks[:200]:
            if not isinstance(block, dict):
                continue
            text = block.get("text", "") or ""
            if text.strip():
                first_page_text += text + "\n"

        patterns = [
            r'题\s*目\s*[：:\s]+(.+)',
            r'论文题\s*目\s*[：:\s]+(.+)',
            r'课题名\s*称\s*[：:\s]+(.+)',
            r'论文标\s*题\s*[：:\s]+(.+)',
        ]
        for pat in patterns:
            m = re.search(pat, first_page_text)
            if m:
                title = m.group(1).strip()
                title = title.split("\n")[0]
                title = re.sub(r"\s+", " ", title).strip()
                if len(title) > 3:
                    logger.info("Found Chinese title: %s", title)
                    return title

        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = block.get("text", "") or ""
            text_nosp = re.sub(r"\s+", "", text)
            if "题目" in text_nosp:
                for line in text.split("\n"):
                    line_nosp = re.sub(r"\s+", "", line)
                    if "题目" in line_nosp:
                        title = re.sub(r"\s*题\s*目\s*[：:\s]*", "", line).strip()
                        if len(title) > 2:
                            logger.info("Found Chinese title (block): %s", title)
                            return title

        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", "")).lower()
            if btype in ("text", "paragraph"):
                text = block.get("text", "").strip()
                text_clean = re.sub(r"\s+", " ", text)
                if text_clean and len(text_clean) > 10 and block.get("page_idx", 0) == 0:
                    return text_clean.split("\n")[0]

        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type", "")).lower()
            if btype in ("text", "paragraph"):
                text = block.get("text", "").strip()
                if text and len(text) > 5:
                    return re.sub(r"\s+", " ", text).split("\n")[0]

        return None

    # ── English ──────────────────────────────────────────────────────────
    tried_section_headings = False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", "")).lower()
        if btype in ("title",):
            text = block.get("text", "")
            if text and len(text.strip()) > 3:
                candidate = text.strip()
                if not _is_likely_section_heading(candidate):
                    return candidate
                tried_section_headings = True
        if block.get("text_level") == 1:
            text = block.get("text", "")
            if text and len(text.strip()) > 3:
                candidate = text.strip()
                if not _is_likely_section_heading(candidate):
                    return candidate
                tried_section_headings = True

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", "")).lower()
        if btype in ("text", "paragraph", "title"):
            text = block.get("text", "")
            if text and len(text.strip()) > 5:
                candidate = text.strip().split("\n")[0]
                if not _is_likely_section_heading(candidate):
                    return candidate
                tried_section_headings = True

    # All candidates were section headings; return the first one anyway as fallback
    return None


# ── DOI extraction from filename ───────────────────────────────────────────────

def extract_doi_from_filename(filename: str) -> str | None:
    """Extract DOI from PDF filename patterns.

    Examples:
        PhysRevLett.107.177207.pdf → 10.1103/PhysRevLett.107.177207
        srep24795.pdf → 10.1038/srep24795
        nnano.2016.18.pdf → 10.1038/nnano.2016.18
        10.1038@nphys2279.pdf → 10.1038/nphys2279
    """
    name = filename.replace(".pdf", "")

    # Pattern 1: 10.xxxx/... or 10.xxxx@... (explicit DOI, @ replaces / in filenames)
    m = re.match(r'(10\.\d{4,}[/@].+)', name)
    if m:
        return m.group(1).replace("@", "/")

    # Pattern 2: PhysRevXxx.xx.xxxxx (article ID may include letters, e.g. L020408)
    m = re.match(r'(PhysRev[A-Za-z]+\.[\d]+\.[A-Za-z0-9]+)', name)
    if m:
        return f"10.1103/{m.group(1)}"

    # Pattern 3: srepNNNNN
    if re.match(r'^srep\d+', name):
        return f"10.1038/{name}"

    # Pattern 4: nnano.xxxx / nmatNNNN
    m = re.match(r'(nnano\.\d+\.\d+)', name)
    if m:
        return f"10.1038/{m.group(1)}"
    m = re.match(r'(nmat\d+)', name)
    if m:
        return f"10.1038/{m.group(1)}"

    # Pattern 5: sNNNNN-NNNN-NNNN-N
    m = re.match(r'(s\d+-\d+-\d+-\d+)', name)
    if m:
        return f"10.1038/{m.group(1)}"

    return None


# ── Semantic Scholar metadata ─────────────────────────────────────────────────

def search_semantic_scholar_by_doi(doi: str) -> Optional[dict]:
    """Look up paper by DOI (exact match). Priority method."""
    if not doi:
        return None
    url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}"
    params = {"fields": "title,year,authors,externalIds,venue"}
    for attempt in range(3):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                logger.info("S2 DOI rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                logger.debug("DOI not found on S2: %s", doi)
                return None
            if resp.status_code != 200:
                logger.debug("S2 returned %d for DOI %s", resp.status_code, doi)
                return None
            data = resp.json()
            return {
                "title": data.get("title", ""),
                "year": str(data.get("year", "")),
                "authors": [a.get("name", "") for a in data.get("authors", [])],
                "doi": doi,
                "venue": data.get("venue", ""),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            logger.warning("S2 DOI error: %s", e)
            return None
    return None


def search_semantic_scholar(title: str) -> Optional[dict]:
    """Search Semantic Scholar API for paper metadata (fallback)."""
    if not title:
        return None
    _rate_limit()
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": title,
        "limit": 3,
        "fields": "title,year,authors,externalIds,venue",
        "api_key": SEMANTIC_SCHOLAR_KEY,
    }
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = (attempt + 1) * 5
                logger.info("S2 rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.debug("S2 returned status %d", resp.status_code)
                return None
            data = resp.json()
            papers = data.get("data", [])
            if not papers:
                return None
            best = papers[0]
            return {
                "title": best.get("title", title),
                "year": str(best.get("year", "")),
                "authors": [a.get("name", "") for a in best.get("authors", [])],
                "doi": best.get("externalIds", {}).get("DOI", ""),
                "venue": best.get("venue", ""),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            logger.warning("Semantic Scholar API error: %s", e)
            return None
    return None


# ── File operations ───────────────────────────────────────────────────────────


# ── File operations ───────────────────────────────────────────────────────────

def _find_content_json(directory: Path) -> Path | None:
    """Find MinerU content list JSON file."""
    for name in ("_content_list_v2.json", "_content_list.json", "content_list_v2.json", "content_list.json"):
        p = directory / name
        if p.exists():
            return p
    for f in directory.rglob("*content_list*.json"):
        return f
    return None


def rename_and_move(
    mineru_output_dir: Path,
    pdf_path: Path,
    metadata: Optional[dict] = None,
    title_override: str | None = None,
    year_override: str | None = None,
) -> dict:
    """Rename and move MinerU output files to their proper locations."""
    v2_json_path = _find_content_json(mineru_output_dir)
    raw_title = None
    # 语言判定：从已抽内容算（不再有独立步骤，也不再看文件名）
    is_chinese = (
        _content_looks_chinese(v2_json_path)
        if v2_json_path and v2_json_path.exists()
        else False
    )

    # If title_override provided (e.g. from vault md file), use it directly
    if title_override:
        raw_title = title_override
        logger.info("Using title override from md file: %s", raw_title)
    elif v2_json_path and v2_json_path.exists():
        raw_title = extract_title_from_mineru_json(v2_json_path, is_chinese=is_chinese)
        logger.info("Extracted title from MinerU: %s", raw_title)

    if not raw_title:
        raw_title = pdf_path.stem
        raw_title = re.sub(r"\s*_\s*", " ", raw_title)

    # ── DOI extraction from PDF filename ───────────────────────────────────
    doi_from_filename = extract_doi_from_filename(pdf_path.name)

    # ── Semantic Scholar metadata (DOI preferred, title fallback) ──────────
    scholar_meta = None
    if doi_from_filename:
        logger.info("DOI from filename: %s", doi_from_filename)
        scholar_meta = search_semantic_scholar_by_doi(doi_from_filename)
        if scholar_meta and scholar_meta.get("title"):
            logger.info("S2 DOI lookup: %s", scholar_meta["title"][:60])

    if not scholar_meta and raw_title:
        scholar_meta = search_semantic_scholar(raw_title)
        if scholar_meta and scholar_meta.get("title"):
            logger.info("S2 title lookup: %s", scholar_meta["title"][:60])

    # Final title
    if scholar_meta and scholar_meta.get("title"):
        final_title = scholar_meta["title"]
    else:
        final_title = raw_title or pdf_path.stem

    # Year: override > S2 (DOI exact > title search) > filename > UNKN
    year = ""
    if year_override:
        year = year_override
        logger.info("Year override from md file: %s", year)
    elif scholar_meta and scholar_meta.get("year"):
        year = scholar_meta["year"]
    if not year:
        year = extract_year(pdf_path.name)
    if not year and scholar_meta and scholar_meta.get("doi"):
        # Try CrossRef as fallback (more reliable than S2 for year)
        try:
            cr_url = f"https://api.crossref.org/works/{scholar_meta['doi']}"
            cr_resp = requests.get(cr_url, timeout=10)
            if cr_resp.status_code == 200:
                cr_data = cr_resp.json()
                issued = cr_data.get("message", {}).get("issued", {}).get("date-parts", [[]])[0]
                if issued:
                    year = str(issued[0])
                    logger.info("Year from CrossRef: %s", year)
        except Exception as e:
            logger.debug("CrossRef lookup failed: %s", e)
    if not year:
        year = "UNKN"

    # ── JCR 分区查询 ──────────────────────────────────────────────────────
    jcr_zone = None
    venue = (scholar_meta or {}).get("venue", "")
    if not JCR_ENABLED:
        logger.debug("JCR lookup skipped (RAG_JCR=0)")
    elif venue:
        try:
            jcr_zone = lookup_jcr_zone(venue)
            if jcr_zone:
                logger.info("JCR zone for %s: %s", venue, jcr_zone)
            else:
                logger.debug("JCR zone not found for venue: %s", venue)
        except Exception as e:
            logger.warning("JCR zone lookup error for %s: %s", venue, e)

    # Sanitize title for filename
    final_title = sanitize_filename_part(final_title)
    base_name = build_filename(final_title, year)
    base_name = sanitize_filename_part(base_name)
    logger.info("Base filename: %s", base_name)

    # Move raw MinerU JSON → RAG_Json/
    if v2_json_path and v2_json_path.exists():
        safe_move(v2_json_path, RAG_JSON_DIR / f"{base_name}_raw.json")
        logger.info("Moved raw MinerU JSON → RAG_Json/%s_raw.json", base_name)
    else:
        logger.warning("No content list JSON found in %s", mineru_output_dir)

    # Move origin PDF → DonePDF/
    origin_pdf = None
    for name in ("_origin.pdf", "origin.pdf"):
        p = mineru_output_dir / name
        if p.exists():
            origin_pdf = p
            break
    if not origin_pdf:
        for f in mineru_output_dir.rglob("*.pdf"):
            if "_content" not in f.name and "layout" not in f.name.lower():
                origin_pdf = f
                break
    if origin_pdf and origin_pdf.exists():
        safe_move(origin_pdf, DONE_PDF_DIR / f"{base_name}.pdf")
        logger.info("Moved PDF → DonePDF/%s.pdf", base_name)
    else:
        logger.warning("No PDF found to move to DonePDF/")

    # Move images/
    image_dir = None
    for name in ("images", "imgs", "figures"):
        p = mineru_output_dir / name
        if p.exists() and p.is_dir():
            image_dir = p
            break
    if image_dir and KEEP_IMAGES:
        target = IMAGES_DIR / f"{base_name}_images"
        safe_move(image_dir, target)
        logger.info("Moved images → images/%s_images", base_name)
    elif image_dir:
        # 默认不保留：图留在 .tmp 里，由 step8 随 tmp 一起清理
        logger.info(
            "Images not kept (RAG_KEEP_IMAGES=0)：%d 个文件留在 .tmp 待清理",
            sum(1 for _ in image_dir.rglob("*") if _.is_file()),
        )

    return {
        "base_name": base_name,
        "year": year,
        "title": final_title,
        "is_chinese": is_chinese,
        "jcr_zone": jcr_zone,
        "venue": venue,
        "doi": scholar_meta.get("doi", "") if scholar_meta else "",
        "scholar_meta": scholar_meta,
    }
