"""Unified bibliography (references.bib) builder — incremental-friendly.

Merges every per-paper .bib under cnRead/ plus any v2.json papers that lack one
into a single `references.bib`, keyed by `Title_Year` (matching knowledge-core /
project_scope citation keys). Uses pybtex so the output is valid BibTeX.

This is cheap: it only re-reads metadata and re-merges — it does NOT re-run
MinerU, translation, or embedding. The RAG pipeline is already incremental
(via rag_state.json); this step just syncs the bibliography afterwards.
"""

import glob
import logging
import io
import json
import os
import re
from pathlib import Path

from pybtex.database import BibliographyData, Entry, Person, parse_string

from .config import CNREAD_DIR, RAG_JSON_DIR

logger = logging.getLogger(__name__)

# Month map incl. full names + 3-letter + 'sept' (the source .bib uses these)
_MONTH = {
    'january': '1', 'february': '2', 'march': '3', 'april': '4', 'may': '5',
    'june': '6', 'july': '7', 'august': '8', 'september': '9', 'october': '10',
    'november': '11', 'december': '12',
    'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'jun': '6', 'jul': '7',
    'aug': '8', 'sep': '9', 'sept': '9', 'oct': '10', 'nov': '11', 'dec': '12',
}


def _clean(raw: str) -> str:
    """Normalize month macros and strip embedded MathML so pybtex can parse."""
    def _mon(m):
        name = m.group(1).lower()
        return ('month = {%s}' % _MONTH[name]) if name in _MONTH else m.group(0)
    s = re.sub(r'\bmonth\s*=\s*\{?([A-Za-z]{3,9})\}?', _mon, raw)
    s = re.sub(r'<mml:math.*?</mml:math>', '', s, flags=re.S)
    s = re.sub(r'</?mml:\w+[^>]*>', '', s, flags=re.S)
    return s


def _norm_authors(a):
    """v2.json authors -> list of name strings ('and'-split, comma-safe)."""
    if not a:
        return []
    if isinstance(a, list):
        out = []
        for x in a:
            out += [y.strip() for y in str(x).split(' and ') if y.strip()]
        return out
    return [x.strip() for x in str(a).split(' and ') if x.strip()] or \
           [x.strip() for x in str(a).split(',') if x.strip()]


def _key_for(book, year):
    k = str(book).strip().replace(' ', '')
    if year and re.fullmatch(r'\d{4}', str(year)):
        k = k + '_' + str(year)
    return k


def _v2_meta_map():
    """key(Title_Year) -> v2.json meta for papers in the RAG JSON dir."""
    m = {}
    for f in glob.glob(str(RAG_JSON_DIR / '*_v2.json')):
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        meta = d.get('meta', {}) or {}
        key = _key_for(meta.get('book') or Path(f).stem, meta.get('year'))
        if key not in m:
            m[key] = meta
    return m


def refresh_references_bib(out_path: Path | None = None) -> int:
    """Incrementally update the unified references.bib.

    Base-first strategy so already-merged (rich) entries survive even when the
    per-paper .bib sources are gone: start from the existing references.bib,
    then add any new cnRead/*.bib and any new v2.json paper not yet present.

    Args:
        out_path: where to write the merged .bib. Defaults to
            `RAG_REFS_BIB` env, else `config.DATA_ROOT / 'references.bib'`.

    Returns:
        Number of entries in the refreshed bibliography.
    """
    if out_path is None:
        # 默认跟随 DATA_ROOT，而不是硬编码绝对路径 —— 否则用
        # RAG_DATA_ROOT 跑隔离测试时会写坏真库（2026-10-13 冒烟测试踩到过）。
        from .config import DATA_ROOT
        out_path = Path(os.environ.get('RAG_REFS_BIB', str(DATA_ROOT / 'references.bib')))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = {}

    # 1) BASE: existing references.bib (preserve already-merged richness)
    if out_path.exists():
        try:
            base = parse_string(io.open(out_path, encoding='utf-8').read(), 'bibtex')
            merged.update(base.entries)
        except Exception as e:
            # 绝不在 base 解析失败时静默重建。
            # 2026-10-13 事故：一次 base 解析失败导致 merged={}，整个库被从 v2.json
            # 重建，而 v2.json 的 meta.title 覆盖率只有 3/371 → 390/390 条标题退化成
            # CamelCase 文件名。宁可失败，也不要静默降级。
            raise RuntimeError(
                f"无法解析现有 {out_path}（{type(e).__name__}: {e}）—— 拒绝重建，"
                f"请先修复或移走该文件；确实要重建时把 out_path 指到一个新路径。"
            ) from e

    v2meta = _v2_meta_map()

    # 2) per-paper .bib (rich) — add any not already present
    def _authors_ok(e):
        return any(p for p in e.persons.get('author', [])
                   if str(p).strip().lower() not in ('unknown', ''))
    for f in sorted(glob.glob(str(CNREAD_DIR / '*.bib'))):
        key = Path(f).stem
        if key in merged:
            continue
        try:
            bd = parse_string(_clean(io.open(f, encoding='utf-8').read()), 'bibtex')
        except Exception:
            continue
        for _k, entry in bd.entries.items():
            if key not in merged:
                merged[key] = entry

    # 3) override field/author from v2.json for entries with Unknown/whitespace authors
    for key, e in merged.items():
        meta = v2meta.get(key, {})
        real = _norm_authors(meta.get('authors') or [])
        if real and not _authors_ok(e):
            e.persons['author'] = [Person(a) for a in real]

    # 4) v2.json papers with no bib entry yet (fallback)
    for key, meta in v2meta.items():
        if key in merged:
            continue
        # 标题优先 meta.title；meta.book 是 CamelCase 文件名，只作最后兜底
        _title = str(meta.get('title') or '').strip()
        fields = {'title': _title or str(meta.get('book') or key)}
        if meta.get('venue'):
            fields['journal'] = str(meta['venue'])
        year = meta.get('year')
        if year and re.fullmatch(r'\d{4}', str(year)):
            fields['year'] = str(year)
        if meta.get('doi'):
            fields['doi'] = str(meta['doi'])
        persons = {}
        names = _norm_authors(meta.get('authors') or [])
        if names:
            persons['author'] = [Person(a) for a in names]
        merged[key] = Entry('article', fields=fields, persons=persons)

    # 写盘前体检：标题里没有空格的基本就是 CamelCase 文件名（真标题极少单词）
    stem_like = 0
    for _k, _e in merged.items():
        _t = str(_e.fields.get('title', '')).strip()
        if _t and ' ' not in _t:
            stem_like += 1
    if merged and stem_like / len(merged) > 0.5:
        logger.warning(
            "references.bib: %d/%d 条标题疑似 CamelCase 文件名（真标题缺失）—— "
            "多半是 base 丢失或 v2.json 的 meta.title 为空", stem_like, len(merged))

    out = BibliographyData(entries=merged).to_string('bibtex')
    io.open(out_path, 'w', encoding='utf-8').write(out)
    return len(merged)
