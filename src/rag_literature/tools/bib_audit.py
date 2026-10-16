#!/usr/bin/env python3
"""Full-library bibliography health check + batch fix against Crossref.

Audit (read-only):
    python -m rag_literature.tools.bib_audit --audit
    python -m rag_literature.tools.bib_audit --audit --limit 20      # sample

Fix (backup first, then rewrite HIGH-CONFIDENCE entries from Crossref):
    python -m rag_literature.tools.bib_audit --fix
    python -m rag_literature.tools.bib_audit --fix --min-confidence medium
    python -m rag_literature.tools.bib_audit --fix --kinds title

Roll back:
    python -m rag_literature.tools.bib_audit --rollback <backup.bib>

Checks per entry (key = Title_Year):
  unknown_year / year_mismatch / key_year_stale / title_camel /
  title_mismatch / doi_mismatch / no_crossref / duplicate

Confidence decides what --fix may touch:
  high   = matched by DOI, or title-matched AND year agrees
  medium = title-matched strongly but year disagrees (possible wrong paper)
  low    = weak / no match

Crossref responses are cached, so re-runs and fixes need no extra network calls.
This script NEVER deletes; every fix writes a timestamped backup first.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from pybtex.database import parse_string

from ..config import DATA_ROOT

# Set RAG_CONTACT_MAIL to a real address when hitting Crossref: polite clients get
# better rate limits (https://api.crossref.org/swagger-ui/index.html).
_TMPWORK = Path(os.environ.get("RAG_TMPWORK_DIR", str(Path.home() / ".cache" / "rag" / "tmpwork")))
BIB = Path(os.environ.get("RAG_REFS_BIB", str(DATA_ROOT / "references.bib")))
CACHE = Path(os.environ.get("BIB_AUDIT_CACHE", str(_TMPWORK / "bib_audit_cache.json")))
REPORT = Path(os.environ.get("BIB_AUDIT_REPORT", str(_TMPWORK / "bib_audit_report.json")))
UA = "bib-audit/1.0"
CONTACT = os.environ.get("RAG_CONTACT_MAIL", "")
SLEEP = float(os.environ.get("BIB_AUDIT_SLEEP", "1.2"))
SIM_TITLE = 0.88
SIM_MATCH = 0.80


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print(f"  (cache save skipped: {e})", file=sys.stderr)


def _curl_json(url: str):
    try:
        out = subprocess.run(["curl", "-s", "-m", "30", "-A", UA, url],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def cr_by_doi(doi: str, cache: dict):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    if CONTACT:
        url += "?mailto=" + urllib.parse.quote(CONTACT)
    if url in cache:
        return cache[url]
    time.sleep(SLEEP)
    msg = _curl_json(url)
    if msg and msg.get("status") == "ok":
        cache[url] = msg["message"]
        return msg["message"]
    return None


def cr_by_title(title: str, cache: dict, rows: int = 5) -> list:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": title, "rows": rows, "mailto": CONTACT})
    if url in cache:
        return cache[url]
    time.sleep(SLEEP)
    msg = _curl_json(url)
    items = (msg or {}).get("message", {}).get("items", []) if msg else []
    cache[url] = items
    return items


def cr_year(m: dict) -> str:
    y = (m.get("published-print") or m.get("published-online") or
         m.get("issued") or {}).get("date-parts", [[None]])[0][0]
    return str(y) if y else ""


def cr_first_title(m: dict) -> str:
    return (m.get("title") or [""])[0]


def cr_authors(m: dict) -> str:
    au = []
    for a in m.get("author", []):
        fam, giv = a.get("family", "").strip(), a.get("given", "").strip()
        if fam:
            au.append(f"{fam}, {giv}" if giv else fam)
    return " and ".join(au)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def camel_split(k: str) -> str:
    y = re.search(r"_\d{4}$", k)
    base = k[:y.start()] if y else re.sub(r"_UNKN$", "", k)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)


def key_year(k: str) -> str:
    m = re.search(r"_(\d{4})$", k)
    return m.group(1) if m else ""


def best_title_match(entry_title: str, key_title: str, items: list):
    best, best_score = None, 0.0
    for it in items:
        t = cr_first_title(it)
        s = max(sim(t, key_title), sim(t, entry_title))
        if s > best_score:
            best, best_score = it, s
    return (best, best_score) if best is not None and best_score >= SIM_MATCH else (None, best_score)


def audit(bib_path: Path, limit=None, verbose: bool = True) -> dict:
    cache = _load_cache()
    bd = parse_string(bib_path.read_text(encoding="utf-8"), "bibtex")
    entries = list(bd.entries.items())
    if limit:
        entries = entries[:limit]

    findings = []
    for i, (key, e) in enumerate(entries, 1):
        cur_title = e.fields.get("title", "")
        cur_year = e.fields.get("year", "")
        cur_doi = (e.fields.get("doi") or e.fields.get("DOI") or "").strip()
        kt, ky = camel_split(key), key_year(key)
        flags = []
        cr, cr_src, score = None, "", 0.0

        if not cur_year or cur_year.upper() == "UNKN" or key.endswith("_UNKN"):
            flags.append("unknown_year")

        if cur_doi:
            m = cr_by_doi(cur_doi, cache)
            if m:
                cr, cr_src = m, "doi"
        if cr is None:
            m, score = best_title_match(cur_title, kt, cr_by_title(kt or cur_title, cache))
            if m:
                cr, cr_src = m, "title"
        if cr is None:
            flags.append("no_crossref")

        info = {"key": key, "flags": flags, "crossref_from": cr_src, "confidence": "low",
                "current": {"title": cur_title, "year": cur_year, "doi": cur_doi},
                "crossref": None}
        if cr:
            ct, cy = cr_first_title(cr), cr_year(cr)
            info["crossref"] = {"title": ct, "year": cy, "doi": cr.get("DOI", ""),
                                "journal": (cr.get("container-title") or [""])[0]}
            s_cur, s_key = sim(cur_title, ct), sim(kt, ct)
            if cur_title and " " not in cur_title.strip() and len(cur_title) > 18:
                flags.append("title_camel")
            if cur_title and s_cur < SIM_TITLE:
                flags.append("title_mismatch")
            if cur_year and cy and cur_year != cy:
                flags.append("year_mismatch")
            if ky and cy and ky != cy:
                flags.append("key_year_stale")
            if cur_doi and cr_src == "doi" and max(s_key, s_cur) < 0.70:
                flags.append("doi_mismatch")
            if cr_src == "doi" and max(s_key, s_cur) >= 0.70:
                info["confidence"] = "high"
            elif cr_src == "title" and max(s_key, s_cur) >= SIM_MATCH:
                info["confidence"] = "high" if (cur_year and cy and cur_year == cy) else "medium"
        findings.append(info)
        if verbose and (i % 20 == 0 or i == len(entries)):
            print("  audited %d/%d ..." % (i, len(entries)), flush=True)

    _save_cache(cache)

    def groups(field):
        g = {}
        for fi in findings:
            val = (fi["crossref"] or {}).get(field) or ""
            if val:
                g.setdefault(norm(val), []).append(fi["key"])
        return {k: v for k, v in g.items() if len(v) > 1}

    dup = {"by_doi": groups("doi"), "by_title": groups("title")}
    seen, dup_keys = set(), []
    for src in ("by_doi", "by_title"):
        for _, keys in dup[src].items():
            sig = tuple(sorted(keys))
            if sig not in seen:
                seen.add(sig)
                dup_keys.append(list(sig))

    summary = {}
    for fi in findings:
        for f in fi["flags"]:
            summary[f] = summary.get(f, 0) + 1
        summary["conf_" + fi["confidence"]] = summary.get("conf_" + fi["confidence"], 0) + 1
    summary["duplicate_groups"] = len(dup_keys)

    report = {"generated": datetime.now().isoformat(), "bib": str(bib_path),
              "total": len(findings), "summary": summary,
              "duplicates": dup_keys, "entries": findings}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_summary(rep: dict):
    s = rep["summary"]
    print("\n" + "=" * 64)
    print(" 参考文献库体检 · %s · %d 条" % (rep["bib"], rep["total"]))
    print("=" * 64)
    label = [("unknown_year", "缺年份(UNKN)"), ("year_mismatch", "年份字段错"),
             ("key_year_stale", "key 年份标签旧"), ("title_camel", "标题为压缩形式"),
             ("title_mismatch", "标题与权威不符"), ("doi_mismatch", "DOI 指向别篇"),
             ("no_crossref", "Crossref 无匹配"), ("duplicate_groups", "同名撞车(组)")]
    for k, lab in label:
        if k in s:
            print("  %-16s %d" % (lab, s[k]))
    print("  ---- 置信度: high=%d  medium=%d  low=%d" %
          (s.get("conf_high", 0), s.get("conf_medium", 0), s.get("conf_low", 0)))
    if rep["duplicates"]:
        print("\n  同名撞车明细:")
        for keys in rep["duplicates"]:
            print("    - " + "  ==  ".join(keys))
    print("\n 报告: %s" % REPORT)


def _build_entry(key: str, m: dict) -> str:
    def q(x):
        return str(x).replace('"', "'")
    lines = ['    author = "%s",' % q(cr_authors(m)), '    title = "%s",' % q(cr_first_title(m))]
    j = (m.get("container-title") or [""])[0]
    if j:
        lines.append('    journal = "%s",' % q(j))
    for src, dst in (("volume", "volume"), ("issue", "number"), ("page", "pages")):
        if m.get(src):
            lines.append('    %s = "%s",' % (dst, q(m[src])))
    y = cr_year(m)
    if y:
        lines.append('    year = "%s",' % y)
    if m.get("DOI"):
        lines.append('    doi = "%s",' % m["DOI"])
        lines.append('    url = "https://doi.org/%s"' % m["DOI"])
    return "@article{%s,\n%s\n}\n" % (key, "\n".join(lines))


def fix(bib_path: Path, kinds: set, min_conf: str):
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    cache = _load_cache()
    allowed = {"high"} if min_conf == "high" else {"high", "medium"}
    src = bib_path.read_text(encoding="utf-8")
    backup = bib_path.with_suffix(".bak.%s.bib" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(bib_path, backup)
    changed, skipped = [], []
    for fi in rep["entries"]:
        key, cr = fi["key"], fi["crossref"]
        if not cr or fi["confidence"] not in allowed:
            continue
        hit = [f for f in fi["flags"] if f in kinds]
        if not hit:
            continue
        # resolve the RAW Crossref record (from cache; no extra network)
        raw = cr_by_doi(cr["doi"], cache) if cr.get("doi") else None
        if raw is None:
            items = cr_by_title(camel_split(key), cache)
            raw, _ = best_title_match(fi["current"]["title"], camel_split(key), items)
        if raw is None:
            skipped.append((key, "no raw Crossref record"))
            continue
        t = cr_first_title(raw).strip()
        # SAFETY GATE: never write a degenerate entry
        if len(t) < 10 or " " not in t or not cr_authors(raw).strip():
            skipped.append((key, "built entry rejected"))
            continue
        pat = re.compile(r"@\w+\{" + re.escape(key) + r",.*?\n\}\n?", re.S)
        if not pat.search(src):
            skipped.append((key, "bib 无此条目"))
            continue
        src = pat.sub(lambda _: _build_entry(key, raw), src, count=1)
        changed.append((key, ",".join(sorted(hit)), cr_year(raw), t[:46]))
    bib_path.write_text(src, encoding="utf-8")
    log = REPORT.with_suffix(".changes.txt")
    with log.open("w", encoding="utf-8") as fh:
        fh.write("backup: %s\nchanged: %d\n\n" % (backup, len(changed)))
        for k, h, y, t in changed:
            fh.write("%s\t[%s]\t%s\t%s\n" % (k, h, y, t))
    print("备份: %s\n已修: %d 条 | 跳过: %d | 变更日志: %s" % (backup, len(changed), len(skipped), log))
    print("回滚: python -m rag_literature.tools.bib_audit --rollback %s" % backup)
    return backup


def rollback(backup: Path, bib_path: Path):
    shutil.copy2(backup, bib_path)
    print("已从 %s 回滚到 %s" % (backup, bib_path))


def main():
    ap = argparse.ArgumentParser(description="Bibliography health check + batch fix (Crossref)")
    ap.add_argument("--bib", default=str(BIB))
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--rollback", metavar="BACKUP")
    ap.add_argument("--kinds", default="year,title,doi")
    ap.add_argument("--min-confidence", default="high", choices=["high", "medium"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bib = Path(args.bib)
    if args.rollback:
        rollback(Path(args.rollback), bib)
        return
    if args.fix:
        kinds = set()
        if "year" in args.kinds:
            kinds |= {"year_mismatch", "key_year_stale"}
        if "title" in args.kinds:
            kinds |= {"title_camel", "title_mismatch"}
        if "doi" in args.kinds:
            kinds |= {"doi_mismatch"}
        fix(bib, kinds, args.min_confidence)
        return
    print_summary(audit(bib, limit=args.limit))


if __name__ == "__main__":
    main()
