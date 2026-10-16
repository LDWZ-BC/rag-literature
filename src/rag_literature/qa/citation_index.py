
"""Citation index — tracks which papers cite which external papers."""

import json, re
from pathlib import Path
from collections import defaultdict

from ..config import RAG_JSON_DIR, VECTOR_DB_DIR

V2_DIR = RAG_JSON_DIR
INDEX_PATH = VECTOR_DB_DIR / "citation_index.json"
DOI_RE = re.compile(r'(10\.\d{4}(?:\.\d+)*/[^\s"\']+(?:\([^)]*\))?[^\s,;.\'")\]]*)', re.I)


def build_index():
    """Scan all v2.json files and build citation index."""
    v2_files = sorted(V2_DIR.glob("*_v2.json"))
    local_dois = {}
    paper_to_refs = defaultdict(set)

    for f in v2_files:
        d = json.loads(f.read_text())
        name = f.name.replace("_v2.json", "")
        doi = (d.get("meta", {}) or {}).get("doi", "") or ""
        if doi:
            local_dois[doi.lower().strip()] = name

    for f in v2_files:
        name = f.name.replace("_v2.json", "")
        d = json.loads(f.read_text())
        texts = []
        for b in d.get("blocks", []):
            for s in b.get("sentences", []):
                if isinstance(s, dict) and "en" in s:
                    texts.append(s["en"])
        full = " ".join(texts)

        ref_start = None
        for marker in ["references", "bibliography", "works cited"]:
            idx = full.lower().rfind(marker)
            if idx >= 0 and idx > len(full) * 0.3:
                ref_start = idx
                break
        if ref_start is None:
            last_part = full[len(full) * 7 // 10:]
            idx = last_part.lower().find("[1]")
            if idx >= 0:
                ref_start = len(full) * 7 // 10 + idx
        if ref_start is None:
            continue

        ref_text = full[ref_start:]
        ref_dois = set()
        for doi in DOI_RE.findall(ref_text):
            doi_clean = doi.strip().rstrip('.,;:)"[]').lower()
            ref_dois.add(doi_clean)
        paper_to_refs[name] = ref_dois

    doi_to_papers = defaultdict(list)
    for paper, refs in paper_to_refs.items():
        for doi in refs:
            doi_to_papers[doi].append(paper)

    index = {
        "local_dois": local_dois,
        "paper_to_refs": {k: list(v) for k, v in paper_to_refs.items()},
        "doi_to_papers": dict(doi_to_papers),
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index


def load_index():
    """Load citation index, auto-build if missing."""
    if not INDEX_PATH.exists():
        return build_index()
    return json.loads(INDEX_PATH.read_text())


def get_citing_papers(doi: str):
    """Return local papers that cite the given DOI."""
    idx = load_index()
    return idx.get("doi_to_papers", {}).get(doi.lower(), [])


def get_missing_papers():
    """Return DOIs cited locally but not in collection, sorted by citation count."""
    idx = load_index()
    local = set(idx.get("local_dois", {}).keys())
    missing = []
    for doi, papers in idx.get("doi_to_papers", {}).items():
        if doi not in local:
            missing.append((doi, papers))
    missing.sort(key=lambda x: len(x[1]), reverse=True)
    return missing


if __name__ == "__main__":
    idx = build_index()
    missing = get_missing_papers()
    n_papers = len(idx.get("paper_to_refs", {}))
    n_refs = len(idx.get("doi_to_papers", {}))
    print(f"Index built: {n_papers} papers, {n_refs} refs")
    print(f"Missing papers (cited but not in local collection): {len(missing)}")
    for doi, papers in missing[:10]:
        print(f"  {doi} — cited by {len(papers)} papers")
