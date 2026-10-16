"""manifest.json builder — data warehouse completeness index + dedup support.

Scans the configured data root and writes DATA_ROOT/manifest.json.
Run:

    python -m rag_literature.tools.manifest_build
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# --- Config -----------------------------------------------------------------
from ..config import DATA_ROOT, RAG_JSON_DIR, DONE_PDF_DIR, CNREAD_DIR, IMAGES_DIR
# 向量库以 config 的 CHROMA_DATA_DIR 为准（自 2026-08 起管线写的是工作区运行副本
# ~/.cache/rag/vector_db/chroma_data；/data 侧 .vector_db 只是只读归档、会陈旧）。
# 曾硬编码 VECTOR_DB_DIR = DATA_ROOT/".vector_db" 导致 chunks/no_chunks 全部失真
# （2026-10-07 修）。可用 RAG_CHROMA_DIR 环境变量覆盖。
from ..config import CHROMA_DATA_DIR  # noqa: E402
from ..config import CHROMA_COLLECTION_NAME as COLLECTION_NAME  # noqa: E402
VECTOR_DB_DIR = Path(CHROMA_DATA_DIR).parent
MANIFEST_FILE = DATA_ROOT / "manifest.json"
EMPTY_JSON_THRESHOLD = 500  # bytes

_SUFFIXES = ("_v2.json", ".json", ".md", ".pdf", "_notes.md")


def base_of(filename: str) -> str:
    """Strip known suffixes to get the canonical paper key."""
    name = filename
    for suf in _SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.strip()


def content_hash_of(data: dict) -> str:
    """MD5 (first 10 hex) of all 'en' sentence text concatenated."""
    text = ""
    for block in data.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for sent in block.get("sentences", []):
            if isinstance(sent, dict):
                text += str(sent.get("en", ""))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def _chunk_counts() -> dict:
    """Return {file_key: chunk_count} from ChromaDB."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DATA_DIR))
        col = client.get_or_create_collection(name=COLLECTION_NAME)
        counts = defaultdict(int)
        offset = 0
        batch = 5000
        while True:
            res = col.get(offset=offset, limit=batch, include=["metadatas"])
            ids = res.get("ids") or []
            if not ids:
                break
            metas = res.get("metadatas") or []
            for m in metas:
                if isinstance(m, dict) and m.get("file"):
                    counts[m["file"]] += 1
            offset += len(ids)
            if len(ids) < batch:
                break
        return dict(counts)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ChromaDB unavailable: {e}", file=sys.stderr)
        return {}


def build() -> dict:
    entries = {}
    json_files = {}
    pdf_files = {}
    cnread_files = {}
    notes_files = {}

    # Collect file inventory
    for f in RAG_JSON_DIR.glob("*_v2.json"):
        json_files[base_of(f.name)] = f
    for f in DONE_PDF_DIR.glob("*.pdf"):
        pdf_files[base_of(f.name)] = f
    for f in CNREAD_DIR.glob("*.md"):
        b = base_of(f.name)
        if f.name.endswith("_notes.md"):
            notes_files[b] = f
        else:
            cnread_files[b] = f

    chunk_counts = _chunk_counts()

    # Build entries for every paper (source of truth = filesystem union)
    all_bases = set(json_files) | set(pdf_files)
    for base in sorted(all_bases):
        entry = {
            "json": None,
            "json_size": 0,
            "pdf": None,
            "cnread": None,
            "notes": None,
            "meta": {},
            "content_hash": "",
            "chunks": 0,
        }
        if base in json_files:
            jf = json_files[base]
            entry["json"] = jf.name
            entry["json_size"] = jf.stat().st_size
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                m = data.get("meta", {}) if isinstance(data, dict) else {}
                entry["meta"] = {
                    k: m.get(k)
                    for k in ("year", "doi", "jcr_zone", "is_chinese", "book")
                }
                entry["content_hash"] = content_hash_of(data)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] bad json {jf.name}: {e}", file=sys.stderr)
        if base in pdf_files:
            entry["pdf"] = pdf_files[base].name
        if base in cnread_files:
            entry["cnread"] = cnread_files[base].name
        if base in notes_files:
            entry["notes"] = notes_files[base].name
        entry["chunks"] = chunk_counts.get(entry["json"], 0)
        entries[base] = entry

    # Issues
    no_json = sorted(b for b, e in entries.items() if not e["json"])
    no_pdf = sorted(b for b, e in entries.items() if e["json"] and not e["pdf"])
    no_cnread = sorted(b for b, e in entries.items() if e["json"] and not e["cnread"])
    no_chunks = sorted(b for b, e in entries.items() if e["json"] and e["chunks"] == 0)
    empty_json = sorted(
        b for b, e in entries.items()
        if e["json"] and e["json_size"] < EMPTY_JSON_THRESHOLD
    )

    dup_doi = defaultdict(list)
    for b, e in entries.items():
        doi = (e["meta"].get("doi") or "").strip()
        if doi:
            dup_doi[doi].append(b)
    duplicates_by_doi = {d: v for d, v in sorted(dup_doi.items()) if len(v) > 1}

    # Skip empty hash (empty-JSON shells are already flagged as empty_json)
    dup_hash = defaultdict(list)
    for b, e in entries.items():
        if e["content_hash"] and e["json_size"] >= EMPTY_JSON_THRESHOLD:
            dup_hash[e["content_hash"]].append(b)
    duplicates_by_hash = {h: v for h, v in sorted(dup_hash.items()) if len(v) > 1}

    orphan_images = []
    for d in sorted(IMAGES_DIR.iterdir()):
        if not d.is_dir():
            continue
        b = d.name
        if b.endswith("_images"):
            b = b[: -len("_images")]
        if b not in entries:
            orphan_images.append(d.name)

    totals = {
        "json": sum(1 for e in entries.values() if e["json"]),
        "pdf": sum(1 for e in entries.values() if e["pdf"]),
        "cnread_md": sum(1 for e in entries.values() if e["cnread"]),
        "notes": sum(1 for e in entries.values() if e["notes"]),
        "chunks_total": sum(e["chunks"] for e in entries.values()),
    }

    manifest = {
        "version": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": totals,
        "entries": entries,
        "issues": {
            "no_json": no_json,
            "no_pdf": no_pdf,
            "no_cnread": no_cnread,
            "no_chunks": no_chunks,
            "empty_json": empty_json,
            "duplicates_by_doi": duplicates_by_doi,
            "duplicates_by_hash": duplicates_by_hash,
            "orphan_images": orphan_images,
        },
    }
    return manifest


def main() -> int:
    manifest = build()
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest.json written: {MANIFEST_FILE}")
    print(
        "  entries=%d chunks_total=%d"
        % (len(manifest["entries"]), manifest["totals"]["chunks_total"])
    )
    issues = manifest["issues"]
    for k, v in issues.items():
        if v:
            n = len(v) if isinstance(v, list) else len(v)
            print(f"  issue {k}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
