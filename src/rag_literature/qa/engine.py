#!/usr/bin/env python3
"""Query ChromaDB RAG knowledge base — improved version.

Usage:
  python3 -m rag_literature.qa.engine "你的研究问题" -k 5
  python3 -m rag_literature.qa.engine "domain wall motion" -k 3 --text
"""

import json, sys, os, argparse
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
import torch

from ..config import CHROMA_DATA_DIR, EMBEDDING_MODEL

# ── Config ──────────────────────────────────────────────────────────
DB_PATH = str(CHROMA_DATA_DIR)
MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", str(EMBEDDING_MODEL))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache model globally (don't reload on every query)
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"[model] Loading {MODEL_NAME} on {DEVICE}...", file=sys.stderr)
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE, local_files_only=True)
    return _model

def main():
    parser = argparse.ArgumentParser(
        description="Query ChromaDB RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", type=str, help="Search query (English or Chinese)")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--collection", type=str, default="", help="Search a specific collection only")
    parser.add_argument("--text", action="store_true", help="Human-readable text output (instead of JSON)")
    parser.add_argument("--show-zh", action="store_true", help="Include Chinese translation in output")
    parser.add_argument("--show-distance", action="store_true", help="Show distance/similarity score")
    parser.add_argument("--save", type=str, default="", help="Save output to file path (bypass bash truncation)")
    args = parser.parse_args()

    # Embed query
    model = get_model()
    query_vec = model.encode(args.query, show_progress_bar=False).tolist()

    # Query ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)

    if args.collection:
        targets = [args.collection]
    else:
        targets = [c.name for c in client.list_collections()]

    if not targets:
        print(json.dumps({"error":"No collections found"}, ensure_ascii=False))
        sys.exit(1)

    all_results = []
    for col_name in targets:
        try:
            col = client.get_collection(col_name)
        except Exception:
            continue

        raw = col.query(
            query_embeddings=[query_vec],
            n_results=args.top_k,
            include=["documents","metadatas","distances"],
        )

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            article = meta.get("book", meta.get("article", meta.get("source", "Unknown")))
            year = meta.get("year", "")
            page = meta.get("page", "")
            zh = meta.get("sentence_zh", "")

            entry = {
                "article": f"{article}_{year}" if year else article,
                "document": doc.strip() if doc else "",
                "similarity": round(1 / (1 + dist), 4), # convert L2 distance to 0-1 score
            }
            if page:
                entry["page"] = page
            if args.show_zh and zh:
                entry["chinese"] = zh.strip()
            if args.show_distance:
                entry["distance"] = round(dist, 4)
            all_results.append(entry)

    # ── Output ─────────────────────────────────────────────────────
    output_data = {"results": all_results, "query": args.query, "count": len(all_results)}

    if args.text:
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f" Query: {args.query}")
        lines.append(f" Results: {len(all_results)}")
        lines.append(f"{'='*60}\n")
        for i, r in enumerate(all_results, 1):
            lines.append(f" [{i}] {r['article']}")
            if r.get("page"):
                lines.append(f" Page: {r['page']}")
            lines.append(f" {r['document']}")
            lines.append(f" Score: {r['similarity']}")
            if r.get("chinese"):
                lines.append(f" ZH: {r['chinese']}")
            lines.append("")
        output_text = "\n".join(lines)
    else:
        output_text = json.dumps(output_data, ensure_ascii=False, indent=2)

    # If --save is specified, write to file and print only a tiny JSON to stdout
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output_text)
        # Tiny stdout to avoid bash truncation
        summary = {"saved_to": os.path.abspath(args.save), "count": len(all_results), "query": args.query}
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(output_text)


if __name__ == "__main__":
    main()
