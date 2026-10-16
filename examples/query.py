#!/usr/bin/env python3
"""Query the retrieval layer directly (no HTTP service).

    python examples/query.py "why do kinks feel a Peierls-Nabarro barrier"

This loads the embedding model once and answers any number of questions, which is the
right pattern for notebooks/batch jobs. For interactive use prefer `qa.serve`
(the model stays warm between processes).
"""

import sys

from rag_literature.qa.retriever import Retriever


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    query = " ".join(sys.argv[1:])
    rt = Retriever()
    try:
        print(f"index: {rt}\n")
        for i, hit in enumerate(rt.search(query, k=5, hybrid=True, window=1), 1):
            text = " ".join((hit.get("text") or "").split())
            s = hit.get("scores") or {}
            print(f"[{i}] {hit.get('citation')}  p.{hit.get('page')}  "
                  f"similarity={hit.get('similarity')}  "
                  f"(dense_rank={s.get('dense_rank')} lexical_rank={s.get('lexical_rank')} "
                  f"dense_cos={s.get('dense_cos')})")
            print(f"    {text[:300]}{'…' if len(text) > 300 else ''}\n")
    finally:
        rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
