#!/usr/bin/env python3
"""Process a single PDF, then export the read-only retrieval index.

    python examples/process_one_pdf.py ~/Downloads/paper.pdf

Useful when you want to drive the pipeline from your own script instead of the CLI.
Set RAG_DATA_ROOT / RAG_MODELS_DIR first (or rely on the defaults under $HOME).
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from rag_literature.config import INPUT_DIR, DONE_PDF_DIR, RAG_JSON_DIR  # noqa: E402
from rag_literature.pipeline.orchestrator import process_pdf  # noqa: E402
from rag_literature.tools.export_index import main as export_index_main  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.is_file():
        print(f"not a file: {src}")
        return 2

    # The orchestrator expects new PDFs under the data root's .input/
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    staged = INPUT_DIR / src.name
    if not staged.exists():
        staged.write_bytes(src.read_bytes())
    print(f"staged → {staged}")

    info = process_pdf(staged)
    if not info:
        print("processing failed — see the log above, or retry with RAG_MINERU_METHOD=ocr")
        return 1

    print(f"\nbook       : {info.get('base_name')}")
    print(f"DonePDF    : {DONE_PDF_DIR / (str(info.get('base_name')) + '.pdf')}")
    print(f"structured : {RAG_JSON_DIR / (str(info.get('base_name')) + '_v2.json')}")

    # Refresh the read-only cache so the new paper is immediately queryable.
    print("\nrebuilding the retrieval index …")
    sys.argv = [sys.argv[0]]
    return export_index_main()


if __name__ == "__main__":
    sys.exit(main())
