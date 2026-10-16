# Getting started

This walks you from an empty machine to "ask a question, get the sentence + citation".
Everything is local; the only network traffic is model downloads and metadata lookups.

## 0. What you need

| | |
|---|---|
| NVIDIA GPU | ≥ 16 GB VRAM for MinerU (it peaks ~15 GB). 24 GB is comfortable. CPU-only works but parsing becomes slow |
| Python | 3.10+ (MinerU may pin a version; 3.10 is the safe choice) |
| Disk | ~5 GB models + ~1 GB per 200 k text chunks. PDFs/JSON are small |
| Tools | `conda` (MinerU is easiest to isolate in its own env), `git` |

## 1. Two environments (on purpose)

MinerU drags in a heavy, opinionated dependency set (torch, layout models, sometimes
vLLM). Keep it away from the pipeline's own env:

```bash
# (a) the pipeline
git clone <this repo> rag-literature && cd rag-literature
conda create -n rag python=3.10 -y && conda activate rag
pip install -e .                 # chromadb, sentence-transformers, torch

# (b) MinerU, in its own env
conda create -n mineru python=3.10 -y
conda activate mineru
pip install "mineru[core]"       # or `pip install -U "mineru[all]"` for the VLM backend
mineru --version
conda deactivate
```

The pipeline invokes MinerU as `conda run -n <RAG_MINERU_CONDA_ENV> mineru ...`, so the
pipeline env never needs MinerU's dependencies.

## 2. Decide where things live

Three locations, and keeping them separate saves you pain later:

```bash
export RAG_DATA_ROOT=$HOME/rag-data        # everything the pipeline produces
export RAG_MODELS_DIR=$HOME/rag-models     # weights + MinerU model cache
# retrieval cache defaults to $RAG_DATA_ROOT/.retrieval_index
mkdir -p "$RAG_DATA_ROOT/.input" "$RAG_MODELS_DIR"
```

If you have a NAS/large volume, put `RAG_DATA_ROOT` and `RAG_MODELS_DIR` there — but keep
the *retrieval cache* (`RAG_INDEX_DIR`) on fast local disk if queries feel slow.

## 3. Models

| Model | Where it comes from | Notes |
|---|---|---|
| **bge-m3** (embeddings, 1024-d) | `huggingface-cli download BAAI/bge-m3 --local-dir $RAG_MODELS_DIR/bge-m3` | The pipeline loads it from a local path via `RAG_EMBEDDING_MODEL`. Offline flags are set by default |
| **bge-reranker-v2-m3** (optional) | `huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir $RAG_MODELS_DIR/bge-reranker-v2-m3` | **Leave it off** unless you re-run the evaluation yourself — see `docs/retrieval.md` |
| **MinerU layout/OCR models** (~5 GB) | downloaded automatically on the first parse into `MODELSCOPE_CACHE` | The pipeline pins that variable to `$RAG_MODELS_DIR/modelscope` and injects it into the MinerU subprocess — so the weights land where **you** said, not in `~/.cache` |

Gotcha worth repeating: MinerU resolves its cache through the `MODELSCOPE_CACHE`
environment variable, and it runs as a subprocess (non-interactive, non-login shell).
Anything exported only from `~/.bashrc` is invisible to it, and ModelScope then
re-downloads into `~/.cache/modelscope`. This repo sets it in code
(`config.MINERU_MODELSCOPE_CACHE` → injected in `pipeline/step2_mineru.py`).

Optional: `~/mineru.json` is MinerU's own config file. It is only consulted when
`MINERU_MODEL_SOURCE=local` (see `docs/troubleshooting.md` §1).

## 4. Configure

```bash
cp .env.example .env && $EDITOR .env
# minimum: RAG_DATA_ROOT, RAG_MODELS_DIR, RAG_MINERU_CONDA_ENV
set -a && . ./.env && set +a      # or use direnv / systemd / your own loader
```

## 5. First run

```bash
cp ~/Downloads/some-paper.pdf "$RAG_DATA_ROOT/.input/"
python -m rag_literature.cli            # processes everything in .input/
```

What you should see afterwards:

```bash
ls "$RAG_DATA_ROOT/DonePDF"             # <Book>_<Year>.pdf   (PDFs are renamed + moved here)
ls "$RAG_DATA_ROOT/.RAG_Json"           # <Book>_<Year>_v2.json (structured text: blocks + sentences)
ls "$RAG_DATA_ROOT/.vector_db"          # chroma_data/ rag_state.json pipeline_state.json dedup_hashes.json
ls "$RAG_DATA_ROOT/.tmp"                # empty (cleaned in step 8)
```

`.input/` is emptied automatically once a paper succeeds: the original is moved to
`._trash/input_done_<timestamp>/` (recoverable, never deleted outright).

If MinerU fails on a scanned/broken PDF, retry it with `RAG_MINERU_METHOD=ocr`.

## 6. Retrieval layer

```bash
python -m rag_literature.tools.export_index          # build the read-only cache (a few minutes)
python -m rag_literature.tools.export_index --check  # exit 3 = cache stale, rebuild needed
python -m rag_literature.qa.serve --host 127.0.0.1 --port 8077 &
curl -s '127.0.0.1:8077/health'
curl -s '127.0.0.1:8077/search?q=your+question&k=5' | python -m json.tool
```

The service keeps the embedding model warm: ~0.4–0.7 s per query instead of ~20 s when
each query has to load the model. Re-export after every ingest — the service notices the
new `meta.json` and reloads by itself (no restart).

## 7. Verify before trusting it

```bash
# 1. chunk count in the vector store == chunk count in the read-only cache
python - <<'PY'
import sqlite3, json, os
from rag_literature.config import CHROMA_DATA_DIR, RETRIEVAL_INDEX_DIR
n = sqlite3.connect(f"file:{CHROMA_DATA_DIR}/chroma.sqlite3?mode=ro", uri=True) \
      .execute("select count(*) from embeddings").fetchone()[0]
print("chroma rows:", n)
print("cache meta :", json.load(open(RETRIEVAL_INDEX_DIR / "meta.json"))["count"])
PY
```

```bash
# 2. the index actually covers all stored rows (see troubleshooting §3 for why this can lag)
python - <<'PY'
import pickle, sqlite3, os
from rag_literature.config import CHROMA_DATA_DIR as b
n = sqlite3.connect(f"file:{b}/chroma.sqlite3?mode=ro", uri=True) \
      .execute("select count(*) from embeddings").fetchone()[0]
for d in os.listdir(b):
    p = os.path.join(b, d, "index_metadata.pickle")
    if os.path.exists(p):
        k = len(pickle.load(open(p, "rb"))["id_to_label"])
        print(f"stored={n} indexed={k} missing={n-k}")
PY
```

```bash
# 3. retrieval sanity check: a phrase you know is in a paper must find it
python -m rag_literature.qa.engine "a sentence you copied from the PDF" -k 3 --text
```

## 8. Growing the library

- Drop new PDFs into `.input/` and re-run `python -m rag_literature.cli` — already-processed
  files are skipped via `dedup_hashes.json` / `pipeline_state.json`.
- Then re-run `export_index` (or run the pipeline, which does it as step 6c).
- `RAG_PIPELINE_WORKERS=2` is a reasonable default: step 3/4/6 are I/O bound and overlap
  with MinerU, but MinerU itself must not run twice on the same GPU (VRAM).

Next: [architecture](architecture.md) · [pipeline](pipeline.md) ·
[retrieval](retrieval.md) · [troubleshooting](troubleshooting.md)
