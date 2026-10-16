# rag-literature

A **fully offline** pipeline that turns a folder of research PDFs into a queryable
knowledge base: `PDF → parse → metadata → clean → embeddings → hybrid retrieval`,
with an optional local LLM for translation / notes. No paid APIs, no cloud, no telemetry.

This repository is the *starter kit*: the pipeline core, the retrieval layer and the
operational notes that took the longest to get right. Copy it, point it at your own
data directory, and run.

```
                 ┌──────────────────────── your data root (RAG_DATA_ROOT) ───────────────────────┐
 .input/*.pdf ─▶ │ step2  MinerU parse (GPU)        → .tmp/<paper>/_content_list_v2.json          │
                 │ step3  metadata + naming         → DonePDF/<Book>_<Year>.pdf, .RAG_Json/*_v2.json │
                 │ step4  clean (no translation)    → .RAG_Json/<Book>_<Year>_v2.json              │
                 │ step6  embed + index             → .vector_db/chroma_data (bge-m3, 1024-d)      │
                 │ step6b unified bibliography      → references.bib                               │
                 │ step6c export read-only index    → .retrieval_index/ (or ~/.cache/rag/index)    │
                 │ step8  clean .tmp, archive .input→ ._trash/                                     │
                 └──────────────────────────────────────────────────────────────────────────────────┘
 retrieval:  rag-serve (HTTP, models kept warm)  ──▶  qa/retriever.py (dense + BM25 → RRF → window)
```

## Why this exists / what makes it different

Most "chat with your PDFs" demos stop at embeddings. This one is the whole loop for a
**growing personal library**, and it encodes the failure modes that cost real time:

| Problem you will hit | What this repo does |
|---|---|
| MinerU silently re-downloads ~5 GB of models into `~/.cache` | pins the cache and **injects it into the MinerU subprocess** (`docs/troubleshooting.md` §1) |
| Rebuilding a HNSW index *in place* makes the file grow without bound (we saw a 6.8 TB sparse file) | always build into a **new directory**, verify, then switch (`docs/troubleshooting.md` §2) |
| Last few hundred chunks never become searchable | Chroma only persists every 1000 writes → explicit flush at the end (`docs/troubleshooting.md` §3) |
| Dense-only retrieval fails on exact terms (`Peierls-Nabarro`, author names) | hybrid dense + BM25 with RRF: R@5 **0.38 → 0.87** on rare-term queries |
| Cross-encoder rerankers feel like they should help | measured: they **lowered** paper-level recall in all three test groups → off by default |
| Translation is the expensive part of the pipeline | removed from the pipeline; it contributed nothing to retrieval (English text is what gets embedded) |

## Requirements

| | |
|---|---|
| OS | Linux (tested) — macOS should work; Windows via WSL |
| GPU | NVIDIA, **≥ 16 GB VRAM recommended** (MinerU peaks ~15 GB; a 24 GB card is comfortable) |
| Python | 3.10+ (two conda envs, see below) |
| Disk | PDFs + JSON: small. Embeddings: ~1 GB per ~200 k chunks. MinerU models: ~5 GB |
| Optional | `llama.cpp` server + a GGUF model, only for translation / notes |

## Quick start

```bash
git clone <this repo> rag-literature && cd rag-literature
pip install -e .            # light deps: chromadb, sentence-transformers, torch

# 1. MinerU lives in its own env (heavy: torch + vllm + layout models)
conda create -n mineru python=3.10 -y
conda activate mineru && pip install "mineru[core]" && conda deactivate

# 2. Tell the pipeline where things live
cp .env.example .env && $EDITOR .env      # RAG_DATA_ROOT, RAG_MODELS_DIR, RAG_MINERU_CONDA_ENV ...

# 3. Fetch embedding + layout models into RAG_MODELS_DIR (see docs/getting-started.md)
# 4. Drop a PDF in and run
mkdir -p "$RAG_DATA_ROOT/.input" && cp your-paper.pdf "$RAG_DATA_ROOT/.input/"
python -m rag_literature.cli                     # parse + index the whole .input/
python -m rag_literature.tools.export_index      # build the read-only retrieval cache
python -m rag_literature.qa.serve --host 127.0.0.1 --port 8077 &
curl -s '127.0.0.1:8077/search?q=your+question&k=5' | python -m json.tool
```

Step-by-step version, including model downloads and a smoke test:
**[docs/getting-started.md](docs/getting-started.md)**.

## Documentation

| Doc | Contents |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | install, models, env vars, first run, verification |
| [docs/architecture.md](docs/architecture.md) | code vs data layout, why the split, module map |
| [docs/pipeline.md](docs/pipeline.md) | the seven steps, what each produces, parallelism, cost |
| [docs/retrieval.md](docs/retrieval.md) | dense + BM25 + RRF, the window trick, the serve API, evaluation harness |
| [docs/troubleshooting.md](docs/troubleshooting.md) | **read this before you lose data** — the five costly mistakes |

## Environment variables (most useful ones)

Everything has a working default, so you only set what differs. Full list in
[`.env.example`](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `RAG_DATA_ROOT` | `~/rag-data` | one root holding `.input/`, `DonePDF/`, `.RAG_Json/`, `.vector_db/`, … |
| `RAG_MODELS_DIR` | `~/rag-models` | local model weights (bge-m3, reranker, MinerU cache) |
| `RAG_MODELSCOPE_CACHE` | `$RAG_MODELS_DIR/modelscope` | MinerU/ModelScope cache — pinned for you |
| `RAG_MINERU_CONDA_ENV` | `mineru` | conda env that has MinerU installed |
| `RAG_MINERU_METHOD` | `txt` | `txt` (fast, uses the PDF text layer) or `ocr` (rescues broken text layers) |
| `RAG_MINERU_LANG` | `ch` | MinerU language hint; `ch` handles Chinese **and** English |
| `RAG_EMBEDDING_MODEL` | `$RAG_MODELS_DIR/bge-m3` | sentence-transformers model (1024-d) |
| `RAG_INDEX_DIR` | `$RAG_DATA_ROOT/.retrieval_index` | read-only retrieval cache |
| `RAG_PIPELINE_WORKERS` | `2` | papers processed concurrently (GPU memory is the real limit) |
| `RAG_JCR` | off | opt-in JCR quartile lookup (needs your own EasyScholar key) |

## License

MIT — see [LICENSE](LICENSE). Third-party models and tools (MinerU, bge-m3, llama.cpp,
ChromaDB) keep their own licenses.
