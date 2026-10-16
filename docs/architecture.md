# Architecture

Three locations, each with one job. Mixing them is what turns a working prototype into a
repository full of half-gigabyte binaries.

| Layer | Default | Holds | Never holds |
|---|---|---|---|
| **Code** | this repository | pipeline, retrieval, tools, docs | PDFs, JSON, vector stores, caches |
| **Data root** | `$RAG_DATA_ROOT` (`~/rag-data`) | `.input/`, `DonePDF/`, `.RAG_Json/`, `.vector_db/`, `references.bib`, `._trash/` | code |
| **Regenerable cache** | `$RAG_INDEX_DIR` (`$RAG_DATA_ROOT/.retrieval_index`) | read-only retrieval cache (`vectors.f32.npy`, `index.sqlite`, `meta.json`) | anything unique |
| **Models** | `$RAG_MODELS_DIR` (`~/rag-models`) | bge-m3, reranker, MinerU/ModelScope cache (~5 GB) | git repositories |

Rules of thumb:

- If it can be rebuilt from the PDFs, it does not belong in git.
- If it would be expensive to re-download, keep it out of `~` only if `~` is small; the
  only hard rule is *one* location, referenced by an env var.
- The vector store is the single source of truth for embeddings — never keep a second copy.

## Module map

```
src/rag_literature/
├── config.py              all paths/parameters, env-driven, generic defaults
├── cli.py                 entry point: process everything in .input/
├── refs.py                unified references.bib builder (incremental, refuses to
│                          rebuild from scratch if the base file cannot be parsed)
├── pipeline/
│   ├── orchestrator.py    per-PDF state machine, dedup hashes, resumable
│   ├── step2_mineru.py    PDF → structured blocks (GPU), injects MODELSCOPE_CACHE
│   ├── step3_metadata.py  title/venue/year/DOI (Semantic Scholar → Crossref), naming
│   ├── step4_clean.py     math protection + sentence splitting (no translation)
│   ├── step6_rag_update.py embeddings → Chroma, incremental by file hash, flush fix
│   └── step7_cleanup.py   .tmp → ._trash
├── models/
│   ├── llama.py           llama.cpp server lifecycle + HTTP client
│   ├── manager.py         model/job management helpers
│   ├── translator.py      optional EN↔ZH translation (batch, token-aware)
│   └── journal_tier.py    optional JCR quartile lookup (needs your own API key)
├── qa/
│   ├── retriever.py       read-only hybrid retrieval (dense + BM25 → RRF → window)
│   ├── serve.py           tiny HTTP service keeping the model warm
│   ├── engine.py          one-shot CLI query (loads the model each call)
│   └── citation_index.py  paper → references map extracted from the JSON
├── tools/
│   ├── export_index.py    Chroma → read-only cache (vectors + FTS5 + meta)
│   ├── manifest_build.py  completeness index over the data root
│   └── eval_retrieval.py  synthetic golden-set evaluation (A/B/C groups)
└── utils/                 file helpers, naming/year parsing, language detection
```

## Data flow

```
 .input/a.pdf
   │ step2  MinerU (subprocess, conda run)
   ▼
 .tmp/a/…_content_list_v2.json
   │ step3  metadata lookup + naming  → DonePDF/<Book>_<Year>.pdf
   ▼
 .RAG_Json/<Book>_<Year>_v2.json     ← title/venue/year/doi + blocks + sentences
   │ step4  math protection, sentence split, English text only
   │ step6  bge-m3 embeddings → .vector_db/chroma_data
   │ step6b references.bib (incremental merge, never a blind rebuild)
   │ step6c export_index → RAG_INDEX_DIR (vectors.f32.npy + index.sqlite + meta.json)
   │ step8  .tmp → ._trash ; .input original → ._trash/input_done_<ts>/
   ▼
 qa/retriever.py  (mmap dense + sqlite FTS5 → RRF → ±window)  ← qa/serve.py (HTTP)
```

## State files (what makes runs resumable)

| File | Purpose |
|---|---|
| `.vector_db/pipeline_state.json` | per-PDF step status (`step2`…`step8`, `failed`), so a crashed run resumes instead of re-parsing |
| `.vector_db/dedup_hashes.json` | MD5 of each processed PDF — a re-added identical file is skipped |
| `.vector_db/rag_state.json` | ingestion bookkeeping for the vector store |
| `.vector_db/chroma_data/` | Chroma persistent store: **the vectors themselves live in the HNSW segment files here** |
| `RAG_INDEX_DIR/meta.json` | count / dim / model / `exported_at`; written last, and the serve process watches its mtime for hot reload |

Because state lives in the data root, you can delete the code directory and re-clone
without losing work — and you can run the same code against two different corpora by
pointing `RAG_DATA_ROOT` at two directories. That is exactly how this project is tested
in isolation.

## Design decisions worth knowing (and their reasons)

1. **Data/code separation** — the pipeline is deliberately *not* a package that owns your
   PDFs. Every path comes from config, so testing against a scratch root cannot corrupt
   the real library. (A hard-coded cache path once overwrote a real 199 k-chunk index
   with a 31-row test index; see `retrieval.md`.)
2. **Translation and cnRead generation are outside the pipeline** — step 6 embeds English
   text only; translations were 81 % of a batch's wall-clock time and contributed nothing
   to retrieval. They remain available as standalone tools.
3. **Images are opt-in** — nothing consumed them, so keeping them only inflated the data
   root. `RAG_KEEP_IMAGES=1` restores the old behaviour.
4. **Retrieval reads a cache, not Chroma** — opening Chroma's persistent client takes a
   write lock; read-only cache files make queries safe from any process, at any time.
5. **The bibliography is never blindly rebuilt** — `refs.py` refuses to regenerate from
   scratch when the existing base cannot be parsed, because a silent rebuild once replaced
   390 real titles with file names.
