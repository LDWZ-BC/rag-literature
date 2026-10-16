# Retrieval layer

Two paths exist on purpose: **Chroma for writing**, a **read-only cache for reading**.

## Why not query Chroma directly

`chromadb.PersistentClient` takes a write lock when it opens the store. Consequences:

- Queries fail (or need elevated permissions) whenever the data directory is mounted
  read-only — a very common setup when the library lives on a NAS.
- A query process and an ingest process cannot share the store.
- Any bug in a query path can, in principle, corrupt the store.

So the pipeline treats Chroma as the *ingest* format, and exports a read-only snapshot that
any process can query safely:

```
Chroma (.vector_db/chroma_data)          ← written by step 6, needs a write lock
   │  python -m rag_literature.tools.export_index        (~3–5 min / 200 k chunks)
   ▼
$RAG_INDEX_DIR/
├── vectors.f32.npy     (N, 1024) float32, row order == rec.idx, mmap-able
├── index.sqlite        rec(idx,id,book,year,page,file,text) + FTS5 table for BM25
└── meta.json           count / dim / model / exported_at / collection   (written LAST)
```

`meta.json` is written last on purpose: the serve process watches its mtime, so a
half-written export can never be picked up. `export_index --check` exits **3** when the
cache is stale, which is convenient in scripts.

The exporter also refuses to overwrite a much larger existing cache (guard against the
"31-row test index silently replacing a 199 k-row real index" accident). Override with
`--force` only when you really did shrink the library.

## Query pipeline

```
query
 ├─ dense  : bge-m3 encode → mmap float32 dot product → top-pool
 ├─ lexical: sqlite FTS5 (BM25) → top-pool
 ├─ RRF fuse (k = 60)                       ← rank fusion, no score normalisation needed
 ├─ optional cross-encoder rerank           ← OFF by default, see below
 └─ window expansion: ids are "{file}_{n}", so take neighbours n±window
                                                  → passage-level context, not one sentence
```

Results carry `citation` (`Book_Year`, the key used in `references.bib`), `book`, `year`,
`page`, `text` (the expanded window) and per-stage scores, so you can see *why* something
was retrieved.

## Service (recommended)

```bash
python -m rag_literature.qa.serve --host 127.0.0.1 --port 8077
```

| Endpoint | Parameters |
|---|---|
| `GET /health` | — → `{"ok":true,"count":N,"reranker":bool,"device":"cuda"}` |
| `GET /search` | `q` (required), `k`, `pool`, `hybrid` (0/1), `rerank` (0/1), `window`, `project` |
| `POST /search` | same fields as JSON |

Why a service rather than a CLI call: the bottleneck is **model loading** (bge-m3 ≈ 6.6 s,
reranker ≈ 1.4 s), not retrieval (0.3–0.7 s). One-shot CLI calls therefore cost 19–34 s;
the service answers in 0.35–0.69 s. The service reloads the index by itself when
`meta.json` changes, so ingesting does not require a restart.

A user-level systemd unit is enough (bind to `127.0.0.1`, `Restart=always`); with
`loginctl enable-linger` it also survives logout.

## Evaluation harness

```bash
python -m rag_literature.tools.eval_retrieval -n 100 -k 10
```

Three complementary golden sets are built from your own corpus:

| Group | Question | Target |
|---|---|---|
| **A** paragraph → paper | can retrieval find the source of a passage? | `book` (excluding the source's own neighbours) |
| **B** rare term → paper | exact-term lookup (the citation-finding use case) | **set** of acceptable papers |
| **C** Chinese question → English paper | cross-lingual recall | user-supplied (`ZH_CASES`, empty by default) |

Reference numbers from the corpus this code was developed on (RTX 3090, ~200 k chunks):

| Group | Config | R@1 | R@5 | R@10 | MRR | ms/query |
|---|---|---|---|---|---|---|
| A paragraph→paper | dense | 0.560 | 0.830 | 0.910 | 0.681 | 103 |
| | **hybrid** | **0.590** | 0.810 | 0.880 | **0.684** | 241 |
| | hybrid+rerank | 0.480 | 0.740 | 0.820 | 0.576 | 676 |
| B term→paper | dense | 0.130 | 0.380 | 0.480 | 0.241 | 87 |
| | **hybrid** | **0.360** | **0.870** | **0.940** | **0.615** | 68 |
| | hybrid+rerank | 0.380 | 0.600 | 0.710 | 0.466 | 504 |
| C zh→en (n=12) | dense | 0.917 | 0.917 | 1.000 | 0.925 | 86 |
| | hybrid | 0.833 | 0.917 | 1.000 | 0.863 | 68 |

Two conclusions that transferred to every later experiment:

1. **Hybrid retrieval is not optional.** For rare exact terms (the "find the paper that
   introduced this term" use case) R@5 goes **0.38 → 0.87** and R@10 **0.48 → 0.94**. Dense
   embeddings are close to useless for strings like `Peierls-Nabarro` (R@1 = 0.13).
2. **A cross-encoder reranker hurt paper-level recall in all three groups** (and was ~10×
   slower). The model scores sensibly, but its notion of semantic similarity overrides the
   "same paper as the one cited" signal. It stays off (`RAG_RERANK=1` to experiment) — and
   if you swap in another reranker, **re-run this evaluation** instead of trusting intuition.

Golden-set gotcha worth copying: in group B a rare term often appears in *several* papers.
Scoring only "the alphabetically first" paper marks correct answers as misses — targets must
be **acceptable sets**. Getting this wrong once produced a wrong conclusion about reranking.
