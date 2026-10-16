# The pipeline

One entry point (`python -m rag_literature.cli`) processes every PDF in
`$RAG_DATA_ROOT/.input/`. Steps are numbered 2, 3, 4, 6, 6b, 6c, 8 because steps 1, 5 and 7
were removed after measurement — the numbers were kept so old state files still make sense.

| Step | Input | Output | Cost |
|---|---|---|---|
| **2** MinerU parse | `.input/*.pdf` | `.tmp/<pdf>/<pdf>/…_content_list_v2.json` | **GPU-bound**, ~15 GB VRAM peak, seconds–minutes per paper |
| **3** metadata + naming | MinerU JSON + PDF | `DonePDF/<Book>_<Year>.pdf`, `.RAG_Json/*_v2.json` | network: Semantic Scholar, then Crossref by DOI; rate-limited |
| **4** clean | `_content_list_v2.json` | `_v2.json` with `blocks[]` + `sentences[]` | CPU, fast |
| **6** embed + index | `_v2.json` | `.vector_db/chroma_data` | GPU (bge-m3), ~1 k sentences/s on a 3090 |
| **6b** bibliography | all `_v2.json` | `references.bib` | fast, incremental |
| **6c** export retrieval cache | `chroma_data` | `$RAG_INDEX_DIR/{vectors.f32.npy,index.sqlite,meta.json}` | a few minutes per 200 k chunks |
| **8** cleanup | `.tmp/`, `.input/` | moves to `._trash/` | fast |

## Step 2 — MinerU (the GPU step)

The pipeline shells out to a **separate conda env**:

```bash
conda run -n "$RAG_MINERU_CONDA_ENV" mineru -p <pdf> -o <out> -l ch -b pipeline -m txt -f true -t true
```

- `-b pipeline` (not the default hybrid engine) is ~1.8× faster for research PDFs.
- `-m txt` uses the PDF's text layer; `-m ocr` forces OCR. Some publisher PDFs have a
  broken `ToUnicode` map and extract as private-use garbage under `txt` — those need `ocr`.
  Retry a failed paper with `RAG_MINERU_METHOD=ocr` before giving up on it.
- Only `_content_list_v2.json` is kept; layout/span/markdown intermediates are deleted
  immediately (they were the bulk of the disk usage).

**VRAM is the real concurrency limit**, not CPU: one MinerU instance per GPU. The
orchestrator enforces this with a process-wide `_mineru_lock`; extra `RAG_PIPELINE_WORKERS`
only overlap the I/O steps with parsing.

## Step 3 — metadata and naming

Tries Semantic Scholar (title/DOI → venue, year, authors), then Crossref by DOI, and falls
back to the PDF's own metadata. `title` is preferred over the file name; if a title looks
like a CamelCase file name the pipeline warns, because that is the signature of a failed
metadata extraction.

Everything is renamed to `<Book>_<Year>` and the PDF is moved into `DonePDF/`. That name is
the join key used by the vector store metadata, the bibliography keys
(`[@Book_Year]`) and the citation index — so it is decided exactly once, here.

Optional: `RAG_JCR=1` adds a JCR quartile lookup. It needs **your own** EasyScholar key
(`RAG_EASYSCHOLAR_KEY`); without a key the lookup is skipped.

## Step 4 — cleaning (no translation)

Two jobs:

1. **Protect math** before touching text: `$…$` regions are replaced by ASCII tokens
   (`[[MATH0]]`) so that math symbols are not mangled, then restored.
2. **Split sentences** without breaking inside math or abbreviations.

The output keeps the English text; `zh` is deliberately left empty. Reason (measured):
translation ran at ~4.5 sentences/s and consumed **81 %** of an 80-minute batch, while step 6
only embeds the English text — `documents.append(en_text)`. Translation therefore had zero
effect on retrieval quality. Use `models/translator.py` when you actually want a translation.

## Step 6 — embeddings and incremental indexing

- Model: **bge-m3**, 1024-d, loaded from `RAG_EMBEDDING_MODEL` with HF offline flags set.
- Collection name: `RAG_CHROMA_COLLECTION` (default `literature_sentences`).
- Ids are `"{Book_Year}_{n}"` — the naming convention the retriever relies on to expand
  context windows (±N neighbouring sentences of the same paper).
- Incremental: each `_v2.json` is hashed; unchanged files are skipped
  (`dedup_hashes.json`), so re-running is cheap and idempotent.

Two non-obvious behaviours you should know before touching the store — both are explained
in `troubleshooting.md`:

- Chroma only flushes every 1000 writes, so the last few hundred records of a run would
  never become searchable without an explicit final flush (implemented).
- A vector store must never be rebuilt *in place*. Build into a new directory, verify the
  counts, then switch the pointer.

## Step 6b — the unified bibliography

`refs.py` merges per-paper entries into a single `references.bib` keyed by `Title_Year`.
It is **incremental by design** and refuses to rebuild from scratch when the existing base
file cannot be parsed. That guard exists because a silent rebuild once replaced 390 real
titles with CamelCase file names — a failure that is invisible until citations look wrong.

## Step 6c — export the read-only retrieval cache

See [`retrieval.md`](retrieval.md). Also runnable on demand:

```bash
python -m rag_literature.tools.export_index          # rebuild
python -m rag_literature.tools.export_index --check   # exit 3 → stale
```

## Step 8 — cleanup

`.tmp/` is moved into `._trash/` (never deleted outright), and the `.input/` original is
moved to `._trash/input_done_<timestamp>/` after step 3 succeeded. This matters more than
it sounds: step 3 moves MinerU's *copy* of the PDF into `DonePDF/`, so an untouched
`.input/` made every finished run look like it had "not processed anything".

## Measured costs (for planning)

| Operation | Throughput | Note |
|---|---|---|
| MinerU parse | dominated by VRAM | one instance per GPU (~15 GB peak) |
| bge-m3 embedding | ~940 sentences/s | RTX 3090, 200 k sentences ≈ 3–4 min |
| Chroma write to a **network** volume | ~35 rows/s | local disk was ~149 rows/s in the same test |
| Retrieval export (200 k chunks) | ~3–5 min | re-run after every ingest |
| Query with warm service | 0.35–0.69 s | ~20 s if the model has to be loaded per query |

If ingestion is dominated by Chroma writes, build the store on local disk and publish it to
the network volume afterwards — writing 200 k rows straight to NFS is the slow path.
