# Troubleshooting / the mistakes that cost the most

Each section is a failure that actually happened, with the diagnostic command and the fix.

---

## 1. MinerU silently re-downloads ~5 GB of models

**Symptom.** Parsing works, but the model cache reappears under `~/.cache/modelscope`, disk
usage jumps, and every new shell can trigger a fresh download.

**Cause.** MinerU resolves its cache from the `MODELSCOPE_CACHE` environment variable, and
the pipeline starts it via `conda run` inside `subprocess` — a *non-interactive, non-login*
shell. Anything exported only from `~/.bashrc` / `~/.profile` never reaches it.

```bash
# proof: same machine, same interpreter, different answer
python -c "from modelscope.hub.utils.utils import get_cache_dir; print(get_cache_dir())"
#   (no env var)      → /home/you/.cache/modelscope/hub
#   MODELSCOPE_CACHE=… → /your/rag-models/modelscope
```

**Fix (already in this repo).** `config.MINERU_MODELSCOPE_CACHE` has a permanent default and
`pipeline/step2_mineru.py` injects it into the subprocess environment, then logs it:

```
MinerU 模型缓存: /home/you/rag-models/modelscope
```

If you see a warning that the directory does not exist, MinerU will download into it — make
sure that path is the volume you intended.

**Related trap.** `~/mineru.json` (`models-dir`, `model-source`) is only consulted when
`MINERU_MODEL_SOURCE=local`. With the default `modelscope` source, that file is *ignored* and
the cache path above is what matters. Use `local` if you want a fully offline, no-network
run — at the cost of a hard error (instead of an automatic download) when a file is missing.

---

## 2. Never rebuild a vector store in place

**Symptom.** The index directory grows without bound: a *sparse* file of 6.8 TB, ~753 GB
actually occupied, root filesystem full. Deleting the segment directory "fixes" it — and
silently destroys all vectors.

**Cause.** HNSW-based stores (hnswlib, and therefore Chroma's persistent segments) do not
shrink when you delete and re-add rows: link lists get rewritten and the file keeps growing.

**Rules.**

1. Build a new store in a **new directory**; verify; then switch the pointer. Never
   `delete + re-add` in place.
2. **Never delete `.vector_db/chroma_data/<segment-uuid>/`.** In this Chroma version the
   vectors live in the segment's HNSW files (`data_level0.bin`), *not* in `chroma.sqlite3` —
   the `embeddings` table only holds ids and metadata. Deleting a segment silently degrades
   the store to a few hundred rows: retrieval still returns results, coverage collapses to
   ~0.4 %. A sqlite backup cannot restore it.

```bash
# rebuild into a fresh directory, verify, then switch
RAG_CHROMA_DIR=/your/rag-data/.vector_db/chroma_data.new \
  python -c "from rag_literature.pipeline.step6_rag_update import incremental_update; \
             print(incremental_update(force=True))"
# verify dimension == 1024 and the row count matches your corpus, then:
#   mv chroma_data chroma_data.old && mv chroma_data.new chroma_data
```

---

## 3. The last few hundred chunks are never indexed

**Symptom.** Coverage is *almost* complete: the tail of the most recently processed paper
(typically a few hundred sentences) can never be retrieved, and nothing reports an error.

**Cause.** Chroma's `PersistentLocalHnswSegment._apply_batch()` only calls `_persist()` once
`_num_log_records_since_last_persist >= sync_threshold` (default **1000**). Everything written
after the last persist stays out of the index.

**Check coverage** (stored rows vs indexed rows):

```bash
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

`missing > 0` means the index lags. **Fix (implemented in `step6_rag_update.py`):** after the
incremental update, re-upsert ~1500 already-present records to push the counter past the
threshold and force a final persist. Measured effect: coverage 99.63 % → **100.00 %**.

If you write to the store by hand (e.g. patching metadata), do the same flush or re-run the
incremental update afterwards.

---

## 4. The retrieval cache must follow the data root

**Symptom.** Your real retrieval cache (199 k rows) suddenly contains 31 rows. No warning.

**Cause.** A hard-coded cache path. Running the pipeline against an isolated test root
(`RAG_DATA_ROOT=/tmp/test`) overwrote the production cache with test data.

**Fix (implemented).** `RETRIEVAL_INDEX_DIR = RAG_INDEX_DIR` → `$RAG_DATA_ROOT/.retrieval_index`
→ `~/.cache/rag/index`, in that order, plus a guard in `export_index`: it refuses to replace a
much larger cache unless you pass `--force`. When you introduce a new cache/state path, make
it derive from `RAG_DATA_ROOT` — this bug class shows up repeatedly.

---

## 5. Cross-encoder rerankers can make retrieval worse

**Symptom.** Adding a reranker "feels" like it should improve results, but citations point at
the wrong paper.

**Measured result** on this corpus: `bge-reranker-v2-m3` **reduced paper-level recall in all
three test groups** and was ~10× slower. The model's semantic-similarity preference overrides
the "this is the paper that was cited" signal.

**Action.** Keep reranking off (default). If you try another model, re-run
`python -m rag_literature.tools.eval_retrieval -n 100 -k 10` and compare — and when you
evaluate term-based queries, remember the target is a **set** of acceptable papers.

---

## 6. GPU memory, not CPU, limits ingestion

MinerU peaks around 15 GB VRAM. Two concurrent parses will OOM on a 24 GB card that is also
serving other users.

- The orchestrator serialises parsing with a lock; `RAG_PIPELINE_WORKERS>1` only overlaps the
  I/O steps with parsing — it does not let two MinerU instances share the GPU.
- Before starting a batch, check whether something else is already using the GPU
  (`nvidia-smi`), especially on a shared machine.

---

## 7. "The pipeline says it processed nothing"

Look at `.input/` — a successful paper's original is moved to
`._trash/input_done_<timestamp>/`. Step 3 moves MinerU's *copy* into `DonePDF/`, so an
untouched `.input/` used to make completed runs look unfinished. Nothing is deleted; restore
from `._trash/` if you need the file back in place.

---

## 8. `references.bib` degraded into file names

**Symptom.** Citation keys still resolve, but every `title` is a CamelCase file name.

**Cause.** A rebuild that silently started from an empty base after a parse failure
(`merged = {}`), then reconstructed entries from JSON whose `meta.title` coverage was ~1 %.

**Fix (implemented).** The builder refuses to rebuild when the existing base cannot be parsed,
prefers `meta.title` over the file name, and warns when > 50 % of titles look like file names.
**Always back up `references.bib` before maintenance**, and audit after:

```bash
python -m rag_literature.tools.bib_audit --audit --kinds title   # add --fix to write back
```

---

## 9. Images and other "might be useful later" outputs

MinerU extracts images whether you want them or not. Nothing in this pipeline consumes them,
so by default they are **not** moved out of `.tmp/` and disappear with it
(`RAG_KEEP_IMAGES=1` to keep them). The same principle applies to translation: if no consumer
reads an output, generating it is pure cost.
