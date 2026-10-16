#!/usr/bin/env python3
"""把 Chroma 向量库导出成**只读检索缓存**（供查询路径使用）。

## 为什么需要
Chroma 的 `PersistentClient` 打开库必须拿写锁，而 `/data` 在会话沙箱里是只读挂载
→ preset 每次检索都会撞 `attempt to write a readonly database`，必须提权
（`auto-approve` 判断约 20 秒，或弹人工确认）。

导出到 `~/.cache/rag/index/` 之后，查询路径只做「只读 mmap + sqlite ro」，
**完全不需要提权**，且比走 Chroma 更快（无 Chroma 开销）。

## 产物（默认 `~/.cache/rag/index/`）
| 文件 | 内容 |
|---|---|
| `vectors.f32.npy` | (N, 1024) float32，行序 = `rec.idx`，可 `np.load(mmap_mode='r')` |
| `index.sqlite` | `rec(idx,id,book,year,page,file,text)` + FTS5 `fts(text)` 供 BM25 |
| `meta.json` | count / dim / model / exported_at / collection |

## 用法
```bash
python -m rag_literature.tools.export_index            # 全量（实测约 4 分钟）
python -m rag_literature.tools.export_index --check    # 只比对条数，判断要不要重导
```
`--check` 退出码：0 = 缓存是最新的；3 = 需要重导。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from ..config import (CHROMA_COLLECTION_NAME, CHROMA_DATA_DIR, EMBEDDING_MODEL,
                      RETRIEVAL_INDEX_DIR)

DEFAULT_OUT = RETRIEVAL_INDEX_DIR   # 跟随 RAG_DATA_ROOT / RAG_INDEX_DIR
PAGE = 20000

SCHEMA = """
CREATE TABLE IF NOT EXISTS rec (
    idx  INTEGER PRIMARY KEY,
    id   TEXT NOT NULL,
    book TEXT, year TEXT, page INTEGER, file TEXT,
    text TEXT
);
CREATE INDEX IF NOT EXISTS rec_book ON rec(book);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, content='rec', content_rowid='idx');
"""


def chroma_record_count(db_dir: Path | None = None) -> int:
    """直接只读读 sqlite 的记录数（不需要 Chroma 写锁）。"""
    f = Path(db_dir or CHROMA_DATA_DIR) / "chroma.sqlite3"
    c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    try:
        return c.execute("select count(*) from embeddings").fetchone()[0]
    finally:
        c.close()


def _cached_count(out: Path) -> int | None:
    m = out / "meta.json"
    if not m.exists():
        return None
    try:
        return int(json.loads(m.read_text(encoding="utf-8")).get("count"))
    except Exception:
        return None


def check(out: Path) -> int:
    live = chroma_record_count()
    cached = _cached_count(out)
    if cached is None:
        print(f"缓存不存在或不可读：{out}（live={live}）")
        return 3
    if cached != live:
        print(f"缓存过期：缓存 {cached} 条 vs 库内 {live} 条 → 需要重导")
        return 3
    print(f"缓存最新：{cached} 条")
    return 0


def export(out: Path, page: int = PAGE, force: bool = False) -> int:
    import chromadb
    import warnings
    warnings.filterwarnings("ignore")

    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 防呆闸：新导出量不足现有缓存的 50% 时拒绝覆盖。
    # 2026-10-14 踩坑 —— 在隔离测试根下跑 pipeline 时，缓存路径写死到 ~/.cache/rag/index，
    # 测试用的小索引（31 条）把真实缓存（199,493 条）覆盖掉了，且**没有任何提示**。
    # 宁可报错也不要静默毁掉检索层。
    prev = _cached_count(out)
    live = chroma_record_count()
    if prev and live < prev * 0.5 and not force:
        raise RuntimeError(
            f"拒绝导出：现有缓存 {prev} 条，而库内仅 {live} 条（<50%）。\n"
            f"  若确实是一次大规模缩库，请显式加 --force；\n"
            f"  若你正在隔离测试，说明缓存路径没跟随 RAG_DATA_ROOT（见 config.RETRIEVAL_INDEX_DIR）。")

    # settings 必须与 step6 的 _get_chroma_client() 一致，否则同进程内再建客户端会报
    # "An instance of Chroma already exists ... with different settings"
    # （pipeline 的 step6c 就是在 step6 之后同进程调用的，2026-10-14 实际踩到）
    from chromadb.config import Settings as ChromaSettings
    client = chromadb.PersistentClient(
        path=str(CHROMA_DATA_DIR), settings=ChromaSettings(anonymized_telemetry=False))
    col = client.get_collection(CHROMA_COLLECTION_NAME)
    total = col.count()
    print(f"Chroma 集合 {CHROMA_COLLECTION_NAME}: {total} 条 → 导出到 {out}", flush=True)

    npy = out / "vectors.f32.npy"
    dbf = out / "index.sqlite"
    for p in (npy, dbf):
        if p.exists():
            p.unlink()
    db = sqlite3.connect(dbf)
    db.executescript(SCHEMA)

    arr = None
    written = 0
    norm_lo, norm_hi = None, None
    t0 = time.time()
    for off in range(0, total, page):
        r = col.get(limit=page, offset=off,
                    include=["embeddings", "documents", "metadatas"])
        ids = r["ids"]
        if not ids:
            break
        emb = np.asarray(r["embeddings"], dtype=np.float32)
        if arr is None:
            dim = emb.shape[1]
            arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(total, dim))
            print(f"  向量维度 {dim}，预分配 {total}×{dim} = {total*dim*4/2**20:.0f} MiB", flush=True)
        arr[written:written + len(ids)] = emb

        n = np.linalg.norm(emb, axis=1)
        lo, hi = float(n.min()), float(n.max())
        norm_lo = lo if norm_lo is None else min(norm_lo, lo)
        norm_hi = hi if norm_hi is None else max(norm_hi, hi)

        rows = []
        for i, _id, doc, md in zip(range(written, written + len(ids)), ids,
                                   r["documents"], r["metadatas"]):
            md = md or {}
            pg = md.get("page")
            rows.append((i, str(_id), str(md.get("book") or ""), str(md.get("year") or ""),
                         int(pg) if isinstance(pg, (int, float)) else None,
                         str(md.get("file") or ""), doc or ""))
        db.executemany("INSERT INTO rec(idx,id,book,year,page,file,text) VALUES (?,?,?,?,?,?,?)", rows)
        db.commit()
        written += len(ids)
        el = time.time() - t0
        print(f"  {written}/{total}  {written/el:7.1f} 条/s  已用 {el/60:.1f} 分  预计剩余 {(total-written)/max(written/el,1e-9)/60:.1f} 分",
              flush=True)

    if arr is not None:
        arr.flush()
        del arr
    print("  建 FTS5 全文索引（BM25）…", flush=True)
    t1 = time.time()
    db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"  FTS5 完成 {time.time()-t1:.1f}s", flush=True)

    meta = {
        "collection": CHROMA_COLLECTION_NAME,
        "count": written,
        "dim": int(np.load(npy, mmap_mode="r").shape[1]),
        "model": str(EMBEDDING_MODEL),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vector_norm_min": norm_lo,
        "vector_norm_max": norm_hi,
        "npy": npy.name,
        "sqlite": dbf.name,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{written} 条，用时 {(time.time()-t_start)/60:.1f} 分；向量范数 {norm_lo:.4f}~{norm_hi:.4f}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="导出只读检索缓存")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="只比对条数，判断是否需要重导")
    ap.add_argument("--page", type=int, default=PAGE)
    ap.add_argument("--force", action="store_true", help="允许导出量大幅小于现有缓存时覆盖")
    a = ap.parse_args()
    if a.check:
        return check(a.out)
    export(a.out, a.page, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
