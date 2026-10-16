#!/usr/bin/env python3
"""只读混合检索器（给 preset / 问答 / 写作审查用）。

## 为什么不是直接查 Chroma
Chroma 的 `PersistentClient` 打开库必须拿写锁，而数据目录常常是只读挂载 →
每次检索都要提权（auto-approve 判断约 20 秒，或弹人工确认）。本模块改读
`tools/export_index.py` 导出的只读缓存，**查询路径完全不写盘、不需要提权**。

## 检索链路
```
query ─┬─ 稠密召回：bge-m3 编码 → mmap 向量点积 top-pool ─┐
       └─ 词法召回：sqlite FTS5（BM25）top-pool ──────────┤
                                                          ├─ RRF 融合
                                                          ├─ 可选 cross-encoder 重排
                                                          └─ 窗口扩展（±window 句）→ 结果
```
窗口扩展靠 id 命名约定：`{file}_{n}`，取同文件 n±window 的句子拼成上下文。

## 输出
每条含 `citation`（`{book}_{year}`，与 `references.bib` / `[@Title_Year]` 对齐）、
`book`/`year`/`page`/`text`（窗口文本）/`scores`（dense·lexical·fused·rerank）。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np

# 离线优先（模型全本地）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    from ..config import RETRIEVAL_INDEX_DIR as DEFAULT_INDEX_DIR, MODELS_DIR
except Exception:                                     # 独立运行时兜底
    DEFAULT_INDEX_DIR = Path.home() / ".cache" / "rag" / "index"
    MODELS_DIR = Path.home() / "rag-models"
DEFAULT_RERANK_DIR = MODELS_DIR / "bge-reranker-v2-m3"
RRF_K = 60
# 重排默认关：2026-10-14 实测 bge-reranker-v2-m3 在三组用例上**全部**降低文献级
# recall（A 组 R@5 0.81→0.74、B 0.80→0.53、C R@1 0.83→0.75），且慢约 10 倍。
# 需要时 RAG_RERANK=1 或 search(rerank=True) 打开。
RERANK_DEFAULT = os.environ.get("RAG_RERANK", "0").strip().lower() in ("1", "true", "yes", "on")
_ID_RE = re.compile(r"^(?P<file>.+)_(?P<n>\d+)$")


class Retriever:
    def __init__(self, index_dir: Path | str | None = None,
                 rerank_model: str | Path | None = None, device: str | None = None):
        self.dir = Path(index_dir or DEFAULT_INDEX_DIR)
        meta_f = self.dir / "meta.json"
        if not meta_f.exists():
            raise FileNotFoundError(
                f"检索缓存不存在：{self.dir}\n先跑：python -m rag_literature.tools.export_index")
        self.meta = json.loads(meta_f.read_text(encoding="utf-8"))
        self.vecs = np.load(self.dir / self.meta["npy"], mmap_mode="r")
        self.dim = int(self.meta["dim"])
        self.count = int(self.meta["count"])
        # 只读连接（uri=ro）——沙箱里也能开。
        # check_same_thread=False：常驻服务用 ThreadingHTTPServer，每个请求一个新线程；
        # 只读连接配一把锁即可安全共享（sqlite3 默认 serialized 模式）。
        self.db = sqlite3.connect(
            f"file:{self.dir / self.meta['sqlite']}?mode=ro", uri=True, check_same_thread=False)
        self._db_lock = threading.Lock()
        self.db.row_factory = sqlite3.Row
        self._model = None
        self._reranker = None
        self._rerank_dir = Path(rerank_model or os.environ.get("RAG_RERANK_MODEL") or DEFAULT_RERANK_DIR)
        self._dense_scores = None
        self._device = device

    # ── 模型（懒加载）─────────────────────────────────────────────
    @property
    def model(self):
        if self._model is None:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            import torch
            from sentence_transformers import SentenceTransformer
            from ..config import EMBEDDING_MODEL
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            self._model = SentenceTransformer(self.meta.get("model") or EMBEDDING_MODEL, device=dev)
        return self._model

    @property
    def reranker(self):
        """cross-encoder 重排器；模型目录不存在则返回 None（静默降级为不重排）。"""
        if self._reranker is None and self._rerank_dir.is_dir():
            try:
                from sentence_transformers import CrossEncoder
                import torch
                dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                self._reranker = CrossEncoder(str(self._rerank_dir), device=dev)
            except Exception:
                self._reranker = False
        return self._reranker or None

    # ── 两条召回 ─────────────────────────────────────────────────
    def dense(self, query: str, pool: int = 50, book_filter: list[str] | None = None) -> list[int]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype=np.float32)[0]
        # 分块点积，避免一次物化整段
        scores = np.empty(self.count, dtype=np.float32)
        step = 50000
        for s in range(0, self.count, step):
            e = min(s + step, self.count)
            scores[s:e] = np.asarray(self.vecs[s:e]) @ q
        if book_filter:
            allowed = self._idx_for_books(book_filter)
            mask = np.zeros(self.count, dtype=bool)
            if allowed.size:
                mask[allowed] = True
            scores = np.where(mask, scores, -np.inf)
        self._dense_scores = scores          # 供 similarity 用（向量已归一化 → 点积即余弦）
        pool = min(pool, self.count)
        part = np.argpartition(-scores, pool - 1)[:pool]
        return [int(i) for i in part[np.argsort(-scores[part])]]

    def lexical(self, query: str, pool: int = 50, book_filter: list[str] | None = None) -> list[int]:
        terms = [t for t in re.findall(r"[\w\u4e00-\u9fff\-]+", query) if len(t) > 1]
        if not terms:
            return []
        expr = " OR ".join(f'"{t}"' for t in terms[:12])
        sql = ("select f.rowid as idx, bm25(fts) as s from fts f "
               "where fts match ? order by s limit ?")
        try:
            with self._db_lock:
                rows = self.db.execute(sql, (expr, pool)).fetchall()
        except Exception:
            return []
        out = [r["idx"] for r in rows]
        if book_filter:
            keep = set(self._idx_for_books(book_filter).tolist())
            out = [i for i in out if i in keep]
        return out

    # ── 辅助 ─────────────────────────────────────────────────────
    _book_cache: dict[tuple, np.ndarray] = {}

    def _idx_for_books(self, books: list[str]) -> np.ndarray:
        key = tuple(sorted(books))
        if key not in self._book_cache:
            ph = ",".join("?" * len(key))
            with self._db_lock:
                rows = self.db.execute(
                    f"select idx from rec where book in ({ph})", key).fetchall()
            self._book_cache[key] = np.asarray([r["idx"] for r in rows], dtype=np.int64)
        return self._book_cache[key]

    def _rows(self, idxs: list[int]) -> dict[int, sqlite3.Row]:
        if not idxs:
            return {}
        ph = ",".join("?" * len(idxs))
        with self._db_lock:
            return {r["idx"]: r for r in self.db.execute(
                f"select * from rec where idx in ({ph})", idxs).fetchall()}

    def _expand(self, row: sqlite3.Row, window: int) -> tuple[str, list[int]]:
        """把命中句扩展成 ±window 句的上下文。返回 (合并文本, 涉及页号)。"""
        if window <= 0:
            return row["text"] or "", ([row["page"]] if row["page"] is not None else [])
        m = _ID_RE.match(row["id"] or "")
        if not m:
            return row["text"] or "", ([row["page"]] if row["page"] is not None else [])
        f, n = m.group("file"), int(m.group("n"))
        ids = [f"{f}_{i}" for i in range(max(0, n - window), n + window + 1)]
        ph = ",".join("?" * len(ids))
        with self._db_lock:
            got = self.db.execute(
                f"select idx, id, page, text from rec where id in ({ph})", ids).fetchall()
        order = {f"{f}_{i}": i for i in range(max(0, n - window), n + window + 1)}
        got = sorted(got, key=lambda r: order.get(r["id"], 0))
        text = " ".join((r["text"] or "").strip() for r in got).strip()
        pages = sorted({r["page"] for r in got if r["page"] is not None})
        return text, pages

    # ── 主入口 ───────────────────────────────────────────────────
    def search(self, query: str, k: int = 8, pool: int = 50,
               project_scope: list[str] | None = None, hybrid: bool = True,
               rerank: bool | None = None, window: int = 1) -> list[dict]:
        """检索。

        Args:
            query: 查询串（中英均可）
            k: 返回条数
            pool: 每条召回通道的候选数
            project_scope: 限定 `book` 列表（项目语料作用域）；None = 全库
            hybrid: 是否加词法召回（FTS5/BM25）并做 RRF 融合
            rerank: 是否用 cross-encoder 重排；None = 有模型就重排
            window: 命中句向两侧扩展的句数（0 = 只返回该句）
        """
        dense = self.dense(query, pool=pool, book_filter=project_scope)
        lex = self.lexical(query, pool=pool, book_filter=project_scope) if hybrid else []

        # RRF 融合
        fused: dict[int, float] = {}
        for rank, i in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(lex):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        cand = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

        rows = self._rows(cand[: max(pool, k)])

        do_rerank = rerank if rerank is not None else (RERANK_DEFAULT and self.reranker is not None)
        rer_scores: dict[int, float] = {}
        if do_rerank and self.reranker is not None and cand:
            top = [i for i in cand[: max(pool, k)] if i in rows]
            pairs = [(query, rows[i]["text"] or "") for i in top]
            try:
                sc = self.reranker.predict(pairs)
                rer_scores = {i: float(s) for i, s in zip(top, sc)}
                cand = sorted(top, key=lambda i: -rer_scores[i]) + [i for i in cand if i not in set(top)]
            except Exception:
                rer_scores = {}

        out = []
        for i in cand[:k]:
            r = rows.get(i)
            if r is None:
                continue
            text, pages = self._expand(r, window)
            citation = f"{r['book']}_{r['year']}" if r["year"] else (r["book"] or r["id"])
            # 可解释的相似度：重排分开 sigmoid；否则用稠密余弦；词法独有命中退回归一化融合分
            dc = None
            if self._dense_scores is not None and 0 <= i < len(self._dense_scores):
                dc = float(self._dense_scores[i])
                if dc == -np.inf:
                    dc = None
            if i in rer_scores:
                sim = round(1 / (1 + pow(2.718281828, -rer_scores[i])), 4)
            elif dc is not None:
                sim = round(dc, 4)
            else:
                mx = max(fused.values()) or 1.0
                sim = round(fused.get(i, 0.0) / mx, 4)
            out.append({
                "citation": citation,
                "book": r["book"], "year": r["year"],
                "page": pages[0] if pages else r["page"],
                "pages": pages,
                "file": r["file"], "sentence": r["id"],
                "text": text,
                "similarity": sim,
                "scores": {
                    "dense_rank": (dense.index(i) + 1) if i in dense else None,
                    "lexical_rank": (lex.index(i) + 1) if i in lex else None,
                    "fused": round(fused.get(i, 0.0), 6),
                    "dense_cos": round(dc, 4) if dc is not None else None,
                    "rerank": round(rer_scores[i], 4) if i in rer_scores else None,
                },
            })
        return out

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"<Retriever {self.count} 条 dim={self.dim} "
                f"reranker={'yes' if self.reranker else 'no'} dir={self.dir}>")
