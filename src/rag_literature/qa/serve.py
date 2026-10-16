#!/usr/bin/env python3
"""常驻检索服务（本机 127.0.0.1）—— 给 preset / 问答 / 写作审查用。

## 为什么需要
逐次调用 `query_rag.py` 的主要开销**不是检索**（0.3–0.7 秒），而是**模型加载**：
bge-m3 约 6.6 秒 + reranker 约 1.4 秒。服务把模型与向量缓存常驻，单次查询降到 **<1 秒**。

服务在本机跑（不受会话沙箱限制），所以能用 GPU；客户端只发 HTTP，
**既不需要提权、也不需要等模型加载**。

## 启动
```bash
python -m rag_literature.qa.serve --port 8077          # 前台
nohup python -m rag_literature.qa.serve >/tmp/rag_serve.log 2>&1 &   # 后台
```

## 接口
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{"ok":true,"count":N,"reranker":bool,"device":"cuda"}` |
| GET | `/search` | 参数 `q`（必填）、`k`、`pool`、`project`、`hybrid`(0/1)、`rerank`(0/1)、`window` |
| POST | `/search` | body 为 JSON，字段同上（`q` 必填） |

返回与 `query_rag.py` 的 JSON 相同：`{"results":[{citation,text,page,scores,...}],"count":N,...}`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_literature.qa.retriever import Retriever  # noqa: E402

DEFAULT_PORT = 8077
PROJECT_SCOPE_PATH = Path(os.environ.get(
    "RAG_PROJECT_SCOPE", str(Path.home() / "Code/RAG/project_scope.json")))

_RT: Retriever | None = None
_RT_MTIME: float | None = None
_RT_LOCK = threading.Lock()


def _cache_mtime():
    try:
        return (_RT.dir / "meta.json").stat().st_mtime
    except Exception:
        return None


def _maybe_reload() -> None:
    """导出完成后（pipeline step6c）自动换成新索引。

    `meta.json` 是 export_index 的**最后一步**才写的 → 拿它当"新索引已就绪"的标记，
    不会读到写了一半的 npy/sqlite。未就绪期间本进程继续用旧 inode 服务旧数据，
    结果自洽（只是暂时看不到新文献）。
    """
    global _RT, _RT_MTIME
    m = _cache_mtime()
    if m is None or m == _RT_MTIME:
        return
    with _RT_LOCK:
        m = _cache_mtime()
        if m is None or m == _RT_MTIME:
            return
        try:
            new = Retriever()
            new.model
            new.reranker
        except Exception as e:
            print(f"[reload] 跳过（{type(e).__name__}: {e}）", flush=True)
            return
        old, _RT, _RT_MTIME = _RT, new, m
        try:
            old.close()
        except Exception:
            pass
        print(f"[reload] 检索缓存已更新：{new.count} 条（exported_at={new.meta.get('exported_at')}）",
              flush=True)


def _books_for(project: str | None) -> list[str] | None:
    if not project:
        return None
    try:
        scope = json.loads(PROJECT_SCOPE_PATH.read_text(encoding="utf-8"))
        proj = (scope.get("projects") or {}).get(project)
        if not proj:
            return None
        return [re.sub(r"_\d{4}$", "", t) for t in (proj.get("corpus_titles") or []) if t]
    except Exception:
        return None


def _flag(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def do_search(p: dict) -> dict:
    try:
        return _do_search(p)
    except Exception as e:  # 失败要明着报，不能给空响应
        import traceback
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}", "results": [], "count": 0}


def _do_search(p: dict) -> dict:
    _maybe_reload()
    q = (p.get("q") or p.get("query") or "").strip()
    if not q:
        return {"error": "missing q", "results": [], "count": 0}
    t0 = time.time()
    res = _RT.search(
        q,
        k=int(p.get("k") or 5),
        pool=int(p.get("pool") or 50),
        project_scope=_books_for(p.get("project")),
        hybrid=_flag(p.get("hybrid"), True),
        rerank=(None if p.get("rerank") in (None, "") else _flag(p.get("rerank"))),
        window=int(p.get("window") if p.get("window") not in (None, "") else 1),
    )
    out = []
    for r in res:
        e = {"citation": r["citation"], "article": r["citation"], "book": r["book"],
             "year": r["year"], "text": r["text"], "document": r["text"],
             "page": r.get("page"), "pages": r.get("pages"),
             "sentence": r["sentence"], "file": r["file"],
             "similarity": r["similarity"], "scores": r["scores"]}
        out.append(e)
    return {"results": out, "count": len(out), "query": q,
            "ms": int(1000 * (time.time() - t0))}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # 安静点，只留错误
        if len(a) > 1 and str(a[1]).startswith(("4", "5")):
            sys.stderr.write("  [serve] %s\n" % (fmt % a))

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            self._send({"ok": True, "count": _RT.count, "dim": _RT.dim,
                        "device": _RT._device,
                        "reranker": _RT.reranker is not None,
                        "exported_at": _RT.meta.get("exported_at")})
            return
        if u.path in ("/search", "/"):
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            self._send(do_search(q))
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send({"error": "bad json"}, 400)
            return
        if urlparse(self.path).path in ("/search", "/"):
            self._send(do_search(body))
        else:
            self._send({"error": "not found"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser(description="常驻检索服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()

    global _RT, _RT_MTIME
    t0 = time.time()
    _RT = Retriever()
    _RT_MTIME = _cache_mtime()
    _RT.model            # 预热 bge-m3
    _RT.reranker         # 预热 cross-encoder（没有模型则为 None）
    print(f"检索缓存 {_RT.dir} | {_RT.count} 条 | dim={_RT.dim} | device={_RT._device} "
          f"| reranker={'有' if _RT.reranker else '无'} | 预热 {time.time()-t0:.1f}s", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"rag-serve 就绪： http://{a.host}:{a.port}/search?q=...&k=5", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _RT.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
