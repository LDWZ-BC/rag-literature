#!/usr/bin/env python3
"""检索质量评测：合成 golden set + recall@k / MRR。

没有评测，任何"检索改好了"都是盲说。本脚本自动造三组测试，把
`稠密` / `稠密+词法(RRF)` / `稠密+词法+重排` 三种配置放在同一把尺子上量。

| 组 | query 来源 | 目标 | 测什么 |
|---|---|---|---|
| A passage→paper | 随机抽一条真实句子（排除它 ±3 句窗口） | 该句所属文献 | 语义能不能找到对的文章 |
| B term→paper | 稀有术语（只出现在 1–5 篇里） | 含该术语的文献 | 精确术语（BM25 的用武之地） |
| C zh→en | 手写中文问题（物理概念） | 期望的文献（子串匹配） | 中文提问检索英文文献（preset 的主用法） |

用法：
```bash
python -m rag_literature.tools.eval_retrieval            # 默认 A/B 各 200 条
python -m rag_literature.tools.eval_retrieval -n 300 --k 10
python -m rag_literature.tools.eval_retrieval --configs dense,hybrid,hybrid+rerank
```
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── C 组：手写中文问题 → 期望文献（子串匹配 book）────────────────────
# 请按**自己的语料**填写，例如：
#   ("你的中文问题", "期望命中的 book 名子串")
# 留空则跳过 C 组（只跑 A/B 两组自动构造的用例）。
ZH_CASES: list[tuple[str, str]] = []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _recall_mrr(hits_per_query: list[list[str]], targets: list, ks=(1, 5, 10)):
    """targets 每项可以是 str 或可接受文献集合（B 组一个术语常在多篇里出现，只认
    「按字母序第一个」会把正确答案判成 miss —— 必须接受集合）。"""
    rec = {k: 0 for k in ks}
    rr = []
    for hits, tgt in zip(hits_per_query, targets):
        acc = {tgt} if isinstance(tgt, str) else set(tgt)
        accn = {_norm(a) for a in acc if a}
        pos = next((i for i, h in enumerate(hits)
                    if any(a and (a in _norm(h) or _norm(h) in a) for a in accn)), None)
        if pos is None:
            rr.append(0.0)
            continue
        for k in ks:
            if pos < k:
                rec[k] += 1
        rr.append(1.0 / (pos + 1))
    n = max(len(targets), 1)
    return {f"recall@{k}": round(rec[k] / n, 3) for k in ks}, round(statistics.mean(rr), 3) if rr else 0.0


def build_cases_A(rt, n: int, seed: int = 17):
    rnd = random.Random(seed)
    rows = rt.db.execute(
        "select idx, id, book, text from rec where length(text) > 80 and book is not null and book != ''"
    ).fetchall()
    rnd.shuffle(rows)
    cases = []
    for r in rows:
        if len(cases) >= n:
            break
        m = re.match(r"^(?P<f>.+)_(?P<n>\d+)$", r["id"] or "")
        if not m:
            continue
        f, i = m.group("f"), int(m.group("n"))
        drop = {f"{f}_{j}" for j in range(i - 3, i + 4)}
        cases.append({"q": r["text"], "target": r["book"], "drop": drop})
    return cases


def build_cases_B(rt, n: int, seed: int = 23):
    """稀有术语：用 FTS 词表找出现在 1–5 篇里的词。"""
    rnd = random.Random(seed)
    try:
        terms = [r[0] for r in rt.db.execute(
            "select term from embedding_fulltext_search_idx limit 1")]  # 占位，若表不可直接用下面的路径
    except Exception:
        terms = []
    # 走 rec 文本自己统计（不依赖 FTS 内部表）
    import collections
    df = collections.defaultdict(set)
    for r in rt.db.execute("select book, text from rec where length(text) between 40 and 200 limit 60000"):
        for w in set(re.findall(r"[A-Za-z][A-Za-z\-]{6,}", r["text"] or "")):
            df[w].add(r["book"])
    cand = [w for w, b in df.items() if 1 <= len(b) <= 5 and len(w) >= 8]
    rnd.shuffle(cand)
    cases = []
    for w in cand[:n]:
        cases.append({"q": w, "target": set(df[w]), "drop": set()})  # 接受所有含该词的文献
    return cases


def run(cfg: str, rt, cases, k: int):
    use_hybrid = "hybrid" in cfg
    use_rerank = "rerank" in cfg
    hits, targets, lat = [], [], []
    for c in cases:
        t0 = time.time()
        res = rt.search(c["q"], k=k, pool=50, hybrid=use_hybrid, rerank=use_rerank, window=0)
        lat.append(time.time() - t0)
        out = [r["book"] for r in res if r["sentence"] not in c["drop"]]
        hits.append(out[:k])
        targets.append(c["target"])
    met, mrr = _recall_mrr(hits, targets, ks=(1, 5, min(10, k)))
    met["MRR"] = mrr
    met["ms/query"] = int(1000 * statistics.mean(lat)) if lat else 0
    return met


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="A/B 组各多少条")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--configs", default="dense,hybrid,hybrid+rerank")
    ap.add_argument("--json", type=Path, default=None, help="把结果写进 json")
    a = ap.parse_args()

    from rag_literature.qa.retriever import Retriever
    rt = Retriever()
    print(f"检索缓存：{rt}\n")
    A = build_cases_A(rt, a.n)
    B = build_cases_B(rt, a.n)
    C = [{"q": q, "target": t, "drop": set()} for q, t in ZH_CASES]
    print(f"用例：A passage→paper {len(A)} | B term→paper {len(B)} | C zh→en {len(C)}\n")

    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    table = {}
    if not C:
        print("（C 组 zh→en 用例为空 —— 请在 ZH_CASES 里按自己的语料填写）\n")
    for name, cases in (("A passage→paper", A), ("B term→paper", B), ("C zh→en", C)):
        print(f"— {name} —")
        print(f"  {'配置':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>8}{'ms/q':>7}")
        table[name] = {}
        for cfg in configs:
            m = run(cfg, rt, cases, a.k)
            table[name][cfg] = m
            print(f"  {cfg:<16}{m['recall@1']:>7.3f}{m['recall@5']:>7.3f}"
                  f"{m.get(f'recall@{min(10,a.k)}', m['recall@5']):>7.3f}{m['MRR']:>8.3f}{m['ms/query']:>7}")
        print()
    if a.json:
        a.json.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print("结果 →", a.json)
    rt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
