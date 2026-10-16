"""语言判断 —— CJK 字符占比。

2026-10-13 重构：pipeline 不再有独立的「语言检测步骤」。翻译线已移除，语言标志只剩
两个用途：(a) step3 的中文标题抽取，(b) step4 的分句器选择与 `meta.is_chinese`。
两者都作用在**已经抽出来的文本**上 —— 所以这里只在已抽文本上算一次占比，
不额外读 PDF、不占一个流程步骤。

MinerU 的语言提示也不再依赖它：改用固定默认 `config.MINERU_LANG`（默认 `ch`，
MinerU 的 ch 模型同时覆盖中英文），可用 `RAG_MINERU_LANG` 覆盖。
"""

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def cjk_ratio(text: str) -> float:
    """非空白字符中 CJK 表意文字的比例。"""
    if not text:
        return 0.0
    body = re.sub(r"\s+", "", text)
    if not body:
        return 0.0
    return len(_CJK.findall(body)) / len(body)


def looks_chinese(text: str, threshold: float = 0.10) -> bool:
    """文本是否以中文为主。

    阈值 0.10 而非 0.30：英文论文几乎不会出现 CJK 字符，出现少量即足以判定；
    库内 4 篇中文原著都是整篇中文。
    """
    return cjk_ratio(text) >= threshold
