"""CamelCase naming for literature files.

Rules (from PROJECT.md):
- Strip special characters, keep letters and digits
- Spaces → CamelCase (no underscores between words)
- Append _YYYY suffix
"""

import os
import re
import unicodedata


def sanitize_name(raw: str) -> str:
    """Convert a raw title to CamelCase with digits preserved.

    Example:
        'Principles of condensed matter physics' → 'PrinciplesOfCondensedMatterPhysics'
        'Acceleration and trapping of particles by radiation pressure' →
            'AccelerationAndTrappingOfParticlesByRadiationPressure'
    """
    normalized = unicodedata.normalize("NFC", raw)
    # Replace common separators with space
    normalized = re.sub(r"[-_/\\,;:]+", " ", normalized)
    # Keep only letters, digits, spaces
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    parts = normalized.split()
    if not parts:
        return "Unknown"

    # CamelCase: capitalize each word
    camel = "".join(part.capitalize() for part in parts)
    # Ensure first char is uppercase letter (not digit)
    if camel and camel[0].isdigit():
        camel = "Paper" + camel
    return camel


def extract_year(text: str) -> str:
    """Extract a 4-digit year from text. Returns the latest plausible year found."""
    # 不能用 \b：像 `ferre2013_universal_dw_dynamics` 这种文件名里年份**紧贴在字母后面**，
    # 字母与数字同属 word 字符 → 没有 \b 边界，会全部漏掉（2026-10-14 实测：
    # ferre2013 / morley2019 / novoselov2003 三篇因此落到 UNKN，而年份明明在文件名里）。
    # 改用「前后都不是数字」的环视，既不漏也不误吞长数字串（如 20241013）。
    years = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
    if not years:
        return ""
    # Return the latest year in plausible range
    valid = [y for y in years if 1900 <= int(y) <= 2030]
    return max(valid) if valid else years[-1]


# 文件名长度上限：NFS/ext4 单个文件名上限 255 字节，而库内还有
# "_v2.json" / "_raw.json" 等后缀。2026-10-13：Yoo 2002 用 OCR 模式后
# 标题抽取抓到整段正文（400+ 字符），生成的文件名触发 [Errno 36] File name too long，
# 整个 step3 失败。这里做硬截断兜底 —— 真正的标题修正走 err.md 的 UNKN 修正流程。
MAX_BASE_NAME_LEN = int(os.getenv("RAG_MAX_BASE_NAME_LEN", "180"))


def build_filename(raw_title: str, year: str = "") -> str:
    """Build CamelCase_YYYY base name (no extension).

    Truncated to MAX_BASE_NAME_LEN so a bad title extraction cannot produce a
    filename longer than the filesystem allows.
    """
    camel = sanitize_name(raw_title)[:MAX_BASE_NAME_LEN]
    if year:
        return f"{camel}_{year}"
    return camel
