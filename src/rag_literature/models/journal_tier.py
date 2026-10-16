"""EasyScholar JCR 分区查询。

查询期刊的 JCR 分区（Q1/Q2/Q3/Q4），本地缓存避免频繁调用 API。
免费版限速 5 分钟/次，全部走 EasyScholar API：/open/getPublicationRank
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from ..config import (
    EASYSCHOLAR_API,
    EASYSCHOLAR_KEY,
    JCR_API_CALL_INTERVAL,
    JCR_CACHE_PATH,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# 上一次 API 调用时间戳（模块级，多进程注意非线程安全）
_last_api_call = 0.0
# 内存缓存
_cache: dict[str, str] = {}
_cache_loaded = False


def _load_cache() -> dict[str, str]:
    """从磁盘加载 JCR 缓存。"""
    global _cache_loaded
    if _cache_loaded:
        return _cache
    if JCR_CACHE_PATH and JCR_CACHE_PATH.exists():
        try:
            with open(JCR_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache.update(json.load(f))
            logger.info("Loaded JCR cache (%d entries)", len(_cache))
        except Exception as e:
            logger.warning("Failed to load JCR cache: %s", e)
    _cache_loaded = True
    return _cache


def _save_cache():
    """将缓存写入磁盘。"""
    if not JCR_CACHE_PATH:
        return
    try:
        JCR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JCR_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
        logger.debug("Saved JCR cache (%d entries)", len(_cache))
    except Exception as e:
        logger.warning("Failed to save JCR cache: %s", e)


def _rate_limit():
    """确保 API 调用间隔 >= JCR_API_CALL_INTERVAL 秒。"""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < JCR_API_CALL_INTERVAL:
        sleep_time = JCR_API_CALL_INTERVAL - elapsed
        logger.debug("JCR rate limit: sleeping %.1fs", sleep_time)
        time.sleep(sleep_time)
    _last_api_call = time.time()


def lookup_jcr_zone(journal_name: str) -> Optional[str]:
    """查询期刊的 JCR 分区。

    Args:
        journal_name: 期刊名称（如 "Physical Review B"）。

    Returns:
        "Q1"/"Q2"/"Q3"/"Q4" 之一，若查询失败或未收录则返回 None。
    """
    if not journal_name or not journal_name.strip():
        return None

    name = journal_name.strip()

    # 加载并查询缓存
    cache = _load_cache()
    if name in cache:
        cached = cache[name]
        if cached:
            logger.info("JCR cache hit: %s → %s", name, cached)
            return cached
        else:
            logger.debug("JCR cache hit (not found): %s", name)
            return None

    # API 调用（带重试）
    _rate_limit()
    for attempt in range(MAX_RETRIES + 1):
        try:
            params = {
                "secretKey": EASYSCHOLAR_KEY,
                "publicationName": name,
            }
            resp = requests.get(EASYSCHOLAR_API, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 200 and data.get("data"):
                    official = data["data"].get("officialRank", {})
                    all_ranks = official.get("all", {})
                    sci_str = all_ranks.get("sci", "")
                    if sci_str and sci_str.startswith("Q"):
                        zone = sci_str.strip()
                        logger.info("JCR lookup: %s → %s", name, zone)
                        cache[name] = zone
                        _save_cache()
                        return zone
                    else:
                        logger.debug(
                            "JCR: %s returned no 'sci' field (got: %s), treating as not found",
                            name, sci_str,
                        )
                        # Cache as "not found" to avoid repeated lookups
                        cache[name] = ""
                        _save_cache()
                        return None
                else:
                    logger.warning(
                        "JCR API error (attempt %d/%d): code=%s msg=%s",
                        attempt + 1, MAX_RETRIES + 1,
                        data.get("code"), data.get("msg"),
                    )
            elif resp.status_code == 429:
                wait = (attempt + 1) * 10
                logger.info("JCR API rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            else:
                logger.warning(
                    "JCR API HTTP %d (attempt %d/%d)", resp.status_code,
                    attempt + 1, MAX_RETRIES + 1,
                )
        except requests.exceptions.Timeout:
            logger.warning(
                "JCR API timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES + 1,
            )
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue
        except Exception as e:
            logger.warning(
                "JCR API error (attempt %d/%d): %s",
                attempt + 1, MAX_RETRIES + 1, e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(5)
                continue

        # Non-retryable failure: break out
        break

    # Cache as "not found" after all retries failed
    cache[name] = ""
    _save_cache()
    return None
