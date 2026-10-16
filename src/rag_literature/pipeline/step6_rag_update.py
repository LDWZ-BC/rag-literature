"""Step 6: Incremental RAG vector database update.

- Reads rag_state.json to track processed files
- For new/modified JSONs, embeds sentences and upserts to Chroma
- Each sentence gets a unique ID and metadata (book, page, year)
"""

import json
import logging
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import (
    RAG_JSON_DIR,
    CHROMA_DATA_DIR,
    RAG_STATE_FILE,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_RETRIES,
)
from ..utils.files import read_json, write_json

logger = logging.getLogger(__name__)

# Embedding function will be set lazily
_embedding_fn = None


def _get_embedding_fn():
    """Lazily load sentence-transformers embedding model."""
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils import embedding_functions
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        # Load the embedding model from EMBEDDING_MODEL (defaults to a local
        # offline path under your models dir; may be a HF/ModelScope id).
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device=device,
        )
        logger.info("Embedding model on %s (CUDA=%s)", device, torch.cuda.is_available())
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _embedding_fn


def _get_chroma_client() -> chromadb.PersistentClient:
    """Get or create Chroma persistent client."""
    CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DATA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_collection(
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """Get or create the literature collection."""
    ef = _get_embedding_fn()
    try:
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        collection = client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"description": "Literature sentences for RAG"},
        )
    return collection


def _compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file for change detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_rag_state() -> dict:
    """Load rag_state.json or return empty dict."""
    if RAG_STATE_FILE.exists():
        return read_json(RAG_STATE_FILE)
    return {"files": {}, "last_updated": ""}


def save_rag_state(state: dict):
    """Save rag_state.json."""
    state["last_updated"] = datetime.now().isoformat()
    write_json(RAG_STATE_FILE, state)


# Chroma 的 PersistentLocalHnswSegment 只在
#   _num_log_records_since_last_persist >= sync_threshold（默认 1000）
# 时才调用 _persist()。因此"最后一次落盘之后"写入的记录永远不会进
# index_metadata.pickle —— 最后一个文件的尾部会静默地不在向量索引里
# （实测缺口 733 / 488 条，都 < 1000，与阈值行为完全吻合）。
# 对策：更新结束后再改写一批 >= 阈值的**已存在**记录，把计数器推过阈值，
# 触发一次落盘 —— 此时内存里的 id_to_label 已包含全部记录，尾部一并写下去。
SYNC_THRESHOLD = 1000


def _force_persist(collection: chromadb.Collection, batch: int = 1500) -> int:
    """Touch existing records to force a final HNSW index persist.

    Returns the number of records touched (0 if the collection is empty).
    """
    got = collection.get(limit=batch, include=["documents", "metadatas"])
    ids, docs, metas = got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
    if not ids:
        return 0
    # upsert（而非 update）以确保产生日志记录 —— 计数器的增长靠的就是日志记录数
    for i in range(0, len(ids), 500):
        collection.upsert(ids=ids[i:i+500], documents=docs[i:i+500], metadatas=metas[i:i+500])
    return len(ids)


def incremental_update(force: bool = False) -> dict:
    """Perform incremental RAG update.

    Checks all JSONs in RAG_Json/ against rag_state.json.
    Only processes new or modified files.

    Args:
        force: If True, reprocess all files regardless of state

    Returns:
        dict with 'added', 'updated', 'removed', 'errors' counts
    """
    state = load_rag_state()
    client = _get_chroma_client()
    collection = _get_collection(client)

    json_files = sorted(RAG_JSON_DIR.glob("*_v2.json"))
    if not json_files:
        logger.info("No JSON files found in %s", RAG_JSON_DIR)
        return {"added": 0, "updated": 0, "removed": 0, "errors": 0}

    stats = {"added": 0, "updated": 0, "removed": 0, "errors": 0}
    processed_files = set()

    for json_path in json_files:
        file_key = json_path.name
        file_hash = _compute_file_hash(json_path)
        processed_files.add(file_key)

        if not force and file_key in state.get("files", {}):
            if state["files"][file_key].get("hash") == file_hash:
                logger.debug("Skipping unchanged: %s", file_key)
                continue
            stats["updated"] += 1
        else:
            stats["added"] += 1

        try:
            _upsert_file(json_path, collection, file_key, file_hash, state)

            state.setdefault("files", {})[file_key] = {
                "hash": file_hash,
                "processed": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Failed to process %s: %s", file_key, e)
            stats["errors"] += 1

    # Remove files no longer present
    removed_keys = set(state.get("files", {}).keys()) - processed_files
    for key in removed_keys:
        try:
            collection.delete(where={"file": key})
            del state["files"][key]
            stats["removed"] += 1
            logger.info("Removed vectors for deleted file: %s", key)
        except Exception as e:
            logger.warning("Failed to remove vectors for %s: %s", key, e)

    # 强制一次收尾落盘，避免最后一个文件的尾部不进索引
    if stats["added"] or stats["updated"]:
        try:
            n = _force_persist(collection)
            logger.info("Forced HNSW persist flush over %d records", n)
        except Exception as e:
            logger.warning("Final persist flush failed: %s", e)

    save_rag_state(state)
    logger.info(
        "RAG update complete: +%d ~%d -%d (errors: %d)",
        stats["added"], stats["updated"], stats["removed"], stats["errors"],
    )
    return stats


def _upsert_file(
    json_path: Path,
    collection: chromadb.Collection,
    file_key: str,
    file_hash: str,
    state: dict,
):
    """Upsert sentences from a single structured JSON file into Chroma.

    Handles both old flat format and new structured format.
    """
    data = read_json(json_path)

    # Detect format: structured has "meta" and "blocks" keys
    if isinstance(data, dict) and "blocks" in data:
        meta = data.get("meta", {})
        blocks = data["blocks"]
        is_chinese = meta.get("is_chinese", False)

        # Delete existing entries for this file first
        try:
            collection.delete(where={"file": file_key})
        except Exception:
            pass

        ids = []
        documents = []
        metadatas = []

        sent_idx = 0
        for block in blocks:
            if block.get("type") != "text":
                continue
            for sent in block.get("sentences", []):
                en_text = sent.get("en", "")
                zh_text = sent.get("zh", "")

                if not en_text.strip():
                    continue

                sent_id = f"{file_key}_{sent_idx}"
                ids.append(sent_id)
                # For Chinese docs, index the Chinese text; for EN docs, index EN
                documents.append(en_text)
                metadatas.append({
                    "file": file_key,
                    "book": meta.get("book", ""),
                    "year": meta.get("year", ""),
                    "page": block.get("page", 0),
                    "sentence_zh": zh_text,
                    "is_chinese": is_chinese,
                    "jcr_zone": meta.get("jcr_zone", ""),
                })
                sent_idx += 1

    else:
        # Old flat format: list of sentence dicts
        sentences = data if isinstance(data, list) else []
        if not sentences:
            logger.warning("No sentences in %s", file_key)
            return

        try:
            collection.delete(where={"file": file_key})
        except Exception:
            pass

        ids = []
        documents = []
        metadatas = []

        for sent in sentences:
            sent_id = f"{file_key}_{sent['id']}"
            ids.append(sent_id)
            documents.append(sent["sentence_en"])
            metadatas.append({
                "file": file_key,
                "book": sent.get("book", ""),
                "year": sent.get("year", ""),
                "page": sent.get("page", 0),
                "sentence_zh": sent.get("sentence_zh", ""),
                "block_type": sent.get("block_type", ""),
                "jcr_zone": "",
            })

    if not ids:
        logger.warning("No sentences to upsert from %s", file_key)
        return

    # Batch insert (Chroma handles embedding)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.upsert(
            ids=ids[batch_slice],
            documents=documents[batch_slice],
            metadatas=metadatas[batch_slice],
        )

    logger.info(
        "Upserted %d sentences from %s", len(ids), file_key
    )
