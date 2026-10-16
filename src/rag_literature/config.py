"""Central configuration — all paths come from environment variables.

Defaults are intentionally *generic* (everything under ``$HOME``) so the pipeline
runs on any machine without editing code.  Point the environment variables at your
own data / model directories, or copy ``.env.example`` and adjust it.
"""

import os
from pathlib import Path

# ── Data directories ──────────────────────────────────────────────────
# One root holds everything the pipeline produces (PDFs, JSON, vector DB, state).
DATA_ROOT = Path(os.getenv("RAG_DATA_ROOT", str(Path.home() / "rag-data")))
INPUT_DIR = DATA_ROOT / ".input"
TMP_DIR = DATA_ROOT / ".tmp"
RAG_JSON_DIR = DATA_ROOT / ".RAG_Json"
DONE_PDF_DIR = DATA_ROOT / "DonePDF"
IMAGES_DIR = DATA_ROOT / "images"
CNREAD_DIR = DATA_ROOT / "cnRead"
TRASH_DIR = DATA_ROOT / "._trash"
SEARCH_HISTORY_DIR = DATA_ROOT / ".search_history"
VECTOR_DB_DIR = DATA_ROOT / ".vector_db"

# ── Model paths ───────────────────────────────────────────────────────
LLAMA_SERVER_BIN = Path(os.getenv("RAG_LLAMA_SERVER_BIN", str(Path.home() / "llama_standalone" / "llama-server")))
LLAMA_STANDALONE = LLAMA_SERVER_BIN.parent
LLAMA_STANDALONE_LIB = LLAMA_STANDALONE / "lib"
MODELS_DIR = Path(os.getenv("RAG_MODELS_DIR", str(Path.home() / "rag-models")))

# MinerU / ModelScope model cache.
# Why this variable matters: MinerU resolves the cache through the ``MODELSCOPE_CACHE``
# environment variable, and it runs as a *subprocess* (``conda run`` / ``bash -c``), i.e.
# as a non-interactive, non-login shell — so anything exported only from ``~/.bashrc``
# is invisible to it.  If the variable is missing, ModelScope silently falls back to
# ``~/.cache/modelscope`` and re-downloads several GB.  Hence a permanent default here,
# which ``pipeline/step2_mineru.py`` injects into the MinerU subprocess explicitly.
MINERU_MODELSCOPE_CACHE = Path(os.getenv("RAG_MODELSCOPE_CACHE", str(MODELS_DIR / "modelscope")))

MODEL_TRANSLATE_7B = os.getenv("RAG_MODEL_TRANSLATE_7B", str(MODELS_DIR / "HY-MT1.5-7B-Q4_K_M.gguf"))
MODEL_TRANSLATE_1_5B = os.getenv("RAG_MODEL_TRANSLATE_1_5B", str(MODELS_DIR / "HY-MT1.5-1.8B_bf16_Q4_K_M.gguf"))
MODEL_GENERATION = os.getenv("RAG_MODEL_GENERATION", str(MODELS_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"))

# ── Translation server (llama.cpp HTTP API) ───────────────────────────
SERVER_HOST = os.getenv("RAG_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("RAG_SERVER_PORT", "8081"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_START_TIMEOUT = int(os.getenv("RAG_SERVER_START_TIMEOUT", "30"))
IDLE_SLEEP_SECONDS = int(os.getenv("RAG_IDLE_SLEEP_SECONDS", "600"))
SERVER_TIMEOUT = int(os.getenv("RAG_SERVER_TIMEOUT", "300"))

# ── Translation parameters ────────────────────────────────────────────
TRANSLATE_N_PREDICT = int(os.getenv("RAG_TRANSLATE_N_PREDICT", "120"))
TRANSLATE_CTX = int(os.getenv("RAG_TRANSLATE_CTX", "4096"))
TRANSLATE_BATCH_SIZE = int(os.getenv("RAG_TRANSLATE_BATCH_SIZE", "20"))
TRANSLATE_N_GPU_LAYERS = int(os.getenv("RAG_TRANSLATE_N_GPU_LAYERS", "999"))

# ── Translation prompts ───────────────────────────────────────────────
TRANSLATION_PROMPT_EN2ZH = os.getenv("RAG_PROMPT_EN2ZH",
    "<|startoftext|>Translate the following English to Chinese. "
    "Output ONLY the Chinese translation, nothing else.\n\n"
    "English: {text}\n\nChinese:"
)
TRANSLATION_PROMPT_ZH2EN = os.getenv("RAG_PROMPT_ZH2EN",
    "<|startoftext|>Translate the following Chinese to English. "
    "Output ONLY the English translation, nothing else.\n\n"
    "Chinese: {text}\n\nEnglish:"
)

# ── MinerU ────────────────────────────────────────────────────────────
MINERU_CONDA_ENV = os.getenv("RAG_MINERU_CONDA_ENV", "mineru")
MINERU_CMD = os.getenv("RAG_MINERU_CMD", "mineru")
# Keep MinerU on local weights: "modelscope" reads/writes the ModelScope cache,
# "local" uses the explicit models-dir from ~/mineru.json and never downloads.
MINERU_MODEL_SOURCE = os.getenv("RAG_MINERU_MODEL_SOURCE", "modelscope")
# Parsing mode: txt = use the PDF text layer (fast); ocr = force OCR (slow, but rescues
# PDFs whose text layer / ToUnicode map is broken — some publisher PDFs extract as garbage).
MINERU_METHOD = os.getenv("RAG_MINERU_METHOD", "txt")
# Per-PDF timeout in seconds. Raise it for books (a 700-page book easily exceeds 20 min).
MINERU_TIMEOUT = int(os.getenv("RAG_MINERU_TIMEOUT", "1200"))


# ── Step switches ─────────────────────────────────────────────────────
# Translation and cnRead generation are NOT part of the pipeline: step 6 embeds English
# text only, so translations add cost without improving retrieval, and nothing consumed
# the generated Markdown.  Both remain available as standalone tools.
def _flag(env: str, default: str = "0") -> bool:
    return os.getenv(env, default).strip().lower() in ("1", "true", "yes", "on")


JCR_ENABLED = _flag("RAG_JCR")

# MinerU's --text-lang hint. "ch" covers both Chinese and English documents.
MINERU_LANG = os.getenv("RAG_MINERU_LANG", "ch")

# Read-only retrieval cache location.
# It must follow RAG_DATA_ROOT: when it was hard-coded to ~/.cache/... a pipeline run
# against an isolated test root silently overwrote the real cache.  RAG_INDEX_DIR wins.
_index_env = os.getenv("RAG_INDEX_DIR")
RETRIEVAL_INDEX_DIR = (
    Path(_index_env) if _index_env
    else (DATA_ROOT / ".retrieval_index" if os.getenv("RAG_DATA_ROOT")
          else Path.home() / ".cache" / "rag" / "index")
)

# Whether MinerU-extracted images are moved into images/ and kept.
# Default OFF: nothing in the pipeline consumes them, so they stay in .tmp and are
# removed together with it.  Set RAG_KEEP_IMAGES=1 if you want them kept.
KEEP_IMAGES = _flag("RAG_KEEP_IMAGES")

# ── RAG / ChromaDB ────────────────────────────────────────────────────
# Vector DB and all pipeline state live inside the data root:
# DATA_ROOT/.vector_db/{chroma_data,rag_state.json,pipeline_state.json,dedup_hashes.json}
RUN_DB_DIR = Path(os.getenv("RAG_RUN_DB_DIR", str(DATA_ROOT / ".vector_db")))
CHROMA_DATA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(RUN_DB_DIR / "chroma_data")))
RAG_STATE_FILE = Path(os.getenv("RAG_STATE_FILE", str(RUN_DB_DIR / "rag_state.json")))
CHROMA_COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION", "literature_sentences")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", str(MODELS_DIR / "bge-m3"))

# ── APIs ──────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API = os.getenv("RAG_S2_API", "https://api.semanticscholar.org/graph/v1")
# Optional: a free API key raises Semantic Scholar rate limits. Empty = anonymous.
SEMANTIC_SCHOLAR_KEY = os.getenv("RAG_S2_KEY", "")
# Optional JCR quartile lookup (opt-in via RAG_JCR=1). No key is shipped: register your
# own at easyscholar.cc and export RAG_EASYSCHOLAR_KEY. Without a key the lookup is skipped.
EASYSCHOLAR_API = os.getenv("RAG_EASYSCHOLAR_API", "https://www.easyscholar.cc/open/getPublicationRank")
EASYSCHOLAR_KEY = os.getenv("RAG_EASYSCHOLAR_KEY", "")
JCR_CACHE_PATH = DATA_ROOT / "_jcr_cache.json"
JCR_API_CALL_INTERVAL = float(os.getenv("RAG_JCR_INTERVAL", "300"))

# ── Retry / limits ────────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))
API_CALL_INTERVAL = float(os.getenv("RAG_API_CALL_INTERVAL", "4.0"))

# ── Legacy aliases ────────────────────────────────────────────────────
LLAMA_SERVER = str(LLAMA_SERVER_BIN)
LLAMA_CLI = str(LLAMA_STANDALONE / "llama-cli")
LLAMA_CLI_ARGS_COMMON = os.getenv("RAG_LLAMA_CLI_ARGS", "")
LLAMA_GENERATION_SERVER_URL = os.getenv("RAG_GENERATION_SERVER_URL", "")
LLAMA_GENERATION_PORT = os.getenv("RAG_GENERATION_PORT", "")
LLAMA_SERVER_URL = SERVER_URL
LLAMA_SERVER_PORT = SERVER_PORT
GENERATION_CTX = os.getenv("RAG_GENERATION_CTX", "")
GENERATION_N_PREDICT = os.getenv("RAG_GENERATION_N_PREDICT", "")
