# code/conf.py
import os
import warnings
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, "env.properties")
load_dotenv(ENV_PATH)

# ------------------------------------------------------------
# Production hygiene: disable noisy telemetry (Chroma/others)
# FIX: use lowercase "false" — Chroma reads lowercase, not "False"
# ------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

# Reduce deprecation warning noise in CLI (keep logs readable)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

Prefix = "/api/operation"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1")

OPENROUTER_MODEL_ACADEMIC = os.getenv("OPENROUTER_MODEL_ACADEMIC", "openai/gpt-4.1")
OPENROUTER_MODEL_PRACTICAL = os.getenv("OPENROUTER_MODEL_PRACTICAL", "qwen/qwen3-235b-a22b")

OPENROUTER_SWITCH_MODEL = os.getenv("OPENROUTER_SWITCH_MODEL", "openai/gpt-4.1")
# Separate fast model for topic_picker (non-critical, fail-fast friendly)
OPENROUTER_MODEL_TOPIC_PICKER = os.getenv("OPENROUTER_MODEL_TOPIC_PICKER", OPENROUTER_SWITCH_MODEL)

# FIX: wrap conversions in try/except so bad env vars give clear error instead of crashing silently
def _safe_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise RuntimeError(f"Config error: {name}='{raw}' is not a valid float")

def _safe_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise RuntimeError(f"Config error: {name}='{raw}' is not a valid integer")

TEMPERATURE_ACADEMIC = _safe_float("TEMPERATURE_ACADEMIC", 0.3)
TEMPERATURE_PRACTICAL = _safe_float("TEMPERATURE_PRACTICAL", 0.2)

MAX_TOKENS_ACADEMIC = _safe_int("MAX_TOKENS_ACADEMIC", 8000)
MAX_TOKENS_ACADEMIC_SLOTS = _safe_int("MAX_TOKENS_ACADEMIC_SLOTS", 3000)
MAX_TOKENS_PRACTICAL = _safe_int("MAX_TOKENS_PRACTICAL", 2000)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MAX_ROUNDS = _safe_int("MAX_ROUNDS", 7)
RETRIEVAL_TOP_K = _safe_int("RETRIEVAL_TOP_K", 20)

LLM_DOCS_MAX_PRACTICAL = _safe_int("LLM_DOCS_MAX_PRACTICAL", 8)
LLM_DOCS_MAX_ACADEMIC = _safe_int("LLM_DOCS_MAX_ACADEMIC", 10)

LLM_DOC_CHARS_PRACTICAL = _safe_int("LLM_DOC_CHARS_PRACTICAL", 250)
LLM_DOC_CHARS_ACADEMIC = _safe_int("LLM_DOC_CHARS_ACADEMIC", 350)

RETRIEVAL_QUERY_MAX_CHARS = _safe_int("RETRIEVAL_QUERY_MAX_CHARS", 200)

# Timeouts (seconds) for LLM and external requests
LLM_REQUEST_TIMEOUT = _safe_int("LLM_REQUEST_TIMEOUT", 60)
# Shorter timeout for topic_picker (non-critical, fast-fail to fallback)
LLM_TOPIC_PICKER_TIMEOUT = _safe_int("LLM_TOPIC_PICKER_TIMEOUT", 8)
SHEETS_REQUEST_TIMEOUT = _safe_int("SHEETS_REQUEST_TIMEOUT", 20)

DEBUG_LATENCY = os.getenv("DEBUG_LATENCY", "true").lower() == "true"

USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI", "./milvus_lite.db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thai_food_business")

LOCAL_VECTOR_DIR = os.getenv("LOCAL_VECTOR_DIR", "./local_chroma")

# NEW: centralized default retrieval fallback query — single source of truth
# All code should import this instead of hardcoding the Thai string
DEFAULT_RETRIEVAL_FALLBACK_QUERY = os.getenv(
    "DEFAULT_RETRIEVAL_FALLBACK_QUERY",
    "กฎหมายร้านอาหาร ใบอนุญาต ภาษี VAT จดทะเบียน สุขาภิบาล ประกันสังคม",
)

# FIX: hard stop on missing API key — fail at startup, not at first LLM call
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Please set it in env.properties before starting the server."
    )


def validate_config() -> None:
    """
    Fail-fast config validation.
    Call once at server startup to catch bad values early.
    """
    errors = []
    if not (0.0 <= TEMPERATURE_ACADEMIC <= 2.0):
        errors.append(f"TEMPERATURE_ACADEMIC={TEMPERATURE_ACADEMIC} must be in [0, 2]")
    if not (0.0 <= TEMPERATURE_PRACTICAL <= 2.0):
        errors.append(f"TEMPERATURE_PRACTICAL={TEMPERATURE_PRACTICAL} must be in [0, 2]")
    if MAX_TOKENS_ACADEMIC < 100:
        errors.append(f"MAX_TOKENS_ACADEMIC={MAX_TOKENS_ACADEMIC} is too low (min 100)")
    if MAX_TOKENS_PRACTICAL < 50:
        errors.append(f"MAX_TOKENS_PRACTICAL={MAX_TOKENS_PRACTICAL} is too low (min 50)")
    if RETRIEVAL_TOP_K < 1:
        errors.append(f"RETRIEVAL_TOP_K={RETRIEVAL_TOP_K} must be >= 1")
    if LLM_REQUEST_TIMEOUT < 5:
        errors.append(f"LLM_REQUEST_TIMEOUT={LLM_REQUEST_TIMEOUT} is very short (min 5s)")
    if errors:
        raise RuntimeError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
