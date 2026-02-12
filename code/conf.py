# code/conf.py
import os
import warnings
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, "env.properties")
load_dotenv(ENV_PATH)

# ------------------------------------------------------------
# Production hygiene: disable noisy telemetry (Chroma/others)
# ------------------------------------------------------------
# ChromaDB uses anonymized telemetry by default in some builds.
# The errors in your log ("capture() takes ...") come from telemetry stack mismatch.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

# Reduce deprecation warning noise in CLI (keep logs readable)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

Prefix = "/api/operation"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.1")

OPENROUTER_MODEL_ACADEMIC = os.getenv("OPENROUTER_MODEL_ACADEMIC", "openai/gpt-5.1")
OPENROUTER_MODEL_PRACTICAL = os.getenv("OPENROUTER_MODEL_PRACTICAL", "qwen/qwen3-next-80b-a3b-instruct")

OPENROUTER_SWITCH_MODEL = os.getenv("OPENROUTER_SWITCH_MODEL", "openai/gpt-5.1")

TEMPERATURE_ACADEMIC = float(os.getenv("TEMPERATURE_ACADEMIC", "0.3"))
TEMPERATURE_PRACTICAL = float(os.getenv("TEMPERATURE_PRACTICAL", "0.2"))

MAX_TOKENS_ACADEMIC = int(os.getenv("MAX_TOKENS_ACADEMIC", "8000"))
MAX_TOKENS_PRACTICAL = int(os.getenv("MAX_TOKENS_PRACTICAL", "650"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "7"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))

LLM_DOCS_MAX_PRACTICAL = int(os.getenv("LLM_DOCS_MAX_PRACTICAL", "8"))
LLM_DOCS_MAX_ACADEMIC = int(os.getenv("LLM_DOCS_MAX_ACADEMIC", "10"))

LLM_DOC_CHARS_PRACTICAL = int(os.getenv("LLM_DOC_CHARS_PRACTICAL", "250"))
LLM_DOC_CHARS_ACADEMIC = int(os.getenv("LLM_DOC_CHARS_ACADEMIC", "350"))

RETRIEVAL_QUERY_MAX_CHARS = int(os.getenv("RETRIEVAL_QUERY_MAX_CHARS", "200"))

DEBUG_LATENCY = os.getenv("DEBUG_LATENCY", "true").lower() == "true"

USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI", "./milvus_lite.db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thai_food_business")

LOCAL_VECTOR_DIR = os.getenv("LOCAL_VECTOR_DIR", "./local_chroma")

if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY not set. LLM calls will fail.")
