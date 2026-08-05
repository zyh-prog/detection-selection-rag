import os


class Config:
    XINFERENCE_BASE_URL = os.environ.get("XINFERENCE_BASE_URL", "http://127.0.0.1:9997")

    LLM_BACKEND       = "xinference"                       # "xinference" | "deepseek"
    LLM_MODEL         = "qwen2.5-instruct"
    # LLM_BACKEND       = "deepseek"                       # "xinference" | "deepseek"
    # LLM_MODEL         = "deepseek-ai/DeepSeek-V3.2"
    # DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
    # DEEPSEEK_API_KEY  = os.environ["DEEPSEEK_API_KEY"]

    LLM_CHAT_EXTRA_BODY = {"enable_thinking": False}

    LLM_SYSTEM_PROMPT = (
        "You are a helpful assistant. Follow the user's format constraints; "
        "answer concisely in English when asked for factual QA."
    )

    EMBEDDING_MODEL = "bge-large-en-v1.5"
    RERANK_MODEL    = "bge-reranker-v2-m3"

    TEMPERATURE       = 0.0
    MAX_TOKENS        = 64
    LLM_TIMEOUT       = 90
    EMBEDDING_TIMEOUT = 120
    RERANK_TIMEOUT    = 60

    AMBIGQA_DATASET_NAME = "ambig_qa"
    AMBIGQA_CONFIG       = "full"
    AMBIGQA_CACHE_DIR    = "./data"

    DATA_DIR            = "data"
    WIKIPEDIA_FILE      = "data/psgs_w100.tsv"
    FAISS_INDEX_PATH    = f"{DATA_DIR}/faiss_index_ivf.bin"
    METADATA_PATH       = f"{DATA_DIR}/metadata.pkl"
    FILTERED_WIKI_PATH  = f"{DATA_DIR}/wiki_filtered.tsv"
    ANNOTATED_DATA_PATH = f"{DATA_DIR}/annotated_validation.json"

    FAISS_GPU_DEVICE = 1
    FAISS_NPROBE     = 32

    TOP_K_RETRIEVE = 8
    TOP_K_RERANK   = 16

    BASE_THRESHOLD            = 0.42
    INTENT_COVERAGE_THRESHOLD = 0.80
    SEMANTIC_DEDUP_THRESHOLD  = 0.90
    THRESHOLD_ALPHA           = 0.10
    THRESHOLD_BETA            = 0.05
    DIVERSITY_ALPHA           = 0.15

    N_REVERSE_INTENTS   = 4
    MAX_ANSWERS         = 4
    ANSWER_COLLAPSE_TAU = 0.80
    ANSWER_MIN_SUPPORT  = 1

    EVAL_SAMPLE_SIZE = 500
    HUMAN_EVAL_SIZE  = 50
    RANDOM_SEED      = 42


# --- reversible env overrides (so both model arms can run without editing this file) ---
# RIV_LLM_BACKEND=deepseek RIV_LLM_MODEL=deepseek-ai/DeepSeek-V3.2 selects the API arm.
_bk = os.environ.get("RIV_LLM_BACKEND")
if _bk:
    Config.LLM_BACKEND = _bk
# vLLM's OpenAI server rejects unknown body fields, and enable_thinking is an
# xinference-specific one. Let a run opt out without editing the class.
if os.environ.get("RIV_NO_EXTRA_BODY"):
    Config.LLM_CHAT_EXTRA_BODY = {}
_lm = os.environ.get("RIV_LLM_MODEL")
if _lm:
    Config.LLM_MODEL = _lm
if Config.LLM_BACKEND == "deepseek":
    Config.DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1")
    Config.DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

config = Config()