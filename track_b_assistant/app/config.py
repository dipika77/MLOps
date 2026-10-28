"""
All the settings for the app are kept in this one file.
Values are read from environment variables (or the .env file) so we
don't have to hardcode API keys or change code when deploying.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# project root folder (ai_assistant/)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------- LLM settings ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# main model and a smaller/faster model used as fallback
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")

# local open-source model served with vLLM (last fallback, optional)
# example: http://localhost:8001/v1  or  http://vllm:8000/v1 inside docker compose
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# generation parameters
# low temperature = more factual answers, which is what we want for a RAG assistant
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
# gpt-oss is a reasoning model: its hidden reasoning tokens also count in max_tokens,
# so 800 was sometimes not enough and the final answer came back empty
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
# only sent to gpt-oss models (low / medium / high). "low" is enough for document Q&A and is faster
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))

# ---------------- W16: agentic loop settings ----------------
# hard cap on Researcher<->Verifier iterations for the cross-source verification loop.
# this is the "must not run indefinitely" stopping condition required by the assignment.
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "4"))
# used by the failure-injection test: when true, the document search tool inside the
# agentic loop always raises instead of returning results (see app/agentic.py)
SIMULATE_SEARCH_FAILURE = os.getenv("SIMULATE_SEARCH_FAILURE", "false").lower() == "true"

# ---------------- RAG settings ----------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters
TOP_K = int(os.getenv("TOP_K", "4"))

# ONNX version of the embedding model (see scripts/export_onnx.py)
USE_ONNX_EMBEDDINGS = os.getenv("USE_ONNX_EMBEDDINGS", "false").lower() == "true"
ONNX_MODEL_DIR = BASE_DIR / os.getenv("ONNX_MODEL_DIR", "models/minilm-onnx")
ONNX_MODEL_FILE = os.getenv("ONNX_MODEL_FILE", "model_quantized.onnx")

# ---------------- API / performance settings ----------------
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "200"))
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "5"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "10"))

# file where the create_support_ticket tool saves tickets
TICKETS_FILE = BASE_DIR / os.getenv("TICKETS_FILE", "tickets.json")

# used by the Streamlit UI
API_URL = os.getenv("API_URL", "http://localhost:8000")
