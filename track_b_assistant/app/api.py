"""
FastAPI backend.

Endpoints:
    GET  /health       - status of the service, vector DB and cache
    POST /chat         - ask one question
    POST /chat/batch   - ask many questions at once (processed concurrently)
    POST /ingest       - re-ingest the documents in the data folder
    POST /cache/clear  - clear the response cache

Run locally:
    uvicorn app.api:app --reload --port 8000
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import config
from app.agentic import agentic_answer_question
from app.assistant import answer_question
from app.cache import TTLCache
from app.llm_client import PROVIDERS
from app.rag import vector_store
from app.rag.embeddings import get_embedder
from app.rag.ingest import ingest_documents
from app.rate_limiter import RateLimiter
from app.schemas import (
    AgenticChatRequest,
    AgenticChatResponse,
    BatchRequest,
    BatchResponse,
    ChatRequest,
    ChatResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

cache = TTLCache(max_items=config.CACHE_MAX_ITEMS, ttl_seconds=config.CACHE_TTL_SECONDS)
rate_limiter = RateLimiter(max_requests=config.RATE_LIMIT_PER_MINUTE, window_seconds=60)


@asynccontextmanager
async def lifespan(app):
    # runs once when the server starts
    logger.info("Loading embedding model...")
    await asyncio.to_thread(get_embedder)
    try:
        if vector_store.count_chunks() == 0:
            logger.info("Vector store is empty, ingesting documents from %s", config.DATA_DIR)
            await asyncio.to_thread(ingest_documents)
    except Exception as e:
        # the API still starts, the assistant will just answer without document context
        logger.error("Startup ingestion failed: %s", e)
    logger.info("Providers configured: %s", [p["name"] for p in PROVIDERS])
    yield


app = FastAPI(title="Acme AI Assistant", version="2.0", lifespan=lifespan)


def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        seconds = rate_limiter.retry_after(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {config.RATE_LIMIT_PER_MINUTE} per minute. Try again in {seconds}s.",
            headers={"Retry-After": str(seconds)},
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # don't show python stack traces to users
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Please try again."})


async def process_question(question, history=None, temperature=None, top_p=None, use_cache=True):
    start = time.time()

    # we only cache questions without chat history, because with history the answer can change
    cache_key = None
    if use_cache and not history:
        cache_key = TTLCache.make_key(question, temperature, top_p)
        cached = cache.get(cache_key)
        if cached is not None:
            return ChatResponse(**cached, cached=True, latency_ms=int((time.time() - start) * 1000))

    result = await answer_question(question, history=history, temperature=temperature, top_p=top_p)

    # don't cache degraded answers, next time the LLM may be working again
    if cache_key and not result["degraded"]:
        cache.set(cache_key, result)

    return ChatResponse(**result, cached=False, latency_ms=int((time.time() - start) * 1000))


@app.get("/health")
async def health():
    try:
        chunks = vector_store.count_chunks()
        vector_db = "ok"
    except Exception as e:
        chunks = 0
        vector_db = f"error: {e}"

    return {
        "status": "ok" if PROVIDERS and vector_db == "ok" else "degraded",
        "llm_providers": [p["name"] for p in PROVIDERS],
        "vector_db": vector_db,
        "chunks": chunks,
        "embedding_backend": get_embedder().backend,
        "cache": cache.stats(),
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_rate_limit)])
async def chat(request: ChatRequest):
    history = [m.model_dump() for m in request.history]
    return await process_question(
        request.question,
        history=history,
        temperature=request.temperature,
        top_p=request.top_p,
        use_cache=request.use_cache,
    )


@app.post("/chat/batch", response_model=BatchResponse, dependencies=[Depends(check_rate_limit)])
async def chat_batch(request: BatchRequest):
    start = time.time()
    # all questions run concurrently. The semaphore in llm_client makes sure
    # we don't send more than MAX_CONCURRENT_LLM_CALLS requests to the provider at once.
    tasks = [process_question(q, use_cache=request.use_cache) for q in request.questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for question, result in zip(request.questions, results):
        if isinstance(result, Exception):
            logger.error("Batch question failed: %s -> %s", question, result)
            responses.append(ChatResponse(answer="Error while processing this question.", confidence="low", degraded=True))
        else:
            responses.append(result)

    return BatchResponse(results=responses, total_latency_ms=int((time.time() - start) * 1000))


@app.post("/agentic_chat", response_model=AgenticChatResponse, dependencies=[Depends(check_rate_limit)])
async def agentic_chat(request: AgenticChatRequest):
    """
    W16 Task 3: Researcher/Verifier cross-source verification loop.
    Not cached (each run can take a different number of iterations depending on what
    the agent finds), and always shows its trajectory so the loop is inspectable.
    """
    start = time.time()
    result = await agentic_answer_question(request.question, max_iterations=request.max_iterations)
    return AgenticChatResponse(**result, latency_ms=int((time.time() - start) * 1000))


@app.post("/ingest")
async def ingest(reset: bool = False):
    stats = await asyncio.to_thread(ingest_documents, reset)
    cache.clear()  # old answers may be outdated after new documents
    return stats


@app.post("/cache/clear")
async def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}
