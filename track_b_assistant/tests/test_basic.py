"""
Unit tests for the parts that don't need the LLM.

    pytest -q
"""
import asyncio
import time

from app import assistant, llm_client
from app.assistant import parse_json_answer
from app.cache import TTLCache
from app.rag.ingest import chunk_text
from app.rate_limiter import RateLimiter
from app.tools import calculator, run_tool


# ---------- chunking ----------
def test_chunks_are_not_too_big():
    text = "\n\n".join(["This is paragraph number %d with some words in it." % i for i in range(100)])
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 300 + 60  # small margin because of the overlap


def test_long_paragraph_is_split():
    text = "word " * 1000  # one very long paragraph
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 5


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", chunk_size=500, overlap=100) == ["hello world"]


# ---------- tools ----------
def test_calculator():
    assert calculator("24 - 7") == "17"
    assert calculator("15 * 12 * 0.8") == "144.0"
    assert "division by zero" in calculator("1/0")


def test_calculator_blocks_code():
    assert calculator("__import__('os').system('ls')").startswith("Error")


def test_unknown_tool():
    assert "does not exist" in run_tool("delete_everything", {})


# ---------- JSON parsing ----------
def test_parse_json_with_markdown_fence():
    text = '```json\n{"answer": "24 days", "sources": ["hr.md"], "confidence": "high", "follow_up_questions": []}\n```'
    parsed = parse_json_answer(text)
    assert parsed.answer == "24 days"
    assert parsed.confidence == "high"


def test_parse_invalid_json():
    assert parse_json_answer("I think the answer is 24 days") is None
    assert parse_json_answer('{"answer": "x", "confidence": "very sure"}') is None


# ---------- cache ----------
def test_cache_normalizes_question():
    key1 = TTLCache.make_key("What is the leave policy?", 0.2, 0.9)
    key2 = TTLCache.make_key("  what is the LEAVE policy ", 0.2, 0.9)
    assert key1 == key2


def test_cache_expiry_and_lru():
    cache = TTLCache(max_items=2, ttl_seconds=1)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # "a" should be removed
    assert cache.get("a") is None
    assert cache.get("c") == 3
    time.sleep(1.1)
    assert cache.get("c") is None


# ---------- rate limiter ----------
def test_rate_limiter():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert all(limiter.is_allowed("1.1.1.1") for _ in range(3))
    assert limiter.is_allowed("1.1.1.1") is False
    assert limiter.is_allowed("2.2.2.2") is True  # other users are not affected


# ---------- graceful degradation ----------
def test_degraded_answer_when_llm_down(monkeypatch):
    async def fake_chat_completion(*args, **kwargs):
        raise llm_client.AllProvidersFailedError("down")

    fake_chunks = [{"text": "Annual leave is 24 days.", "source": "acme_hr_policy.md", "score": 0.8}]
    monkeypatch.setattr(assistant, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(assistant.vector_store, "search", lambda q, k: fake_chunks)

    result = asyncio.run(assistant.answer_question("How many leaves?"))
    assert result["degraded"] is True
    assert result["sources"] == ["acme_hr_policy.md"]
