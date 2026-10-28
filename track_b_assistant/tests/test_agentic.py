"""
Unit tests for the W16 agentic loop (app/agentic.py).

No API key is needed: the LLM is mocked with monkeypatch, the same pattern
`tests/test_basic.py` uses for the graceful-degradation test. Each test fakes a
short, deterministic sequence of Researcher/Verifier JSON replies so we can
check the loop's *control logic* (stopping conditions, tool-failure handling)
without depending on a real model's behaviour.

    pytest -q tests/test_agentic.py
"""
import asyncio
from types import SimpleNamespace

import pytest

from app import agentic, config


def fake_response(content, total_tokens=42):
    """Builds a minimal object shaped like the OpenAI SDK response `chat_completion` returns."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_chat_completion(script):
    """
    Returns a fake `chat_completion(messages, ...)` that yields the next canned
    JSON string from `script` on every call, in order, so a test can script a
    whole researcher/verifier/researcher/... conversation.
    """
    calls = {"i": 0}

    async def fake(messages, tools=None, json_mode=False, temperature=None, top_p=None):
        content = script[calls["i"]]
        calls["i"] += 1
        return fake_response(content), "fake-model"

    return fake


# ---------- stops as soon as the verifier says "sufficient" ----------
def test_loop_stops_when_verifier_says_sufficient(monkeypatch):
    script = [
        '{"action": "search", "query": "paid leave days"}',                       # researcher: search
        '{"verdict": "sufficient", "reason": "found a direct answer"}',           # verifier
        '{"action": "finish", "answer": "24 days per year.", "sources": ["acme_hr_policy.md"], "confidence": "high"}',
    ]
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))
    monkeypatch.setattr(
        agentic.vector_store, "search",
        lambda q, k: [{"text": "Employees get 24 days of paid annual leave.", "source": "acme_hr_policy.md", "score": 0.9}],
    )

    result = asyncio.run(agentic.agentic_answer_question("How many paid leave days?", max_iterations=4))

    assert result["verification"] == "sufficient"
    assert result["iterations"] == 2  # 1 search iteration + 1 finish iteration
    assert "acme_hr_policy.md" in result["sources"]
    assert result["degraded"] is False


# ---------- never exceeds the hard iteration cap ----------
def test_loop_stops_at_max_iterations(monkeypatch):
    # verifier always says "insufficient" -> researcher should be asked to search again every time,
    # and the loop must still stop once max_iterations is used up (the required stopping condition).
    script = []
    for _ in range(5):
        script.append('{"action": "search", "query": "leave policy detail"}')
        script.append('{"verdict": "insufficient", "reason": "not specific enough", "missing": "exact number"}')
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))
    monkeypatch.setattr(
        agentic.vector_store, "search",
        lambda q, k: [{"text": "Some unrelated passage.", "source": "acme_it_support.txt", "score": 0.3}],
    )

    result = asyncio.run(agentic.agentic_answer_question("Some ambiguous question", max_iterations=3))

    assert result["iterations"] == 3
    assert result["verification"] == "max_iterations_reached"
    assert result["confidence"] == "low"


# ---------- conflicting evidence across sources is reported, not silently resolved ----------
def test_conflicting_evidence_is_reported(monkeypatch):
    script = [
        '{"action": "search", "query": "price of business plan"}',
        '{"verdict": "conflicting", "reason": "two different prices found in two documents"}',
        '{"action": "finish", "answer": "Sources disagree on the exact price; please check with sales.", '
        '"sources": ["acme_product_faq.md"], "confidence": "low"}',
    ]
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))
    monkeypatch.setattr(
        agentic.vector_store, "search",
        lambda q, k: [
            {"text": "Business plan costs $20/user/month.", "source": "acme_product_faq.md", "score": 0.8},
            {"text": "Business plan costs $25/user/month.", "source": "acme_it_support.txt", "score": 0.7},
        ],
    )

    result = asyncio.run(agentic.agentic_answer_question("What does the Business plan cost?", max_iterations=4))

    assert result["verification"] == "conflicting"
    assert result["confidence"] == "low"


# ---------- the model can ask the user for clarification instead of guessing ----------
def test_clarification_path(monkeypatch):
    script = ['{"action": "ask_clarification", "clarification_question": "Which plan are you asking about?"}']
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))

    result = asyncio.run(agentic.agentic_answer_question("How much does it cost?", max_iterations=4))

    assert result["verification"] == "clarification_needed"
    assert "Which plan" in result["answer"]
    assert result["iterations"] == 1


# ---------- failure injection: a broken search tool must not be papered over ----------
def test_simulated_search_failure_is_recognised_not_hallucinated(monkeypatch):
    monkeypatch.setattr(config, "SIMULATE_SEARCH_FAILURE", True)
    script = [
        '{"action": "search", "query": "paid leave days"}',
        '{"action": "finish", "answer": "I could not verify this because document search failed.", '
        '"sources": [], "confidence": "low"}',
    ]
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))

    result = asyncio.run(agentic.agentic_answer_question("How many paid leave days?", max_iterations=4))

    # the tool failure must show up in the evidence log as a failure, not as "no results"
    assert any("TOOL FAILURE" in e["note"] for e in result["evidence_log"])
    assert result["confidence"] == "low"
    monkeypatch.setattr(config, "SIMULATE_SEARCH_FAILURE", False)


# ---------- calculator action is threaded through without calling the LLM to do maths ----------
def test_calculate_action(monkeypatch):
    script = [
        '{"action": "calculate", "expression": "24 - 7"}',
        '{"action": "finish", "answer": "You have 17 leave days left.", "sources": [], "confidence": "high"}',
    ]
    monkeypatch.setattr(agentic, "chat_completion", make_chat_completion(script))

    result = asyncio.run(agentic.agentic_answer_question("I used 7 of my 24 leaves, how many are left?", max_iterations=4))

    assert any(e["action"] == "calculate" and "= 17" in e["note"] for e in result["evidence_log"])
    assert result["iterations"] == 2


# ---------- graceful degradation still works inside the agentic loop ----------
def test_llm_down_returns_degraded_result(monkeypatch):
    async def fake_down(*args, **kwargs):
        raise agentic.AllProvidersFailedError("down")

    monkeypatch.setattr(agentic, "chat_completion", fake_down)

    result = asyncio.run(agentic.agentic_answer_question("Anything", max_iterations=4))

    assert result["degraded"] is True
    assert result["verification"] == "llm_unavailable"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
