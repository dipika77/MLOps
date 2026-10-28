"""
W16 Task 3 - agentic feature: Cross-Source Verification.

Why a fixed pipeline is not enough (one sentence, as the assignment asks for):
A fixed "retrieve top-4 chunks once, then answer" pipeline cannot tell whether the
chunks it happened to retrieve actually settle the question or contradict each
other, so it either answers confidently from incomplete evidence or always does a
fixed number of extra searches whether they're needed or not - here, the number and
direction of searches must depend on what a second, independent agent decides the
evidence actually shows.

Design summary (see README.md section 9 "Task 3" for the full write-up):

  RESEARCHER agent  -> decides ONE action per iteration: search / calculate /
                        ask_clarification / finish
  VERIFIER agent    -> after each search, independently judges the evidence
                        gathered so far: sufficient / insufficient / conflicting
                        (a *different* LLM call from the one that will write the
                        final answer - this is the multi-agent, specialization part)

  Loop stops when: the verifier says "sufficient", the researcher asks for
  clarification, or MAX_AGENT_ITERATIONS is reached (hard cap - the loop can
  never run forever).

  Context engineering: STRUCTURED EXTERNAL NOTES. Raw retrieved chunk text is
  used once (to answer/verify the iteration it was retrieved in) and then
  compacted into a one-line note kept in `evidence_log`. Only the compact log -
  never the accumulated raw chunks - is replayed into the researcher/verifier
  prompts on every later iteration, so prompt size stays ~flat across
  iterations instead of growing with every search (see README for the context
  problem this fixes).
"""
import asyncio
import json
import logging
import re

from app import config
from app.agentic_prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_TEMPLATE,
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
    format_evidence_log,
)
from app.llm_client import AllProvidersFailedError, chat_completion
from app.rag import vector_store
from app.tools import calculator

logger = logging.getLogger(__name__)


def _parse_json(text):
    """Same tolerant JSON extraction used in assistant.py (fenced code, stray text, etc.)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


def _count_tokens(response):
    usage = getattr(response, "usage", None)
    return int(usage.total_tokens) if usage and usage.total_tokens else 0


async def _search_documents_for_agent(query):
    """
    Wraps the same vector search the W15 tools use, with an optional injected
    failure (config.SIMULATE_SEARCH_FAILURE) for the failure-injection test
    required by the assignment. On failure this returns an explicit error
    string - it never silently returns empty results - so the agent can tell
    "nothing was found" apart from "the search tool itself is broken".
    """
    if config.SIMULATE_SEARCH_FAILURE:
        logger.warning("SIMULATE_SEARCH_FAILURE is on: returning a fake tool failure for query %r", query)
        raise RuntimeError("simulated failure: document search backend unreachable (503)")
    return await asyncio.to_thread(vector_store.search, query, config.TOP_K)


async def _run_researcher(question, evidence_log, iterations_left):
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RESEARCHER_USER_TEMPLATE.format(
                question=question,
                iterations_left=iterations_left,
                evidence_log=format_evidence_log(evidence_log),
            ),
        },
    ]
    response, model_used = await chat_completion(messages, json_mode=True, temperature=0.1)
    decision = _parse_json(response.choices[0].message.content) or {
        "action": "finish",
        "answer": "Sorry, I could not decide how to proceed. Please rephrase your question.",
        "confidence": "low",
        "sources": [],
        "reasoning": "researcher output was not valid JSON",
    }
    return decision, model_used, _count_tokens(response)


async def _run_verifier(question, evidence_log):
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VERIFIER_USER_TEMPLATE.format(
                question=question, evidence_log=format_evidence_log(evidence_log)
            ),
        },
    ]
    response, model_used = await chat_completion(messages, json_mode=True, temperature=0.0)
    verdict = _parse_json(response.choices[0].message.content) or {
        "verdict": "insufficient",
        "reason": "verifier output was not valid JSON",
        "missing": "unknown",
    }
    return verdict, model_used, _count_tokens(response)


def _summarize_chunks(chunks):
    """Compact, human-readable one-line-per-source summary used for the evidence log."""
    if not chunks:
        return "no matching passages found", []
    lines = []
    sources = sorted({c["source"] for c in chunks})
    for c in chunks[:3]:
        snippet = c["text"][:160].replace("\n", " ")
        lines.append(f"({c['source']}, score={c['score']}) {snippet}...")
    return " | ".join(lines), sources


async def agentic_answer_question(question, max_iterations=None):
    """
    Runs the Researcher/Verifier loop and returns a result dict (see schemas.AgenticChatResponse).
    This is intentionally a separate function from assistant.answer_question(), not a wrapper around
    it in a while-loop, per the assignment's "do not just put the W15 pipeline in a loop" instruction:
    it has its own prompts, its own stopping logic, and a second agent role that W15 never had.
    """
    max_iterations = max_iterations or config.MAX_AGENT_ITERATIONS
    evidence_log = []  # the "structured external notes" - see module docstring
    trajectory = []    # per-iteration record, used by the evaluation harness
    total_tokens = 0
    models_used = set()
    all_sources = set()
    last_chunks = []  # only the MOST RECENT iteration's full chunks are kept in memory

    try:
        for iteration in range(1, max_iterations + 1):
            iterations_left = max_iterations - iteration + 1
            decision, model_used, tokens = await _run_researcher(question, evidence_log, iterations_left)
            total_tokens += tokens
            models_used.add(model_used)
            action = decision.get("action", "finish")

            if action == "ask_clarification":
                trajectory.append({"iteration": iteration, "action": action, "tokens": tokens})
                return {
                    "answer": decision.get("clarification_question", "Could you clarify your question?"),
                    "sources": [],
                    "confidence": "low",
                    "verification": "clarification_needed",
                    "iterations": iteration,
                    "trajectory": trajectory,
                    "evidence_log": evidence_log,
                    "total_tokens": total_tokens,
                    "model_used": sorted(models_used),
                    "degraded": False,
                }

            if action == "calculate":
                expr = decision.get("expression", "")
                result = calculator(expr)
                note = f"calculator({expr!r}) = {result}"
                evidence_log.append({"iteration": iteration, "action": "calculate", "query": expr, "note": note})
                trajectory.append({"iteration": iteration, "action": "calculate", "expression": expr, "tokens": tokens})
                continue  # let the researcher use the result on the next turn

            if action == "search":
                query = decision.get("query", question)
                try:
                    chunks = await _search_documents_for_agent(query)
                    last_chunks = chunks
                    note, sources = _summarize_chunks(chunks)
                    all_sources.update(sources)
                    tool_failed = False
                except Exception as e:
                    logger.error("Search tool failed during agentic loop: %s", e)
                    note = f"TOOL FAILURE: {e}"
                    chunks, sources, tool_failed = [], [], True

                entry = {"iteration": iteration, "action": "search", "query": query, "note": note}
                evidence_log.append(entry)

                if tool_failed:
                    # A broken tool is not the same as "no results" - the verifier must see it
                    # explicitly so the agent doesn't quietly treat silence as a real answer.
                    entry["verdict"] = "insufficient"
                    entry["verifier_reason"] = "search tool is unavailable"
                    trajectory.append({
                        "iteration": iteration, "action": "search", "query": query,
                        "tool_failed": True, "verdict": "insufficient", "tokens": tokens,
                    })
                    continue

                verdict, v_model, v_tokens = await _run_verifier(question, evidence_log)
                total_tokens += v_tokens
                models_used.add(v_model)
                entry["verdict"] = verdict.get("verdict", "insufficient")
                entry["verifier_reason"] = verdict.get("reason", "")
                entry["verifier_missing"] = verdict.get("missing", "")
                trajectory.append({
                    "iteration": iteration, "action": "search", "query": query,
                    "verdict": entry["verdict"], "tokens": tokens + v_tokens,
                })
                continue

            # action == "finish" (or unrecognised action defaults here)
            result = {
                "answer": decision.get("answer", "I could not find a confident answer."),
                "sources": sorted(all_sources) or decision.get("sources", []),
                "confidence": decision.get("confidence", "medium"),
                "verification": _final_verdict(evidence_log),
                "iterations": iteration,
                "trajectory": trajectory,
                "evidence_log": evidence_log,
                "total_tokens": total_tokens,
                "model_used": sorted(models_used),
                "degraded": False,
            }
            return result

        # ---- stopping condition: max iterations reached without the verifier ever saying "sufficient" ----
        logger.warning("Agentic loop hit MAX_AGENT_ITERATIONS (%s) without sufficient evidence", max_iterations)
        best_effort_note, _ = _summarize_chunks(last_chunks)
        return {
            "answer": (
                "I was not able to fully verify this answer within the allowed number of research steps. "
                f"Best available evidence: {best_effort_note}"
            ),
            "sources": sorted(all_sources),
            "confidence": "low",
            "verification": "max_iterations_reached",
            "iterations": max_iterations,
            "trajectory": trajectory,
            "evidence_log": evidence_log,
            "total_tokens": total_tokens,
            "model_used": sorted(models_used),
            "degraded": False,
        }

    except AllProvidersFailedError as e:
        logger.error("LLM unavailable during agentic loop: %s", e)
        return {
            "answer": "The AI service is temporarily unavailable, so I could not run the verification loop.",
            "sources": sorted(all_sources),
            "confidence": "low",
            "verification": "llm_unavailable",
            "iterations": len(trajectory),
            "trajectory": trajectory,
            "evidence_log": evidence_log,
            "total_tokens": total_tokens,
            "model_used": sorted(models_used),
            "degraded": True,
        }


def _final_verdict(evidence_log):
    verdicts = [e.get("verdict") for e in evidence_log if e.get("verdict")]
    if not verdicts:
        return "not_verified"
    if "conflicting" in verdicts:
        return "conflicting"
    if verdicts[-1] == "sufficient":
        return "sufficient"
    return "insufficient"
