"""
W16 evaluation harness for the agentic (Researcher/Verifier) feature.

Built from scratch (no evaluation framework / library) as required by the assignment.
It measures exactly what the assignment asks for:
    - task completion rate
    - tool-call correctness (right tool, valid/parseable arguments)
    - trajectory length (iterations per query)
    - a failure log, classified as hard / soft / cascading-soft failure
    - total tokens per query, and single-agent vs multi-agent token comparison

Two ways to run it:

    python scripts/eval_agentic.py               # real run, needs GROQ_API_KEY in .env
    python scripts/eval_agentic.py --mock         # offline run with a scripted fake LLM,
                                                   # so the harness itself can be graded/demoed
                                                   # without an API key or network access.

Either way it prints a Markdown results table and writes it to
scripts/eval_results.md.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agentic, assistant, config  # noqa: E402

# ---------------------------------------------------------------------------
# Test set. Each query is tagged with what it is meant to exercise, mirroring
# the assignment's "cross-source verification" feature: some queries are
# answerable from one source, some require checking more than one, one is
# ambiguous (should trigger ask_clarification), and one is answered with the
# failure-injection flag on (SIMULATE_SEARCH_FAILURE).
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {"id": "q1", "question": "How many paid annual leave days do employees get per year?", "kind": "single_source"},
    {"id": "q2", "question": "I have used 7 of my 24 annual leave days, how many are left?", "kind": "calculation"},
    {"id": "q3", "question": "What is the price of the Business plan, and does it match what IT support says about it?", "kind": "cross_source"},
    {"id": "q4", "question": "How do I reset my password and how long does IT take to respond to tickets?", "kind": "cross_source"},
    {"id": "q5", "question": "How much does it cost?", "kind": "ambiguous"},  # should ask for clarification
    {"id": "q6", "question": "What is Acme's policy on parental leave?", "kind": "single_source"},
    {"id": "q7", "question": "How many paid annual leave days do employees get per year?", "kind": "failure_injection"},  # run with SIMULATE_SEARCH_FAILURE=true
]

VALID_ACTIONS = {"search", "calculate", "ask_clarification", "finish"}


# ---------------------------------------------------------------------------
# Mock LLM: deterministic canned answers so the harness can run standalone.
# It is intentionally simple pattern matching, not a real model - its only job
# is to exercise the harness's measurement code end to end.
# ---------------------------------------------------------------------------
class MockLLM:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, messages, tools=None, json_mode=False, temperature=None, top_p=None):
        self.call_count += 1
        user_content = messages[-1]["content"]
        system_content = messages[0]["content"]
        is_verifier = "You are the VERIFIER agent" in system_content
        is_single_agent_baseline = "Acme Assistant" in system_content  # app/prompts.py SYSTEM_PROMPT (W15)

        if is_single_agent_baseline:
            # W15-style assistant: answer directly from context in one shot, no tool call, no verifier.
            content = (
                '{"answer": "Based on the provided context, here is the answer.", '
                '"sources": ["acme_hr_policy.md"], "confidence": "medium", "follow_up_questions": []}'
            )
            return self._response(content), "mock/single-agent-v1"

        if is_verifier:
            if "conflict" in user_content.lower() or "20/user" in user_content or "25/user" in user_content:
                content = '{"verdict": "conflicting", "reason": "two different figures found"}'
            elif "no matching passages" in user_content or "TOOL FAILURE" in user_content:
                content = '{"verdict": "insufficient", "reason": "no usable evidence", "missing": "any evidence"}'
            else:
                content = '{"verdict": "sufficient", "reason": "evidence directly answers the question"}'
            return self._response(content), "mock/verifier-v1"

        # researcher
        if "ITERATIONS LEFT: 4" in user_content or "(empty" in user_content:
            if "how much does it cost" in user_content.lower():
                content = '{"action": "ask_clarification", "clarification_question": "Which plan or service are you asking about?"}'
            elif "used 7 of my 24" in user_content.lower():
                content = '{"action": "calculate", "expression": "24 - 7"}'
            else:
                content = '{"action": "search", "query": "' + user_content.splitlines()[1][:40] + '"}'
            return self._response(content), "mock/researcher-v1"

        if "verifier: sufficient" in user_content.lower():
            content = (
                '{"action": "finish", "answer": "Based on the verified evidence, here is the answer.", '
                '"sources": ["acme_hr_policy.md"], "confidence": "high"}'
            )
        elif "verifier: conflicting" in user_content.lower():
            content = (
                '{"action": "finish", "answer": "Sources disagree, so I cannot give one confident number.", '
                '"sources": [], "confidence": "low"}'
            )
        elif "tool failure" in user_content.lower():
            content = (
                '{"action": "finish", "answer": "Document search is unavailable, so I cannot verify this right now.", '
                '"sources": [], "confidence": "low"}'
            )
        elif "= 17" in user_content:
            content = '{"action": "finish", "answer": "You have 17 leave days left.", "sources": [], "confidence": "high"}'
        else:
            content = '{"action": "search", "query": "more specific query"}'
        return self._response(content), "mock/researcher-v1"

    @staticmethod
    def _response(content):
        message = SimpleNamespace(content=content, tool_calls=None)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(total_tokens=180)  # rough, fixed cost per mock call
        return SimpleNamespace(choices=[choice], usage=usage)


def mock_search(query, top_k=None):
    q = query.lower()
    if "business plan" in q or "price" in q:
        return [
            {"text": "The Business plan costs $20/user/month billed annually.", "source": "acme_product_faq.md", "score": 0.81},
            {"text": "Support notes list the Business plan at $25/user/month.", "source": "acme_it_support.txt", "score": 0.74},
        ]
    if "password" in q:
        return [{"text": "Go to Settings > Security > Reset password.", "source": "acme_it_support.txt", "score": 0.9}]
    if "ticket" in q:
        return [{"text": "IT tickets are answered within 1 business day.", "source": "acme_it_support.txt", "score": 0.7}]
    if "parental" in q:
        return [{"text": "Parental leave is 12 weeks paid.", "source": "acme_hr_policy.md", "score": 0.88}]
    return [{"text": "Employees get 24 days of paid annual leave, accrued monthly.", "source": "acme_hr_policy.md", "score": 0.9}]


def classify_failure(result):
    """
    Failure taxonomy used in class:
      - hard failure: the agent gives up / errors out with no usable answer at all.
      - soft failure: the agent answers, but with an outcome you would not want in
        production even though nothing crashed (unverified, low confidence, wrong action).
      - cascading soft failure: one soft failure feeds the next step and compounds
        (here: a tool failure that the agent does NOT recognise and answers confidently
        anyway - the bad tool output cascades into a confidently wrong final answer).
    """
    if result.get("degraded"):
        return "hard failure", "LLM/provider unavailable, no answer produced"
    if result["verification"] == "max_iterations_reached":
        return "soft failure", "ran out of iterations before verification succeeded"
    if result["verification"] == "conflicting" and result["confidence"] != "low":
        return "cascading soft failure", "conflicting evidence but confidence was not lowered"
    tool_failed = any(e.get("action") == "search" and "TOOL FAILURE" in e.get("note", "") for e in result["evidence_log"])
    if tool_failed and result["confidence"] not in ("low",):
        return "cascading soft failure", "search tool failed but the agent still answered with non-low confidence"
    if tool_failed and result["verification"] not in ("insufficient", "not_verified"):
        return "cascading soft failure", "tool failure did not stop the loop from claiming a verdict"
    return None, None


def check_tool_call_correctness(result):
    """A trajectory entry is 'correct' if its action is a known action and (for search/calculate)
    it carries a non-empty, well-formed argument."""
    total, correct = 0, 0
    for step in result["trajectory"]:
        total += 1
        action = step.get("action")
        ok = action in VALID_ACTIONS
        if action == "search":
            ok = ok and bool(step.get("query"))
        if action == "calculate":
            ok = ok and bool(step.get("expression"))
        if ok:
            correct += 1
    return correct, total


async def run_one(query, use_mock, single_agent_baseline=False):
    if query["kind"] == "failure_injection":
        config.SIMULATE_SEARCH_FAILURE = True
    else:
        config.SIMULATE_SEARCH_FAILURE = False

    start = time.time()
    if single_agent_baseline:
        # W15 baseline: single LLM call with the tool-calling loop, no verifier.
        # search_documents/tools still resolve through vector_store.search (mocked below).
        result_raw = await assistant.answer_question(query["question"])
        result = {
            "answer": result_raw["answer"],
            "sources": result_raw["sources"],
            "confidence": result_raw["confidence"],
            "verification": "not_applicable_single_agent",
            "iterations": 1,
            "trajectory": [{"iteration": 1, "action": "finish"}],
            "evidence_log": [],
            "total_tokens": result_raw.get("total_tokens", 0),
            "model_used": [result_raw.get("model_used")] if result_raw.get("model_used") else [],
            "degraded": result_raw["degraded"],
        }
    else:
        result = await agentic.agentic_answer_question(query["question"], max_iterations=config.MAX_AGENT_ITERATIONS)
    latency_ms = int((time.time() - start) * 1000)
    config.SIMULATE_SEARCH_FAILURE = False
    return result, latency_ms


async def main(use_mock):
    if use_mock:
        mock_llm = MockLLM()
        agentic.chat_completion = mock_llm
        assistant.chat_completion = mock_llm
        agentic.vector_store.search = mock_search
        assistant.vector_store.search = mock_search
    elif not config.GROQ_API_KEY and not config.VLLM_BASE_URL:
        print("No GROQ_API_KEY/VLLM_BASE_URL configured and --mock was not passed. Re-run with --mock, "
              "or set GROQ_API_KEY in ai_assistant/.env for a live run.")
        return

    rows = []
    failures = []
    completed = 0
    total_tokens_multi = 0
    total_tokens_single = 0

    for query in TEST_QUERIES:
        result, latency_ms = await run_one(query, use_mock)
        correct, total = check_tool_call_correctness(result)
        failure_type, failure_reason = classify_failure(result)

        # every query that is not a hard failure and produced a non-empty answer counts as "completed",
        # even if it correctly asked for clarification instead of guessing.
        is_completed = not result.get("degraded") and bool(result.get("answer"))
        completed += int(is_completed)
        if failure_type:
            failures.append({"id": query["id"], "question": query["question"], "type": failure_type, "reason": failure_reason})

        total_tokens_multi += result["total_tokens"]

        rows.append({
            "id": query["id"],
            "kind": query["kind"],
            "verification": result["verification"],
            "iterations": result["iterations"],
            "tool_calls_correct": f"{correct}/{total}",
            "tokens": result["total_tokens"],
            "latency_ms": latency_ms,
            "completed": "yes" if is_completed else "no",
        })

        # single-agent (W15) baseline on the same query, for the token comparison
        single_result, _ = await run_one(query, use_mock, single_agent_baseline=True)
        total_tokens_single += single_result["total_tokens"]

    n = len(TEST_QUERIES)
    avg_iterations = sum(r["iterations"] for r in rows) / n
    completion_rate = completed / n
    total_correct = sum(int(r["tool_calls_correct"].split("/")[0]) for r in rows)
    total_calls = sum(int(r["tool_calls_correct"].split("/")[1]) for r in rows) or 1
    tool_call_accuracy = total_correct / total_calls

    report_lines = []
    report_lines.append(f"# Agentic evaluation results ({'mock LLM' if use_mock else 'live LLM'})\n")
    report_lines.append(f"- Task completion rate: **{completion_rate:.0%}** ({completed}/{n} queries)")
    report_lines.append(f"- Tool-call correctness: **{tool_call_accuracy:.0%}** ({total_correct}/{total_calls} trajectory steps)")
    report_lines.append(f"- Average trajectory length: **{avg_iterations:.1f} iterations** (cap: {config.MAX_AGENT_ITERATIONS})")
    report_lines.append(f"- Multi-agent (Researcher+Verifier) total tokens across {n} queries: **{total_tokens_multi}**")
    report_lines.append(f"- Single-agent (W15 baseline) total tokens across {n} queries: **{total_tokens_single}**")
    if total_tokens_single:
        overhead = (total_tokens_multi - total_tokens_single) / total_tokens_single
        report_lines.append(f"- Coordination overhead of the second (Verifier) agent: **{overhead:+.0%}** tokens vs. the single-agent baseline")
    report_lines.append("")
    report_lines.append("## Per-query results\n")
    report_lines.append("| id | kind | verification | iterations | tool-calls correct | tokens | latency (ms) | completed |")
    report_lines.append("|----|------|--------------|-----------|---------------------|--------|---------------|-----------|")
    for r in rows:
        report_lines.append(
            f"| {r['id']} | {r['kind']} | {r['verification']} | {r['iterations']} | {r['tool_calls_correct']} "
            f"| {r['tokens']} | {r['latency_ms']} | {r['completed']} |"
        )
    report_lines.append("")
    report_lines.append("## Failure log\n")
    if not failures:
        report_lines.append("No failures recorded.")
    else:
        report_lines.append("| id | type | reason |")
        report_lines.append("|----|------|--------|")
        for f in failures:
            report_lines.append(f"| {f['id']} | {f['type']} | {f['reason']} |")

    report = "\n".join(report_lines)
    print(report)

    out_path = Path(__file__).resolve().parent / "eval_results.md"
    out_path.write_text(report + "\n")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="run offline with a scripted fake LLM instead of a real provider")
    args = parser.parse_args()
    asyncio.run(main(args.mock))
