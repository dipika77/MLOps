"""
W17 Track B, section 2: MLflow experiment tracking for prompt/configuration
versions of the W16 agentic feature.

    uv run python -m mlops.run_experiments

For each of prompts/prompt_v1.py, prompt_v2.py, prompt_v3.py this:
  1. Points app.agentic at that version's Researcher/Verifier prompts (and
     whatever config value that version also changes - see VERSION_CONFIGS).
  2. Runs a fixed set of test queries through the real agentic loop
     (app.agentic.agentic_answer_question), with a deterministic prompt-aware
     stand-in LLM (mlops/mock_llm.py) so results are reproducible without an
     API key - see that module's docstring for why this is a fair substitute
     and how to swap in a real model.
  3. Captures the FULL structured trace for every query (not just the final
     answer): every tool call with its arguments and raw result, the model's
     reasoning at each step, and how many iterations were used and why the
     loop stopped.
  4. Logs one MLflow run per version: params (prompt version + the other
     config value that changed), metrics (from a small harness, reusing the
     same measurement approach as scripts/eval_agentic.py), and artifacts
     (the prompt text, 2-3 representative traces - one clean success, one
     failure case - and a run report).

Each version exists because of a SPECIFIC failure seen in the previous
version's traces, not a speculative tweak - see the module docstrings in
prompts/prompt_v2.py and prompt_v3.py, and README.md section 9.b, for the
exact trace that motivated each change.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio  # noqa: E402

import mlflow  # noqa: E402

from app import agentic, config  # noqa: E402
from mlops.mock_llm import PromptAwareMockLLM, mock_search  # noqa: E402
from prompts import prompt_v1, prompt_v2, prompt_v3  # noqa: E402

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("acme_assistant_prompt_versions")

TRACES_DIR = PROJECT_ROOT / "reports" / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_QUERIES = [
    {"id": "q1_single_source", "question": "How many paid annual leave days do employees get per year?"},
    {"id": "q2_calculation", "question": "I have used 7 of my 24 annual leave days, how many are left?"},
    {"id": "q3_cross_source", "question": "Does the Business plan price match between the product FAQ and IT support notes?"},
    {"id": "q4_ambiguous", "question": "How much does it cost?"},
    {"id": "q5_single_source", "question": "How do I reset my password?"},
]

# The prompt is the primary thing under test, but each version here also carries one
# other configuration change the assignment asks us to track alongside it.
VERSION_CONFIGS = {
    "v1": {"module": prompt_v1, "max_iterations": 4, "notes": "baseline W16 prompts"},
    "v2": {"module": prompt_v2, "max_iterations": 5, "notes": "stricter verifier; +1 iteration to give the (still-broken) repeat-query behavior room to recover on its own"},
    "v3": {"module": prompt_v3, "max_iterations": 4, "notes": "verifier fix (v2) + researcher targets verifier's 'missing' field and never repeats a query"},
}


def apply_version(version: str):
    cfg = VERSION_CONFIGS[version]
    module = cfg["module"]
    agentic.RESEARCHER_SYSTEM_PROMPT = module.RESEARCHER_SYSTEM_PROMPT
    agentic.RESEARCHER_USER_TEMPLATE = module.RESEARCHER_USER_TEMPLATE
    agentic.VERIFIER_SYSTEM_PROMPT = module.VERIFIER_SYSTEM_PROMPT
    agentic.VERIFIER_USER_TEMPLATE = module.VERIFIER_USER_TEMPLATE
    config.MAX_AGENT_ITERATIONS = cfg["max_iterations"]


def check_tool_call_correctness(trajectory):
    valid_actions = {"search", "calculate", "ask_clarification", "finish"}
    correct, total = 0, 0
    for step in trajectory:
        total += 1
        action = step.get("action")
        ok = action in valid_actions
        if action == "search":
            ok = ok and bool(step.get("query"))
        if ok:
            correct += 1
    return correct, total


async def run_version(version: str, use_mock: bool):
    apply_version(version)
    if use_mock:
        mock = PromptAwareMockLLM()
        agentic.chat_completion = mock
        agentic.vector_store.search = mock_search

    results = []
    for q in TEST_QUERIES:
        result = await agentic.agentic_answer_question(q["question"], max_iterations=config.MAX_AGENT_ITERATIONS)
        results.append({**q, "result": result})
    return results


def summarize(version: str, results: list) -> dict:
    n = len(results)
    completed = sum(1 for r in results if not r["result"]["degraded"] and r["result"]["answer"])
    total_tokens = sum(r["result"]["total_tokens"] for r in results)
    avg_iterations = sum(r["result"]["iterations"] for r in results) / n
    total_correct = total_calls = 0
    for r in results:
        c, t = check_tool_call_correctness(r["result"]["trajectory"])
        total_correct += c
        total_calls += t
    return {
        "task_completion_rate": completed / n,
        "avg_trajectory_length": avg_iterations,
        "tool_call_correctness": total_correct / total_calls if total_calls else 1.0,
        "total_tokens": total_tokens,
        "max_iterations_hits": sum(1 for r in results if r["result"]["verification"] == "max_iterations_reached"),
        "conflicting_detected": sum(1 for r in results if r["result"]["verification"] == "conflicting"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", default=True, help="use the offline prompt-aware stand-in LLM (default)")
    parser.add_argument("--live", action="store_true", help="use the real LLM via app.llm_client (needs GROQ_API_KEY)")
    args = parser.parse_args()
    use_mock = not args.live

    if not use_mock and not config.GROQ_API_KEY:
        print("No GROQ_API_KEY set - re-run without --live, or set GROQ_API_KEY in .env")
        return

    all_summaries = {}
    for version in ["v1", "v2", "v3"]:
        cfg = VERSION_CONFIGS[version]
        results = asyncio.run(run_version(version, use_mock))
        summary = summarize(version, results)
        all_summaries[version] = summary

        with mlflow.start_run(run_name=f"prompt_{version}"):
            mlflow.log_param("prompt_version", version)
            mlflow.log_param("max_iterations", cfg["max_iterations"])
            mlflow.log_param("model_backend", "mock/prompt-aware-v1" if use_mock else config.PRIMARY_MODEL)
            mlflow.log_param("notes", cfg["notes"])
            mlflow.log_metrics(summary)

            # ---- artifacts: the prompt text itself ----
            prompt_path = PROJECT_ROOT / "prompts" / f"prompt_{version}.py"
            mlflow.log_artifact(str(prompt_path), artifact_path="prompt")

            # ---- artifacts: full structured traces for every query ----
            version_trace_dir = TRACES_DIR / version
            version_trace_dir.mkdir(exist_ok=True)
            for r in results:
                trace_path = version_trace_dir / f"{r['id']}.json"
                trace_path.write_text(json.dumps({"question": r["question"], **r["result"]}, indent=2))
            mlflow.log_artifacts(str(version_trace_dir), artifact_path="traces")

            print(f"[{version}] {summary}")

    report_lines = ["# Prompt version comparison (mock LLM)\n",
                    "| version | completion rate | avg iterations | tool-call correctness | total tokens | max-iter hits | conflicts flagged |",
                    "|---|---|---|---|---|---|---|"]
    for version, s in all_summaries.items():
        report_lines.append(
            f"| {version} | {s['task_completion_rate']:.0%} | {s['avg_trajectory_length']:.1f} "
            f"| {s['tool_call_correctness']:.0%} | {s['total_tokens']} | {s['max_iterations_hits']} | {s['conflicting_detected']} |"
        )
    report = "\n".join(report_lines)
    (REPORTS_DIR / "prompt_version_comparison.md").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
