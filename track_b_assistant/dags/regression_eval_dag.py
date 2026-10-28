"""
Optional bonus (Track B): Airflow DAG that runs the W16 evaluation harness
(scripts/eval_agentic.py) on a schedule and flags metric degradation beyond a
defined threshold, compared against the last known-good baseline.

Like track_a_churn/dags/churn_drift_dag.py, this is provided as reviewable
code (TaskFlow API, Airflow 2.x) and was not run against a live Airflow
scheduler in this environment.

    airflow dags test nightly_regression_eval 2026-01-01

Schedule: nightly at 02:00.
Trigger condition: after each run, compares the harness's task-completion
rate and average trajectory length against reports/eval_baseline.json (the
last accepted run). A drop in completion rate beyond COMPLETION_DROP_THRESHOLD,
or a trajectory-length increase beyond TRAJECTORY_INCREASE_THRESHOLD (agent
needing more iterations to do the same job - a cost regression even without a
correctness regression), raises an alert task.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "reports" / "eval_baseline.json"
COMPLETION_DROP_THRESHOLD = 0.10  # 10 percentage points
TRAJECTORY_INCREASE_THRESHOLD = 1.0  # +1 average iteration


def _run_harness() -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/eval_agentic.py", "--mock"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("eval_agentic.py failed")
    return json.loads((PROJECT_ROOT / "scripts" / "eval_results.md").read_text()[:0] or "{}") \
        if False else _parse_results_md()


def _parse_results_md() -> dict:
    """scripts/eval_agentic.py writes a Markdown report, not JSON - pull the two
    headline numbers back out of it so this DAG doesn't need its own duplicate
    harness logic."""
    text = (PROJECT_ROOT / "scripts" / "eval_results.md").read_text()
    completion_rate = float(text.split("Task completion rate: **")[1].split("%")[0]) / 100
    avg_iterations = float(text.split("Average trajectory length: **")[1].split(" iterations")[0])
    return {"task_completion_rate": completion_rate, "avg_trajectory_length": avg_iterations}


@dag(
    dag_id="nightly_regression_eval",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "track-b", "regression"],
)
def nightly_regression_eval():
    @task
    def run_eval() -> dict:
        return _run_harness()

    @task
    def compare_to_baseline(current: dict) -> str:
        if not BASELINE_PATH.exists():
            BASELINE_PATH.write_text(json.dumps(current, indent=2))
            print("No baseline yet - saving this run as the new baseline.")
            raise AirflowSkipException("First run, nothing to compare against")

        baseline = json.loads(BASELINE_PATH.read_text())
        completion_drop = baseline["task_completion_rate"] - current["task_completion_rate"]
        trajectory_increase = current["avg_trajectory_length"] - baseline["avg_trajectory_length"]

        degraded = completion_drop > COMPLETION_DROP_THRESHOLD or trajectory_increase > TRAJECTORY_INCREASE_THRESHOLD
        if not degraded:
            print(f"No significant degradation. completion_drop={completion_drop:.1%}, "
                  f"trajectory_increase={trajectory_increase:+.1f}")
            raise AirflowSkipException("Within thresholds")

        print(
            f"METRIC DEGRADATION DETECTED: completion rate dropped {completion_drop:.1%} "
            f"(threshold {COMPLETION_DROP_THRESHOLD:.0%}) and/or average trajectory length "
            f"increased by {trajectory_increase:+.1f} iterations (threshold "
            f"{TRAJECTORY_INCREASE_THRESHOLD:+.1f}). Baseline={baseline}, current={current}."
        )
        # TODO (left as an alert, not auto-executed): wire this to the team's paging
        # tool. A metric drop should be looked at by a person before any prompt
        # rollback happens automatically.
        return "alert_sent"

    compare_to_baseline(run_eval())


nightly_regression_eval()
