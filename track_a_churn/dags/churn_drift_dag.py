"""
Optional bonus (Track A): Airflow DAG that runs the drift check on a schedule
and, when significant drift is detected, logs a retraining recommendation
(and can be wired to actually trigger `src.train` - see the TODO below).

This DAG is provided as code for review; it was not run against a live
Airflow scheduler in this environment (that would need a full Airflow
metadata DB + webserver, out of scope for this repo's CI). It's written
against the stable Airflow 2.x TaskFlow API so `astro dev start` / any
standard Airflow 2.x install can pick it up by pointing DAGS_FOLDER here.

    airflow dags list                    # after copying this file into $AIRFLOW_HOME/dags
    airflow dags test churn_drift_check 2026-01-01

Schedule: daily at 03:00 (adjust via `schedule` below - weekly/monthly for a
slower-moving dataset like this one would also be reasonable).

Trigger condition: the DAG re-runs src.monitoring's drift check against a
fresh 'current' sample, reads reports/drift_summary.json, and branches:
    - share_drifted_columns > DRIFT_SHARE_THRESHOLD  -> "significant drift":
      log a retraining recommendation (and optionally trigger src.train).
    - otherwise -> log "no action needed" and stop.
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
DRIFT_SHARE_THRESHOLD = 0.25  # >25% of columns drifted is "significant" for this small feature set


def _run_module(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{module} failed with exit code {result.returncode}")


@dag(
    dag_id="churn_drift_check",
    schedule="0 3 * * *",  # daily at 03:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "track-a", "drift"],
)
def churn_drift_check():
    @task
    def run_drift_check() -> dict:
        """Re-runs the reference-vs-current drift report and returns the summary."""
        _run_module("src.monitoring")
        summary_path = PROJECT_ROOT / "reports" / "drift_summary.json"
        return json.loads(summary_path.read_text())

    @task
    def evaluate_and_act(summary: dict) -> str:
        """
        The 'trigger condition' step: decide whether drift is significant enough
        to recommend (or trigger) retraining, and log the decision either way -
        so a human reviewing DAG runs can see why nothing happened on a quiet day.
        """
        share = summary["share_drifted_columns"]
        if share <= DRIFT_SHARE_THRESHOLD:
            print(f"No action: only {share:.1%} of columns drifted (threshold {DRIFT_SHARE_THRESHOLD:.0%}).")
            raise AirflowSkipException("Drift below threshold, nothing to do")

        print(
            f"SIGNIFICANT DRIFT DETECTED: {share:.1%} of columns drifted "
            f"(threshold {DRIFT_SHARE_THRESHOLD:.0%}). Churn rate moved from "
            f"{summary['reference_churn_rate']:.3f} to {summary['current_churn_rate']:.3f}."
        )
        print("Recommendation: retrain telco_churn_classifier on the latest data and "
              "re-run the full run comparison before promoting a new Production version.")
        # TODO (left as a recommendation, not auto-executed, so a human reviews the
        # comparison table before a new model silently replaces Production):
        #   _run_module("src.train")
        return "retrain_recommended"

    evaluate_and_act(run_drift_check())


churn_drift_check()
