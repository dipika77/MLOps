"""
Drift monitoring for the churn model (Evidently AI).

    uv run python -m src.monitoring

What it does:
1. Splits the raw Telco data into `reference` (70%) and `current` (30%), as if
   `reference` were the training-time distribution and `current` were freshly
   arrived production data.
2. Injects SYNTHETIC drift into `current` only, so the reports below have
   something real to detect (see `inject_drift`):
     - numeric drift:      MonthlyCharges shifted by a random per-row offset
     - categorical drift:  Contract oversampled toward "Month-to-month"
     - label drift:        5% of Churn labels flipped
3. Runs Evidently's classic Data Drift report and Target Drift report
   (reference vs. current) and saves them as HTML.
4. Computes one CUSTOM metric beyond Evidently's built-in defaults, using
   Evidently's own Metric/MetricCalculation API (not just a bare Python
   computation): the difference in mean MonthlyCharges between reference and
   current, and separately the shift in churn rate within the
   Contract == "Month-to-month" segment (the exact segment we skewed).
5. Logs both HTML reports and the custom metric values to MLflow as an
   artifact + metrics on a dedicated "drift_monitoring" run.
"""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently import Dataset, Report
from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report as LegacyReport

from src.data import CATEGORICAL_COLS, ID_COL, NUMERIC_COLS, load_raw, reference_current_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("telco_churn_monitoring")

RANDOM_STATE = 7


def inject_drift(current: pd.DataFrame, random_state=RANDOM_STATE) -> pd.DataFrame:
    """
    Deliberately perturbs the 'current' (production) set so the drift report
    has real, engineered drift to detect - see README section 9c for why each
    perturbation was chosen and what it's meant to simulate.
    """
    rng = np.random.default_rng(random_state)
    drifted = current.copy()

    # 1) numeric drift: MonthlyCharges shifted by a random positive offset per row
    #    (simulates a pricing change / new plan mix hitting production)
    offset = rng.uniform(15, 40, size=len(drifted))
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] + offset

    # 2) categorical drift: oversample Contract == "Month-to-month" relative to
    #    its natural frequency (simulates a marketing push skewing the incoming mix)
    mtm_mask = drifted["Contract"] == "Month-to-month"
    non_mtm_idx = drifted.index[~mtm_mask]
    flip_idx = rng.choice(non_mtm_idx, size=int(len(non_mtm_idx) * 0.6), replace=False)
    drifted.loc[flip_idx, "Contract"] = "Month-to-month"

    # 3) label drift: flip 5% of Churn labels (simulates a labeling/process change,
    #    or genuine concept drift in who churns)
    flip_label_idx = rng.choice(drifted.index, size=int(len(drifted) * 0.05), replace=False)
    drifted.loc[flip_label_idx, "Churn"] = drifted.loc[flip_label_idx, "Churn"].map({"Yes": "No", "No": "Yes"})

    return drifted


# ---------------------------------------------------------------------------
# Custom metric (Evidently's Metric/MetricCalculation API, not a built-in preset):
# the shift in churn rate specifically within the segment we skewed
# (Contract == "Month-to-month"), which a generic dataset-level drift check
# would not surface on its own.
# ---------------------------------------------------------------------------
class ChurnRateShiftInSegment(SingleValueMetric):
    segment_column: str
    segment_value: str
    target_column: str


class ChurnRateShiftInSegmentCalculation(SingleValueCalculation[ChurnRateShiftInSegment]):
    def calculate(self, context, current_data: Dataset, reference_data):
        def churn_rate(dataset: Dataset) -> float:
            df = dataset.as_dataframe()
            seg = df[df[self.metric.segment_column] == self.metric.segment_value]
            if len(seg) == 0:
                return float("nan")
            return float((seg[self.metric.target_column] == "Yes").mean())

        cur_rate = churn_rate(current_data)
        ref_rate = churn_rate(reference_data) if reference_data is not None else float("nan")
        shift = cur_rate - ref_rate
        result = self.result(shift)
        return result

    def display_name(self):
        return (
            f"Churn rate shift in {self.metric.segment_column}="
            f"{self.metric.segment_value} (current - reference)"
        )


class MeanValueDiff(SingleValueMetric):
    column: str


class MeanValueDiffCalculation(SingleValueCalculation[MeanValueDiff]):
    def calculate(self, context, current_data: Dataset, reference_data):
        cur_mean = current_data.column(self.metric.column).data.mean()
        ref_mean = reference_data.column(self.metric.column).data.mean() if reference_data is not None else None
        value = float(cur_mean) if ref_mean is None else float(cur_mean - ref_mean)
        return self.result(value)

    def display_name(self):
        return f"Mean diff of {self.metric.column} (current - reference)"


def run_custom_metrics(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_ds = Dataset.from_pandas(reference)
    cur_ds = Dataset.from_pandas(current)

    report = Report(
        metrics=[
            MeanValueDiff(column="MonthlyCharges"),
            ChurnRateShiftInSegment(
                segment_column="Contract", segment_value="Month-to-month", target_column="Churn"
            ),
        ]
    )
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    html_path = REPORTS_DIR / "custom_metrics_report.html"
    snapshot.save_html(str(html_path))

    values = {m["metric_name"]: m["value"] for m in snapshot.dict()["metrics"]}
    return {"html_path": html_path, "values": values}


def main():
    df = load_raw()
    reference, current_clean = reference_current_split(df, current_fraction=0.30, random_state=42)
    current = inject_drift(current_clean, random_state=RANDOM_STATE)

    # customerID is a unique identifier, not a real feature - every split makes it look
    # 100% "drifted" between reference/current (Jensen-Shannon ~1.0) for a trivial reason
    # that has nothing to do with real drift, so it would just be noise in the report.
    reference = reference.drop(columns=[ID_COL])
    current = current.drop(columns=[ID_COL])

    # ---- classic Evidently Data Drift + Target Drift reports (legacy API,
    # same objects the assignment names by name) ----
    data_drift_report = LegacyReport(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference, current_data=current)
    data_drift_html = REPORTS_DIR / "data_drift_report.html"
    data_drift_report.save_html(str(data_drift_html))
    data_drift_result = data_drift_report.as_dict()

    target_drift_report = LegacyReport(metrics=[TargetDriftPreset()])
    target_drift_report.run(reference_data=reference, current_data=current)
    target_drift_html = REPORTS_DIR / "target_drift_report.html"
    target_drift_report.save_html(str(target_drift_html))
    target_drift_result = target_drift_report.as_dict()

    # ---- custom metric (Evidently Metric/MetricCalculation API) ----
    custom = run_custom_metrics(reference, current)

    # ---- pull out the headline numbers for the console + MLflow metrics ----
    dataset_drift_metric = next(
        m for m in data_drift_result["metrics"] if m["metric"] == "DatasetDriftMetric"
    )["result"]
    n_drifted_columns = dataset_drift_metric["number_of_drifted_columns"]
    share_drifted_columns = dataset_drift_metric["share_of_drifted_columns"]
    dataset_drift_detected = dataset_drift_metric["dataset_drift"]

    ref_churn_rate = (reference["Churn"] == "Yes").mean()
    cur_churn_rate = (current["Churn"] == "Yes").mean()

    print(f"Dataset drift detected: {dataset_drift_detected}")
    print(f"Drifted columns: {n_drifted_columns} ({share_drifted_columns:.1%} of columns)")
    print(f"Churn rate: reference={ref_churn_rate:.3f}, current={cur_churn_rate:.3f}, "
          f"delta={cur_churn_rate - ref_churn_rate:+.3f}")
    print(f"Custom metrics: {custom['values']}")

    summary = {
        "dataset_drift_detected": bool(dataset_drift_detected),
        "n_drifted_columns": n_drifted_columns,
        "share_drifted_columns": share_drifted_columns,
        "reference_churn_rate": float(ref_churn_rate),
        "current_churn_rate": float(cur_churn_rate),
        "churn_rate_delta": float(cur_churn_rate - ref_churn_rate),
        "custom_metrics": custom["values"],
        "columns_perturbed": ["MonthlyCharges (numeric noise)", "Contract (categorical skew)", "Churn (5% label flip)"],
    }
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- log everything to MLflow ----
    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_metric("n_drifted_columns", n_drifted_columns)
        mlflow.log_metric("share_drifted_columns", share_drifted_columns)
        mlflow.log_metric("dataset_drift_detected", int(dataset_drift_detected))
        mlflow.log_metric("reference_churn_rate", ref_churn_rate)
        mlflow.log_metric("current_churn_rate", cur_churn_rate)
        mlflow.log_metric("churn_rate_delta", cur_churn_rate - ref_churn_rate)
        for name, value in custom["values"].items():
            safe_name = "".join(c if c.isalnum() else "_" for c in name)[:100]
            mlflow.log_metric(f"custom_{safe_name}", value)
        mlflow.log_artifact(str(data_drift_html), artifact_path="evidently_reports")
        mlflow.log_artifact(str(target_drift_html), artifact_path="evidently_reports")
        mlflow.log_artifact(str(custom["html_path"]), artifact_path="evidently_reports")
        mlflow.log_artifact(str(REPORTS_DIR / "drift_summary.json"), artifact_path="evidently_reports")

    print(f"\nReports saved to {REPORTS_DIR} and logged to MLflow experiment 'telco_churn_monitoring'.")


if __name__ == "__main__":
    main()
