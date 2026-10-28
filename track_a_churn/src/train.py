"""
Trains several churn models with different hyperparameters, logging every run
to MLflow (params, metrics, artifacts), then registers the best one.

    uv run python -m src.train

Produces a local `mlruns/` (MLflow's file-based tracking store) you can browse
with:
    uv run mlflow ui --backend-store-uri ./mlruns
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in this environment
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_raw, split_features_target

warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "telco_churn"
MODEL_NAME = "telco_churn_classifier"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MLflow 3.x put the plain filesystem store into maintenance mode (no Model Registry support),
# so tracking + registry both live in a local sqlite db, as MLflow itself now recommends for
# single-machine use: https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment(EXPERIMENT_NAME)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


# ---------------------------------------------------------------------------
# Run configurations. Hyperparameters genuinely differ between runs (not just
# the seed) and we compare two model families, as the assignment allows.
# ---------------------------------------------------------------------------
RUN_CONFIGS = [
    {
        "run_name": "logreg_c0.1",
        "model_family": "LogisticRegression",
        "params": {"C": 0.1, "penalty": "l2", "max_iter": 1000, "class_weight": "balanced"},
    },
    {
        "run_name": "logreg_c1.0",
        "model_family": "LogisticRegression",
        "params": {"C": 1.0, "penalty": "l2", "max_iter": 1000, "class_weight": "balanced"},
    },
    {
        "run_name": "rf_shallow_100trees",
        "model_family": "RandomForest",
        "params": {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced", "random_state": 42},
    },
    {
        "run_name": "rf_deep_300trees",
        "model_family": "RandomForest",
        "params": {"n_estimators": 300, "max_depth": 12, "class_weight": "balanced", "random_state": 42},
    },
]


def make_model(config):
    if config["model_family"] == "LogisticRegression":
        return LogisticRegression(**config["params"])
    if config["model_family"] == "RandomForest":
        return RandomForestClassifier(**config["params"])
    raise ValueError(config["model_family"])


def run_one(config, x_train, x_test, y_train, y_test):
    with mlflow.start_run(run_name=config["run_name"]) as run:
        model = make_model(config)
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_param("model_family", config["model_family"])
        for k, v in config["params"].items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)

        # ---- artifacts: confusion matrix + ROC curve ----
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["No churn", "Churn"]).plot(ax=ax_cm)
        ax_cm.set_title(f"Confusion matrix - {config['run_name']}")
        cm_path = REPORTS_DIR / f"confusion_matrix_{config['run_name']}.png"
        fig_cm.tight_layout()
        fig_cm.savefig(cm_path, dpi=120)
        plt.close(fig_cm)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc)
        ax_roc.set_title(f"ROC curve - {config['run_name']}")
        roc_path = REPORTS_DIR / f"roc_curve_{config['run_name']}.png"
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path, dpi=120)
        plt.close(fig_roc)
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # ---- model artifact ----
        # RandomForest's tree storage is flagged by skops's (new, stricter) untrusted-types
        # audit because it holds raw node indices; we trust it here since we trained it
        # ourselves in this run, not loaded it from an external source.
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"[{config['run_name']}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return {
            "run_id": run.info.run_id,
            "run_name": config["run_name"],
            "model_family": config["model_family"],
            "params": config["params"],
            **metrics,
        }


def main():
    df = load_raw()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    results = [run_one(cfg, x_train, x_test, y_train, y_test) for cfg in RUN_CONFIGS]

    # rank by F1 first (churn is imbalanced ~26% positive, so accuracy alone is misleading),
    # tie-break on ROC-AUC.
    results_sorted = sorted(results, key=lambda r: (r["f1"], r["roc_auc"]), reverse=True)
    best = results_sorted[0]

    comparison_path = REPORTS_DIR / "run_comparison.json"
    comparison_path.write_text(json.dumps(results_sorted, indent=2))

    # markdown table for the README
    header = "| run | model family | key hyperparams | accuracy | precision | recall | f1 | roc_auc |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = []
    for r in results_sorted:
        hp = ", ".join(f"{k}={v}" for k, v in r["params"].items() if k not in ("class_weight", "random_state"))
        rows.append(
            f"| {r['run_name']} | {r['model_family']} | {hp} | {r['accuracy']:.4f} | {r['precision']:.4f} "
            f"| {r['recall']:.4f} | **{r['f1']:.4f}** | {r['roc_auc']:.4f} |"
        )
    table_md = "\n".join([header, sep] + rows)
    (REPORTS_DIR / "run_comparison.md").write_text(table_md + "\n")
    print("\n" + table_md)
    print(f"\nBest run by F1: {best['run_name']} (f1={best['f1']:.4f}, roc_auc={best['roc_auc']:.4f})")

    # ---- register the best model in the MLflow Model Registry ----
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered {MODEL_NAME} version {mv.version} from run {best['run_id']}")

    # stage transitions: Staging -> Production (assignment asks for >= 2 stages)
    client.transition_model_version_stage(name=MODEL_NAME, version=mv.version, stage="Staging")
    print(f"{MODEL_NAME} v{mv.version} -> Staging")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
    )
    print(f"{MODEL_NAME} v{mv.version} -> Production")

    (REPORTS_DIR / "registered_model.json").write_text(
        json.dumps(
            {"model_name": MODEL_NAME, "version": mv.version, "source_run_id": best["run_id"], "stage": "Production"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
