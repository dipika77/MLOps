# W17 — MLOps Assignment

Two independent tracks, each applying **uv** (environment), **MLflow**
(experiment tracking), and **Evidently** (monitoring/regression), plus an
optional **Airflow** DAG, to a different system:

- **`track_a_churn/`** — Data Science MLOps: a churn-prediction pipeline on the
  real IBM Telco Customer Churn dataset. 4 trained model configs, MLflow model
  registry with Staging→Production promotion, a FastAPI serving endpoint, and
  Evidently data-drift + target-drift monitoring with 2 custom metrics.
- **`track_b_assistant/`** — Agentic AI MLOps: the W15/W16 Researcher/Verifier
  RAG assistant. MLflow tracks 3 prompt/config versions (each fixing a specific
  traced failure), Evidently runs LLM-as-a-judge regression testing
  (`CorrectnessLLMEval` + `DeclineLLMEval`) against a golden set.

Each track has its own `README.md` with the required write-up sections
(Environment & Reproducibility, Experiment Tracking Strategy, Monitoring/
Regression Strategy, Orchestration) and its own `pyproject.toml` + `uv.lock` —
read those first, they're the actual deliverable.

## Real results, at a glance

**Track A** — best model `logreg_c0.1` chosen over the deep random forest despite
a lower raw accuracy, because it had a meaningfully better F1/ROC-AUC trade-off
on the minority (churn) class — see `track_a_churn/README.md` 9.b for the full
run-comparison table. Drift check on an injected-drift current window: 2/20
columns flagged (MonthlyCharges, Contract), dataset-level flag stayed *below*
threshold even though the injected shift was real and visible per-column — see
9.c for why that's not a contradiction.

**Track B** — v1 (baseline prompts) answers a cross-source pricing question
confidently and **wrong** (silently picks one of two conflicting sources). v2
(stricter verifier only) makes the *check* stricter but the researcher can't act
on it, so it loops to `max_iterations` — more expensive and still wrong. v3
(verifier fix + researcher targets the verifier's feedback) is the only version
that resolves it: 2 iterations, states both figures, flags the conflict. Evidently
regression pass rate backs this up: v1 80%, v2 70%, v3 90%. Full trace-by-trace
narrative in `track_b_assistant/README.md` 9.b/9.c, including a real code bug
found and fixed along the way and a documented judge-limitation sanity check.

## Running locally

```bash
cd track_a_churn
uv sync
uv run python -m src.train        # trains 4 configs, logs to MLflow, registers best model
uv run python -m src.monitoring   # drift reports
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs/registry

cd ../track_b_assistant
uv sync
uv run pytest -q                            # 19 passed
uv run python -m mlops.run_experiments      # prompt-version MLflow runs
uv run python -m mlops.regression_tests     # Evidently regression suite
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Both tracks' `mlflow.db` (sqlite) and `reports/` (generated MLflow/Evidently
artifacts, including HTML reports and MLflow UI screenshots) are included so
results can be reviewed without re-running anything, alongside the raw code and
data needed to regenerate them from scratch.

## Publishing to GitHub

I don't have GitHub credentials in this environment, so I can't push a repo or
hand you a live link directly — everything below is ready to push as-is. Pick
whichever structure you prefer; the assignment allows either:

**Option 1 — one repo, two branches** (simplest if you want a single link to
submit for both tracks):

```bash
cd W17_MLOps
git init
git add track_a_churn README.md
git commit -m "Track A: churn MLOps pipeline (uv + MLflow + Evidently)"
git branch -M main
git remote add origin <your-empty-github-repo-url>
git push -u origin main

git checkout -b track-b
git add track_b_assistant
git commit -m "Track B: agentic assistant MLOps (uv + MLflow + Evidently)"
git push -u origin track-b
```

**Option 2 — two separate repos** (cleaner if the assignment wants two
independent submission links):

```bash
cd W17_MLOps/track_a_churn
git init && git add -A && git commit -m "Track A: churn MLOps pipeline"
git branch -M main
git remote add origin <repo-a-url>
git push -u origin main

cd ../track_b_assistant
git init && git add -A && git commit -m "Track B: agentic assistant MLOps"
git branch -M main
git remote add origin <repo-b-url>
git push -u origin main
```

Before pushing, note: each track's `.venv/` was excluded from this delivered zip
(it's fully reproducible via `uv sync` from the committed `uv.lock`, and
including it would make the repo enormous) — `uv sync` regenerates it in
seconds after cloning. `mlruns/` and `mlflow.db` ARE included so a reviewer can
open the MLflow UI immediately without re-running training; add a `.gitignore`
for `.venv/`, `__pycache__/`, and `.pytest_cache/` if you want a tidier repo,
but there's no need to gitignore `mlruns/`/`mlflow.db`/`reports/` — those are
the evidence of real runs the assignment asks for.
