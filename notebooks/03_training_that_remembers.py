# Databricks notebook source

# MAGIC %md
# MAGIC # Module 03 — Training That Remembers (MLflow Tracking)
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Explain MLflow's building blocks: **experiment → run → parameters, metrics, artifacts, model, dataset**
# MAGIC > - Turn on **autologging** and add your own business metrics on top
# MAGIC > - Tune hyperparameters with **Optuna** and see every trial as a **nested run**
# MAGIC >
# MAGIC > ⏱ **Time:** 45 minutes  ·  Answers production question **#2**: *Which parameters, code, and metrics produced `final_v3`? What did we try before it?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC Six weeks ago someone tried a random forest with 300 trees and got an AUC of 0.79, or maybe it was 0.77, and that might have been on the old data. The result lived in a notebook cell output that has since been overwritten. The "experiment log" is a spreadsheet updated whenever someone remembers. When the auditor asks *why* the bank chose this model, the honest answer is a shrug.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: a lab notebook that writes itself
# MAGIC
# MAGIC A chemist doesn't just mix things until something works. Every attempt goes into a lab notebook: what went in, under what conditions, what came out. Science is *reproducible* because of that notebook, not because chemists have good memories.
# MAGIC
# MAGIC **MLflow Tracking** is that lab notebook for machine learning, and on Databricks it is built in, with no server to install. The vocabulary is small:
# MAGIC
# MAGIC - An **experiment** is a project folder, for example *"churn model."*
# MAGIC - A **run** is one attempt: one call to `fit`.
# MAGIC - Each run records **parameters** (the inputs you chose, like `max_depth=6`), **metrics** (the results, like `val_auc=0.78`), **artifacts** (any file: a plot, a CSV, a report), the **model** itself, ready to load, and the **dataset**: which table and which Delta version it was trained on.
# MAGIC
# MAGIC The catch is obvious once said aloud: **a lab notebook only helps if every experiment gets written in it.** People forget. So MLflow offers **autologging**. One line, and every `fit` call records its parameters, training metrics, and model automatically. Tracking stops being a discipline and becomes the default. You then add what only *you* know, such as the business threshold and the metric the retention team actually cares about.
# MAGIC
# MAGIC > ℹ️ **"Tracking" vs "Tracing".** This module is about MLflow **Tracking**: recording training runs. MLflow also has **Tracing**, which records step-by-step execution of GenAI apps and agents (each LLM call, tool call, and retrieval). Same platform, different job. There's a short preview in Module 10.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a notebook `03_training_that_remembers`.
# MAGIC
# MAGIC ### Cell 0 — Libraries

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" scikit-learn==1.6.1 optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - On serverless, each notebook has its own Python environment. `%pip` installs into *this notebook only*, and `restartPython()` loads the fresh versions.
# MAGIC - We **pin** `scikit-learn==1.6.1` in every module. A pickled scikit-learn model generally only loads in the version that saved it, so training, scoring, and monitoring must agree. MLflow records the exact versions in each logged model's `requirements.txt`, which is how you find out which version a model needs.
# MAGIC - In real projects, put dependencies in the notebook's **Environment** side panel (or in the job definition, Module 09) instead of `%pip` cells.
# MAGIC
# MAGIC ### Cell 1 — Config (now with MLflow)

# COMMAND ----------

import re, mlflow
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

mlflow.set_registry_uri("databricks-uc")                 # models will be registered in Unity Catalog
mlflow.set_experiment(f"/Users/{user}/churn_workshop")    # one experiment for the whole day
MODEL_NAME = f"{CATALOG}.{SCHEMA}.churn_model"

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `set_experiment` creates the experiment if it doesn't exist. Every module from here on logs into the **same** experiment, so the whole day's work sits in one place.
# MAGIC - The first run prints two messages, and both are expected. *"Experiment ... does not exist. Creating a new experiment"* is `set_experiment` doing its job. The note about storing traces in Unity Catalog is about **MLflow Tracing** for GenAI apps (Module 10). Today we log runs and models, not traces, so you can ignore it.
# MAGIC - `databricks-uc` means registered models live in **Unity Catalog** next to the tables, with the same permissions model (Module 07).
# MAGIC
# MAGIC ### Cell 2 — Load the training snapshot

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"

train_sdf = spark.table("customer_features").filter("snapshot_date = '2026-06-30'")
pdf = train_sdf.orderBy("customer_id").toPandas()

X = pd.get_dummies(pdf[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=int)
y = pdf[LABEL]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

table_version = spark.sql("DESCRIBE HISTORY customer_features LIMIT 1").first()["version"]
dataset = mlflow.data.from_spark(train_sdf, table_name=f"{CATALOG}.{SCHEMA}.customer_features",
                                 version=str(table_version))
print(f"train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,} | table version {table_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Spark filters the table and `toPandas()` brings the 10k rows to the driver. Small data, so **scikit-learn is the right tool**; Module 04 discusses when it isn't.
# MAGIC - The data is split three ways: **train** to fit, **validation** to compare and tune, and **test** touched once at the very end (Module 07). Tuning on the test set is leakage in disguise.
# MAGIC - `orderBy("customer_id")` makes the split reproducible. Spark doesn't guarantee row order, so without it, the "same" split could differ between runs.
# MAGIC - `mlflow.data.from_spark` captures **which table and which Delta version** we trained on. That links directly back to Module 02.
# MAGIC - `pd.get_dummies` here is still a manual step. Remember this when you get to Module 04.
# MAGIC
# MAGIC ### Cell 3 — Autolog + a reusable training helper

# COMMAND ----------

from sklearn.metrics import roc_auc_score, recall_score, precision_score, ConfusionMatrixDisplay

mlflow.sklearn.autolog(log_post_training_metrics=False)
THRESHOLD = 0.3     # the business decision from Module 01

def train_and_log(model, run_name):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("module", "03")
        mlflow.log_input(dataset, context="training")
        model.fit(X_train, y_train)                              # autolog records params + model

        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        mlflow.log_param("decision_threshold", THRESHOLD)
        mlflow.log_metrics({"val_auc": roc_auc_score(y_val, proba),
                            "val_recall": recall_score(y_val, pred),
                            "val_precision": precision_score(y_val, pred)})
        fig = ConfusionMatrixDisplay.from_predictions(y_val, pred).figure_
        mlflow.log_figure(fig, "confusion_matrix.png")
        print(f"{run_name}: val_auc={roc_auc_score(y_val, proba):.3f}")
        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `mlflow.sklearn.autolog()` is the one line that makes tracking the default. Every `fit` now logs its hyperparameters, training metrics, and the fitted model.
# MAGIC - On top of that we log what autolog can't know: the **business threshold**, recall at that threshold, and a confusion-matrix image as an **artifact**.
# MAGIC - `log_input` attaches the dataset (table plus version) to the run, which gives you **data lineage per run**.
# MAGIC
# MAGIC ### Cell 4 — Two candidate models

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

train_and_log(RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42), "random_forest")
train_and_log(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42), "hist_gbm")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Both models handle missing values natively, so we didn't need to impute `credit_score` here.
# MAGIC - Click the **flask icon** (Experiments) in the right sidebar, or open the experiment from the run link printed under the cell. Each run shows its parameters, metrics, the confusion matrix artifact, the logged model, and the **dataset** with its table version.
# MAGIC - Select both runs → **Compare**. Side-by-side parameters and metrics, no spreadsheet.
# MAGIC
# MAGIC ### Cell 5 — Query your experiments like data

# COMMAND ----------

runs = mlflow.search_runs(
    filter_string="tags.module = '03'",
    order_by=["metrics.val_auc DESC"])
display(runs[["tags.mlflow.runName", "metrics.val_auc", "metrics.val_recall",
              "metrics.val_precision", "params.decision_threshold"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Experiments are queryable, so *"show me the best run this month"* is one function call. That's how automated retraining picks a winner (Module 09).
# MAGIC - `search_runs` returns a pandas DataFrame, which you can chart, filter, or export.
# MAGIC
# MAGIC ### Cell 6 — Hyperparameter tuning, every trial recorded

# COMMAND ----------

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.sklearn.autolog(disable=True)          # log trials ourselves, lightweight

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "max_iter": trial.suggest_int("max_iter", 50, 400),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
    }
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        model = HistGradientBoostingClassifier(**params, random_state=42).fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", auc)
    return auc

with mlflow.start_run(run_name="hist_gbm_tuning") as parent:
    mlflow.set_tag("module", "03")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("val_auc", study.best_value)

print(f"Best val_auc {study.best_value:.3f} with {study.best_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Optuna proposes hyperparameters intelligently: each trial learns from the previous ones instead of searching blindly over a grid.
# MAGIC - `nested=True` puts the 20 trials **under one parent run**. In the UI, expand the parent to see them, then open **Chart view → Parallel coordinates** to see *which* settings drive AUC.
# MAGIC - Tuning gains on clean tabular data are usually modest. The real win is that the search is **documented**: anyone can see what was tried and why the winner won.
# MAGIC - To run trials in parallel across a cluster, Optuna can use a shared storage backend. That's a scale-up step, not something you need today.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC Train a third candidate, a random forest with `n_estimators=400, max_depth=12`, using `train_and_log`, then re-run Cell 5. Does a bigger forest beat the gradient-boosting model? Open the two runs in **Compare**.
# MAGIC
# MAGIC
# MAGIC **Solution**

# COMMAND ----------

mlflow.sklearn.autolog(log_post_training_metrics=False)   # re-enable after the tuning cell
train_and_log(RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42), "random_forest_big")

# COMMAND ----------

# MAGIC %md
# MAGIC Usually it doesn't. Deeper trees overfit a small dataset, so the gain on validation is small or negative. That's a finding worth having in the log: the next person won't waste an afternoon re-trying it.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks |
# MAGIC |---|---|---|
# MAGIC | Recording an experiment | Copy numbers into a spreadsheet (sometimes) | Autologged on every `fit` |
# MAGIC | What's recorded | Whatever someone remembered | Params, metrics, artifacts, model, **dataset + version**, code, user, time |
# MAGIC | Comparing models | Scroll through notebook outputs | Compare view, charts, `search_runs()` |
# MAGIC | Tuning | Nested `for` loops, results in `print()` | Optuna + nested runs, every trial kept |
# MAGIC | Audit question *"why this model?"* | A shrug | A link to the run |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *Autologging already records training metrics. Why did we still log `val_auc` and `val_recall` ourselves?*
# MAGIC >
# MAGIC > Autolog sees only `fit`. It doesn't know which data is your validation set, what threshold the business chose, or which metric matters. Those decisions are yours, so you log them explicitly, and they become searchable alongside everything autolog captured.
# MAGIC
# MAGIC > *Why do we tune on the validation set and keep the test set untouched?*
# MAGIC >
# MAGIC > Every time you pick hyperparameters based on a dataset's score, a little information from that dataset leaks into your choices. If you tuned on the test set, its score would be optimistic. The untouched test set gives the one honest estimate of future performance.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 02 — Data in the Lakehouse | Next: Module 04 — Pipelines, Not Scripts →
