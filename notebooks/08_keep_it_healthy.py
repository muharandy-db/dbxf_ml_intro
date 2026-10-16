# Databricks notebook source

# MAGIC %md
# MAGIC # Module 08 — Keep It Healthy (Scheduled Retraining vs Drift Detection)
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Distinguish **data drift** from **concept drift**, and explain why both break models silently
# MAGIC > - Compute the **Population Stability Index (PSI)** by hand and read it
# MAGIC > - Attach a **Data Quality Monitor** (data profiling) to an inference table, and wire drift to a retraining decision
# MAGIC >
# MAGIC > ⏱ **Time:** 40 minutes  ·  Answers production question **#7**: *Is the model still accurate? When should we retrain?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC "Retrain the model on the first of every month." This rule sounds responsible, but it fails in both directions. In a calm month it burns compute retraining a model that was fine. In a turbulent month, when a competitor launches a digital bank in week one, it leaves a broken model in production for three more weeks. Either way, nobody knows the model is drifting, because nobody is measuring it.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: check the tyre tread, don't just follow the calendar
# MAGIC
# MAGIC Replacing your car's tyres every six months regardless of wear is wasteful on a car that sits in the garage, and dangerous on a car that does a thousand kilometres a week. Sensible drivers **check the tread**. Monitoring is checking the tread, and retraining becomes a response to evidence instead of a date.
# MAGIC
# MAGIC There are two different things to check, and they fail differently:
# MAGIC
# MAGIC - **Data drift**: the *inputs* change. After a mobile-app relaunch, `digital_logins_30d` doubles across the board. The model is now being asked about customers unlike the ones it learned from. You can detect this **immediately**, without labels, by comparing feature distributions.
# MAGIC - **Concept drift**: the *relationship* between inputs and outcome changes. Heavy app users used to be loyal; now a digital-only competitor poaches exactly them. The inputs might look similar, but the right answer has moved. You can only confirm this **once labels arrive**, by measuring the model's actual performance.
# MAGIC
# MAGIC Our `customers_next_quarter` data contains both, on purpose: a mobile-app relaunch, a competitor cutting rates, and a service outage.
# MAGIC
# MAGIC ### The Population Stability Index
# MAGIC
# MAGIC PSI measures how much a feature's distribution has moved between a **baseline** (training data) and a **current** window:
# MAGIC
# MAGIC $$\text{PSI} \;=\; \sum_{b=1}^{B} \big(a_b - e_b\big)\,\ln\!\frac{a_b}{e_b}$$
# MAGIC
# MAGIC Let me unpack every piece. Cut the baseline values of a feature into $B$ bins, usually 10 bins with equal counts (deciles). $e_b$ is the **expected** share of rows in bin $b$, measured on the baseline, so about 0.10 for each decile. $a_b$ is the **actual** share of current rows that fall into the same bin. For each bin, $(a_b - e_b)$ asks *how much did the share change*, and $\ln(a_b/e_b)$ asks *by what factor*. Both have the same sign, so every term is positive. Big moves in either direction add up, and identical distributions give exactly 0.
# MAGIC
# MAGIC The industry rule of thumb, which comes from credit scoring, is a fitting origin for a bank:
# MAGIC
# MAGIC | PSI | Reading |
# MAGIC |---|---|
# MAGIC | < 0.10 | Stable, no action |
# MAGIC | 0.10 – 0.25 | Moderate shift, investigate |
# MAGIC | > 0.25 | Major shift, the model is seeing a different population |
# MAGIC
# MAGIC The one caveat is that PSI is about *inputs*, so it detects data drift only. For concept drift you need labels and a performance metric. That's why a good monitor tracks both.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a notebook `08_keep_it_healthy`.
# MAGIC
# MAGIC ### Cell 0 — Libraries

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.1" scikit-learn==1.6.1 "databricks-sdk>=0.70"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1 — Config

# COMMAND ----------

import re, mlflow
import numpy as np, pandas as pd
from pyspark.sql import functions as F
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.churn_model"

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"

feats = spark.table("customer_features").toPandas()
june = feats[feats["snapshot_date"].astype(str) == "2026-06-30"]
sept = feats[feats["snapshot_date"].astype(str) == "2026-09-30"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — PSI, by hand

# COMMAND ----------

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected, actual = expected.dropna(), actual.dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))   # baseline deciles
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)                  # avoid log(0)
    return float(np.sum((a - e) * np.log(a / e)))

psi_df = (pd.DataFrame({"feature": NUMERIC,
                        "psi": [psi(june[c], sept[c]) for c in NUMERIC]})
          .sort_values("psi", ascending=False))
psi_df["status"] = pd.cut(psi_df["psi"], [-1, 0.10, 0.25, np.inf],
                          labels=["✅ stable", "⚠️ investigate", "🚨 major shift"])
display(psi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC > 🤔 **Predict first:** knowing the story (app relaunch, competitor rate cut, outage), which three features will top the list?
# MAGIC
# MAGIC **What to point out**
# MAGIC - That's the formula above, line for line: baseline deciles for the bins, shares $e_b$ and $a_b$, then the sum.
# MAGIC - 🚨 `digital_logins_30d` and `mobile_share_90d` (the app relaunch) and `complaints_90d` (the outage) show **major shifts**. ⚠️ `balance` (the competitor's rate cut) is a moderate shift worth investigating. All of this is visible **before a single churn label exists**.
# MAGIC - Profile attributes such as `age` and `num_products` stay stable, because they're the same customers.
# MAGIC
# MAGIC ### Cell 3 — Concept drift: once the labels arrive

# COMMAND ----------

from sklearn.metrics import roc_auc_score

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
report = pd.DataFrame([{
    "window": name,
    "actual_churn_rate": df[LABEL].mean(),
    "predicted_churn_rate": (model.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1] >= 0.5).mean(),
    "auc": roc_auc_score(df[LABEL], model.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]),
} for name, df in [("2026-06 (training period)", june), ("2026-09 (next quarter)", sept)]])
display(report)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The actual churn rate jumped, but the model's predicted rate barely moved. **The model hasn't noticed the world changed.**
# MAGIC - AUC drops too (≈0.89 → ≈0.83). The relationship the model learned (for example, "app users are loyal") no longer holds. That's **concept drift**, and it was confirmable only now that September's outcomes are known.
# MAGIC - The June AUC is a little flattering because June includes the training rows. The *drop* is the signal, not the absolute number.
# MAGIC - Note the gap in time. Data drift (Cell 2) was visible on 30 September; concept drift needed outcomes from the *following* quarter. Detecting data drift early buys you time.
# MAGIC
# MAGIC ### Cell 4 — Build an inference table

# COMMAND ----------

champion_version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version

def score_batches(batches):
    for pdf in batches:
        pdf["p_churn"] = model.predict_proba(pdf[NUMERIC + CATEGORICAL])[:, 1]
        pdf["prediction"] = (pdf["p_churn"] >= 0.3).astype("int32")
        pdf[LABEL] = pdf[LABEL].astype("int32")
        yield pdf[["customer_id", "snapshot_date", *NUMERIC, *CATEGORICAL, "p_churn", "prediction", LABEL]]

from pyspark.sql.types import DoubleType, IntegerType
out_schema = (spark.table("customer_features").select("customer_id", "snapshot_date", *NUMERIC, *CATEGORICAL).schema
              .add("p_churn", DoubleType()).add("prediction", IntegerType()).add(LABEL, IntegerType()))

inference_log = (spark.table("customer_features")
    .mapInPandas(score_batches, out_schema)
    .withColumn("model_version", F.lit(champion_version))
    .withColumn("scored_at", F.col("snapshot_date").cast("timestamp")))

inference_log.write.mode("overwrite").option("overwriteSchema", True).saveAsTable("churn_inference_log")
inference_log.filter("snapshot_date = '2026-06-30'").write.mode("overwrite") \
    .option("overwriteSchema", True).saveAsTable("churn_inference_baseline")
spark.sql("ALTER TABLE churn_inference_log SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - An **inference table** is a log of *every* prediction: inputs, output, model version, timestamp, and (later) the true label. It's the raw material for monitoring.
# MAGIC - Model Serving can write this table automatically for real-time endpoints. Here we build it from batch scoring.
# MAGIC - `churn_inference_baseline` holds the training-period predictions. The monitor will compare every later window against it.
# MAGIC - Change Data Feed lets the monitor process only *new* rows on each refresh.
# MAGIC
# MAGIC ### Cell 5 — Attach a monitor

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import (Monitor, DataProfilingConfig, InferenceLogConfig,
                                                InferenceProblemType, AggregationGranularity)
w = WorkspaceClient()
table = w.tables.get(f"{CATALOG}.{SCHEMA}.churn_inference_log")
schema_id = w.schemas.get(f"{CATALOG}.{SCHEMA}").schema_id

try:
    w.data_quality.get_monitor(object_type="table", object_id=table.table_id)
    print("Monitor already exists")
except Exception:
    w.data_quality.create_monitor(monitor=Monitor(
        object_type="table", object_id=table.table_id,
        data_profiling_config=DataProfilingConfig(
            output_schema_id=schema_id,
            assets_dir=f"/Workspace/Users/{user}/monitoring/churn_inference_log",
            baseline_table_name=f"{CATALOG}.{SCHEMA}.churn_inference_baseline",
            inference_log=InferenceLogConfig(
                problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_CLASSIFICATION,
                timestamp_column="scored_at",
                granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],
                prediction_column="prediction",
                label_column=LABEL,
                model_id_column="model_version"))))
    print("Monitor created")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - One API call. The monitor will compute, per time window and per model version:
# MAGIC   - a **profile**: counts, nulls, distributions, and model quality (precision, recall, …) when labels exist
# MAGIC   - **drift**: each window against the baseline, for every column
# MAGIC - It produces two Delta tables (`…_profile_metrics`, `…_drift_metrics`) and an **auto-generated dashboard**. Everything is queryable with SQL.
# MAGIC - In the UI, open the `churn_inference_log` table → **Quality** tab to see the same configuration.
# MAGIC
# MAGIC ### Cell 6 — Wait for the first refresh (≈5–10 min)

# COMMAND ----------

import time
from databricks.sdk.service.dataquality import Refresh

def monitor_status():
    return w.data_quality.get_monitor(object_type="table", object_id=table.table_id).data_profiling_config.status.name

while monitor_status() == "DATA_PROFILING_STATUS_PENDING":       # a brand-new monitor initialises first
    print(f"{time.strftime('%H:%M:%S')}  monitor initialising..."); time.sleep(15)

def latest_refresh():
    refreshes = list(w.data_quality.list_refresh(object_type="table", object_id=table.table_id))
    return max(refreshes, key=lambda r: r.start_time_ms or 0) if refreshes else None

if latest_refresh() is None:
    w.data_quality.create_refresh(object_type="table", object_id=table.table_id,
                                  refresh=Refresh(object_type="table", object_id=table.table_id))
while (r := latest_refresh()).state.name in ("MONITOR_REFRESH_STATE_PENDING", "MONITOR_REFRESH_STATE_RUNNING"):
    print(f"{time.strftime('%H:%M:%S')}  {r.state.name}"); time.sleep(30)
print(f"Refresh finished: {r.state.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - A new monitor first initialises (status `PENDING`), then runs its first **refresh** on serverless compute in the background. Use the wait to discuss: *"What threshold would trigger action at our bank? Who gets the alert?"*
# MAGIC - In production you'd **schedule** refreshes (for example, daily) instead of waiting in a notebook.
# MAGIC
# MAGIC ### Cell 7 — Read the drift metrics

# COMMAND ----------

cfg = w.data_quality.get_monitor(object_type="table", object_id=table.table_id).data_profiling_config
drift = spark.table(cfg.drift_metrics_table_name)

wanted = ["window", "column_name", "drift_type", "js_distance", "population_stability_index",
          "ks_test", "wasserstein_distance"]
display(drift.select(*[c for c in wanted if c in drift.columns])
             .filter("drift_type = 'BASELINE' AND column_name NOT LIKE ':%'")
             .orderBy(F.desc("window"), F.desc_nulls_last("population_stability_index")))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The September window against the baseline shows the same culprits our hand-rolled PSI found, now **computed for every column, every window, automatically**.
# MAGIC - The monitor computes the **same PSI** we wrote by hand, plus other distances: `ks_test` and `wasserstein_distance` for numeric columns, and `js_distance` (Jensen–Shannon, 0 = identical, 1 = completely different) for categorical ones.
# MAGIC - `prediction` itself drifts strongly. The model now flags far more customers, which is **prediction drift**: a cheap, label-free alarm for any model.
# MAGIC - Open the **dashboard** linked from the table's Quality tab. It shows the same numbers as charts, ready to share with the risk team.
# MAGIC
# MAGIC ### Cell 8 — From drift to a retraining decision

# COMMAND ----------

PSI_THRESHOLD = 0.25
RETRAIN_JOB_ID = None          # 👈 set this to the job ID you deploy in Module 09

drifted = psi_df[psi_df["psi"] > PSI_THRESHOLD]["feature"].tolist()
print(f"Features with major drift: {drifted}")

if drifted and RETRAIN_JOB_ID:
    run = w.jobs.run_now(job_id=RETRAIN_JOB_ID)
    print(f"🔁 Retraining triggered: run {run.run_id}")
elif drifted:
    print("Drift detected: retraining would be triggered here (set RETRAIN_JOB_ID after Module 09)")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - This is the core of the module: **retrain because the data says so**, not because it's the first of the month.
# MAGIC - In production the trigger is usually a **SQL alert** on the drift metrics table (for example, `max(js_distance) > 0.2`) that notifies a team or starts a job. No notebook needs to be open.
# MAGIC - Retraining isn't automatically the answer to concept drift. If the business changed, you might need **new features** (for example, "uses competitor app"). Monitoring tells you *that* something changed; humans decide *what to do*.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC The monitor also tracks **model quality** per window when labels are present. Query the profile metrics table and compare the model's performance between June and September.
# MAGIC
# MAGIC
# MAGIC **Solution**

# COMMAND ----------

profile = spark.table(cfg.profile_metrics_table_name)
quality_cols = [c for c in profile.columns if c in ("window", "model_version", "count",
                "accuracy_score", "precision", "recall", "f1_score")]
display(profile.filter("column_name = ':table'").select(*quality_cols).orderBy("window"))

# COMMAND ----------

# MAGIC %md
# MAGIC `column_name = ':table'` selects table-level metrics (as opposed to per-column statistics). The September window shows lower quality. That's concept drift measured by the platform instead of by a hand-written evaluation.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks |
# MAGIC |---|---|---|
# MAGIC | When to retrain | Every month, by calendar | When drift or quality metrics cross a threshold |
# MAGIC | Detecting data drift | Not done | Automatic per-column drift vs a baseline |
# MAGIC | Detecting concept drift | "Complaints from the business" | Model quality per window once labels land |
# MAGIC | Visibility | None | Metric tables + auto-generated dashboard |
# MAGIC | Action | Manual | SQL alert → notification or retraining job |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *In September, PSI for `digital_logins_30d` is 0.6 but we don't have churn labels yet. Can we conclude the model is now wrong?*
# MAGIC >
# MAGIC > Not yet. High PSI says the **inputs** shifted (data drift), so the model is extrapolating and deserves scrutiny. Whether its predictions are wrong (concept drift) is only proven once labels arrive and performance is measured. Data drift is the early-warning signal; quality metrics are the confirmation.
# MAGIC
# MAGIC > *Why is a monthly retraining schedule both too expensive and too slow?*
# MAGIC >
# MAGIC > In stable months it retrains a model that didn't need it, which wastes compute and adds release risk. When the world changes mid-month, it leaves a degraded model in production until the next scheduled date. Drift-triggered retraining spends effort only when, and as soon as, the evidence calls for it.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 07 — Ship It with Governance | Next: Module 09 — From Notebook to Production →
