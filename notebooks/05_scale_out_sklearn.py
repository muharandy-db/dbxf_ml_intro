# Databricks notebook source

# MAGIC %md
# MAGIC # Module 05 — Bring Your scikit-learn, Scale It Out
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Know what "distributed scikit-learn" *can* and *cannot* mean
# MAGIC > - Train **one model per region in parallel** with `groupBy().applyInPandas`
# MAGIC > - Score a table **in parallel** with `mlflow.pyfunc.spark_udf` and `mapInPandas`
# MAGIC >
# MAGIC > ⏱ **Time:** 35 minutes  ·  Answers production question **#4**: *We love scikit-learn. Do we have to rewrite everything to scale?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC The team has years of scikit-learn code and skills. When the business asks for *one model per branch region* or *score all 20 million customers tonight*, the legacy answer is a `for` loop on a single server. The loop runs region after region or chunk after chunk, through the night, and if it fails at 4 a.m. it starts again from the beginning.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: same recipe, many kitchens
# MAGIC
# MAGIC First, the honest part. **A single scikit-learn `fit` cannot be split across machines.** scikit-learn's algorithms assume all the data is in one process's memory. That's what Spark ML's re-designed algorithms are for (Module 04).
# MAGIC
# MAGIC But most "we need to scale scikit-learn" problems aren't one giant `fit`. They are **many independent pieces of work**, and Spark is superb at running independent work in parallel. Two patterns cover almost every case:
# MAGIC
# MAGIC **Pattern A: many models (`applyInPandas`).** You want one model per region, product, or branch. Spark splits the table by a key, hands each group to a worker as an ordinary **pandas DataFrame**, runs *your* function on it, and stitches the results back together. Five regions means five kitchens, each cooking the same recipe on its own ingredients at the same time.
# MAGIC
# MAGIC **Pattern B: one model, many rows (`spark_udf` / `mapInPandas`).** You trained one scikit-learn model on a sample that fits in memory, and now you need to score 20 million rows. Spark ships a copy of the model to every executor, and each executor scores its own partitions in batches. The model is small and the data is big, so you move the model to the data instead of the other way round.
# MAGIC
# MAGIC ### The `applyInPandas` contract
# MAGIC
# MAGIC Your function must follow a simple contract:
# MAGIC
# MAGIC ```text
# MAGIC def my_function(pdf: pandas.DataFrame) -> pandas.DataFrame
# MAGIC ```
# MAGIC
# MAGIC Reading it: `pdf` contains **all rows of one group**, for example every Medan customer, as plain pandas. Inside, you can use any Python library you like. You return a pandas DataFrame of whatever shape you want, as long as its columns match a **schema** you declare up front, such as `"branch_region string, val_auc double"`. Spark needs the schema in advance because it builds its plan before any data moves (lazy evaluation, Module 02).
# MAGIC
# MAGIC The one caveat is memory: one group must fit in one worker's memory. Group by region (2,000 rows) and you're fine. Group by "all customers" and you're back on a single machine.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a notebook `05_scale_out_sklearn`.
# MAGIC
# MAGIC ### Cell 0 — Libraries

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.14" scikit-learn==1.6.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1 — Config

# COMMAND ----------

import logging, re, mlflow
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
# Some workspaces (e.g. Free Edition) don't expose Git details to MLflow; hide that one harmless warning
logging.getLogger("mlflow.tracking.context.registry").addFilter(lambda r: "extraContext" not in r.getMessage())
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/churn_workshop")

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"
train_sdf = spark.table("customer_features").filter("snapshot_date = '2026-06-30'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — Pattern A: the per-region training function

# COMMAND ----------

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_region(pdf: pd.DataFrame) -> pd.DataFrame:
    """Runs on a worker. `pdf` = every customer of ONE region, as plain pandas."""
    region = pdf["branch_region"].iloc[0]
    X_tr, X_va, y_tr, y_va = train_test_split(
        pdf[NUMERIC], pdf[LABEL], test_size=0.25, stratify=pdf[LABEL], random_state=42)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_depth=3, random_state=42)),
    ]).fit(X_tr, y_tr)
    return pd.DataFrame({
        "branch_region": [region],
        "n_customers": [len(pdf)],
        "churn_rate": [pdf[LABEL].mean()],
        "val_auc": [roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])],
        "model_bytes": [pickle.dumps(model)],          # ship the fitted model back as bytes
    })

result_schema = ("branch_region string, n_customers long, churn_rate double, "
                 "val_auc double, model_bytes binary")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - This is **ordinary scikit-learn code**, the kind the team already writes. Nothing Spark-specific happens inside the function.
# MAGIC - It returns a one-row DataFrame per region: some metrics plus the fitted model **pickled into bytes**, so the driver can register it afterwards.
# MAGIC - `result_schema` is the contract: column names and types that Spark can plan around before anything runs.
# MAGIC
# MAGIC ### Cell 3 — Run all regions in parallel

# COMMAND ----------

results = train_sdf.groupBy("branch_region").applyInPandas(train_region, schema=result_schema)
display(results.drop("model_bytes").orderBy("branch_region"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - One line distributes the work. Spark **shuffles** rows so each region lands on one worker, and the five trainings run at the same time.
# MAGIC - With 5 regions the speed-up is modest. With 500 branches or 5,000 products, this is the difference between minutes and a night-long loop.
# MAGIC - Notice the **churn rate and AUC differ by region**. Compare them with the global pipeline from Module 04 (val AUC ≈ 0.85): here, **every regional model is a little worse**. Each one sees only a fifth of the data, and the global model already knows the region through one-hot encoding. Separate models pay off when segments behave *genuinely* differently and each segment has plenty of data. Now that's a cheap experiment to run instead of a debate.
# MAGIC
# MAGIC ### Cell 4 — Register each region's model in MLflow (from the driver)

# COMMAND ----------

region_rows = results.collect()                  # 5 small rows → safe to collect

with mlflow.start_run(run_name="per_region_models"):
    mlflow.set_tag("module", "05")
    for row in region_rows:
        with mlflow.start_run(run_name=f"region_{row.branch_region}", nested=True):
            mlflow.log_param("branch_region", row.branch_region)
            mlflow.log_metrics({"val_auc": row.val_auc, "n_customers": row.n_customers})
            mlflow.sklearn.log_model(pickle.loads(row.model_bytes), name=f"region_{row.branch_region}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Workers *train*; the driver *records*. That's a simple, robust split of responsibilities that works on any compute type.
# MAGIC - In the experiment you'll see one parent run with five nested runs, one per region, each with its own model artifact.
# MAGIC
# MAGIC ### Cell 5 — Pattern B: score a table in parallel with `spark_udf`

# COMMAND ----------

from pyspark.sql import functions as F

candidate = mlflow.search_runs(filter_string="tags.candidate = 'true'",
                               order_by=["start_time DESC"], max_results=1).iloc[0]
model_uri = f"runs:/{candidate.run_id}/sklearn_pipeline"     # the sklearn pipeline from Module 04

predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

scored = (spark.table("customer_features")
          .withColumn("prediction", predict_udf(F.struct(*NUMERIC, *CATEGORICAL))))
display(scored.groupBy("snapshot_date").agg(F.avg("prediction").alias("predicted_churn_rate")))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `spark_udf` turns *any* MLflow model into a Spark function. Spark sends the model to every executor and feeds it **batches of rows as pandas** through Apache Arrow.
# MAGIC - We pass **raw** columns. The pipeline from Module 04 does its own preprocessing, which is why we built it that way.
# MAGIC - Scoring 13k rows or 13 billion rows is the same code. Only the number of executors changes.
# MAGIC
# MAGIC ### Cell 6 — Pattern B, hand-rolled: probabilities with `mapInPandas`

# COMMAND ----------

sk_model = mlflow.sklearn.load_model(model_uri)

def score_batches(batches):
    for pdf in batches:                                   # each `pdf` is a chunk of rows
        pdf["p_churn"] = sk_model.predict_proba(pdf[NUMERIC + CATEGORICAL])[:, 1]
        yield pdf[["customer_id", "snapshot_date", "p_churn"]]

latest = spark.table("customer_features").filter("snapshot_date = '2026-09-30'")
call_list = latest.mapInPandas(
    score_batches, schema="customer_id string, snapshot_date date, p_churn double")
display(call_list.orderBy(F.desc("p_churn")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `spark_udf` calls `predict` and returns *labels*. For a call list we want *probabilities*, so we write the batch loop ourselves with `mapInPandas`.
# MAGIC - `sk_model` is captured by the function and shipped to every executor automatically.
# MAGIC - We score the **latest** snapshot (30 September). The top of this list is the retention team's **call list**: highest churn risk first.
# MAGIC
# MAGIC > 💡 **Other "many independent fits" you can parallelise the same way:** hyperparameter trials (Optuna with a shared storage, or Ray on Databricks) and cross-validation folds. The principle is the same each time: Spark distributes *the work*, and scikit-learn does each piece.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC The business also wants to know **how many customers in each region the model flags at a 0.3 threshold**. Add an `n_flagged` column to `train_region`'s output (computed on the validation split). Remember to update the schema.
# MAGIC
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Inside `train_region`, add a key to the returned dict:
# MAGIC
# MAGIC ```py
# MAGIC "n_flagged": [int((model.predict_proba(X_va)[:, 1] >= 0.3).sum())],
# MAGIC ```
# MAGIC
# MAGIC and append `n_flagged long` to `result_schema`:
# MAGIC
# MAGIC ```py
# MAGIC result_schema = ("branch_region string, n_customers long, churn_rate double, "
# MAGIC                  "val_auc double, model_bytes binary, n_flagged long")
# MAGIC ```
# MAGIC
# MAGIC Re-run Cells 2 and 3. If you forget the schema, Spark raises an error saying the returned columns don't match. That's the contract at work.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks |
# MAGIC |---|---|---|
# MAGIC | One model per segment | `for region in regions:` on one server, sequentially | `groupBy().applyInPandas()`, all segments in parallel |
# MAGIC | Scoring a huge table | Chunked loop, restarts on failure | `spark_udf` / `mapInPandas`, parallel and fault-tolerant |
# MAGIC | Code changes | Rewrite for "big data" | Keep your scikit-learn functions; Spark only distributes them |
# MAGIC | Tracking many models | Folder of pickles named by region | Nested MLflow runs, one per segment |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *A colleague says: "applyInPandas makes scikit-learn distributed, so we can train one model on 500 million rows." What's wrong with that?*
# MAGIC >
# MAGIC > `applyInPandas` distributes **groups**, and each group is still trained by a single scikit-learn process in one worker's memory. It parallelises *many* models. For *one* model on data too big for a machine, use Spark ML.
# MAGIC
# MAGIC > *Why can `spark_udf` score raw columns without any preprocessing code?*
# MAGIC >
# MAGIC > Because the logged model is the whole **pipeline** from Module 04: imputer, encoder, and classifier together. The preprocessing travels with the model to every executor.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 04 — Pipelines, Not Scripts | Next: Module 06 — Features as Shared Products →
