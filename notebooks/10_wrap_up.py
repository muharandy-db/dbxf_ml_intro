# Databricks notebook source

# MAGIC %md
# MAGIC # Module 10 — Wrap-Up: The Whole Game, the Lakehouse Way
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Revisit the eight production questions from Module 01 and answer each in one sentence
# MAGIC > - See the complete lifecycle on one page
# MAGIC > - Know where to go next: AutoML, the Databricks Assistant, and MLflow Tracing for GenAI
# MAGIC >
# MAGIC > ⏱ **Time:** 15 minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🗺 The map, revisited
# MAGIC
# MAGIC This morning, every one of these questions got the answer *"I don't know."* Here's the answer now:
# MAGIC
# MAGIC | # | Question | Answer today | Module |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Where did the data come from? Which version? Will it scale? | Delta tables in Unity Catalog with lineage, history, and time travel; Spark for any size | 02 |
# MAGIC | 2 | Which params, code, and metrics produced the model? | Every run is in MLflow, including the dataset version | 03 |
# MAGIC | 3 | Is scoring using the same preprocessing as training? | Yes, because preprocessing is inside the pipeline artifact | 04 |
# MAGIC | 4 | Do we have to rewrite scikit-learn to scale? | No: `applyInPandas` for many models, `spark_udf` / `mapInPandas` for many rows | 05 |
# MAGIC | 5 | Why does every team compute features differently? | Feature tables, point-in-time lookups, models that fetch their own features | 06 |
# MAGIC | 6 | Which model is live? Can we roll back? | `@champion` alias in Unity Catalog; rollback is one line | 07 |
# MAGIC | 7 | Is it still accurate? When do we retrain? | Drift and quality monitors; retrain on evidence, not on a calendar | 08 |
# MAGIC | 8 | How do we get to production safely? | Asset Bundles: one definition, dev/staging/prod targets, CI/CD | 09 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 The lifecycle on one page
# MAGIC
# MAGIC ```mermaid
# MAGIC flowchart LR
# MAGIC     A[Raw files<br/>Volume] --> B[Delta tables<br/>Unity Catalog]
# MAGIC     B --> C[Feature tables<br/>point-in-time]
# MAGIC     C --> D[Train + track<br/>MLflow]
# MAGIC     D --> E[Register<br/>UC Model Registry]
# MAGIC     E -->|"@champion"| F[Batch scoring<br/>spark_udf / jobs]
# MAGIC     E -->|"@champion"| G[Model Serving<br/>REST API]
# MAGIC     F --> H[Inference table]
# MAGIC     G --> H
# MAGIC     H --> I[Data Quality Monitor<br/>drift + quality]
# MAGIC     I -->|drift alert| J[Retrain job<br/>Asset Bundle]
# MAGIC     J --> D
# MAGIC ```
# MAGIC
# MAGIC Every box is governed by the **same** Unity Catalog permissions, and every arrow is captured as **lineage**. That's the real difference from the legacy platform. It isn't any single feature; it's that all the pieces know about each other.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Where to go next
# MAGIC
# MAGIC ### AutoML: a strong baseline in minutes
# MAGIC
# MAGIC Before hand-tuning anything, let **AutoML** try. Go to **Experiments → Create AutoML experiment**, pick `customer_features`, set the target to `churned`, and start. It trains and tunes several model families, logs every trial to MLflow, and generates an **editable notebook** for the best model. The generated code is a good template, and the leaderboard is an honest baseline to beat.
# MAGIC
# MAGIC > ⚠️ **Compute requirement.** On serverless, AutoML supports only **forecasting**. Classification and regression, like this churn problem, need a **classic cluster** running **Databricks Runtime ML** in **dedicated (single user) access mode**. That's why it's a pointer for later and not a hands-on step today.
# MAGIC
# MAGIC ### The Databricks Assistant: AI in every cell
# MAGIC
# MAGIC Press **Cmd/Ctrl + I** in a cell and ask, for example, *"add a feature for the number of distinct channels used in 90 days"* or *"why did this cell fail?"*. The Assistant knows your tables, their columns, and the error in front of you. Use it to write the boilerplate so you can spend your time on the modelling decisions.
# MAGIC
# MAGIC ### MLflow Tracing: when your model becomes an agent
# MAGIC
# MAGIC Today was classical ML. When the bank builds GenAI applications, such as a retention assistant that reads the churn score, looks up the customer, and drafts an offer, MLflow **Tracing** records every step: inputs, outputs, latency, and errors. It's the same MLflow and the same experiment. Here's a taste:

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.14"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging, os, re, mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities.trace_location import UnityCatalog
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())

# Traces live in UC tables, so reading them back needs a SQL warehouse. Pick one, serverless first.
warehouses = sorted(WorkspaceClient().warehouses.list(), key=lambda wh: not wh.enable_serverless_compute)
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouses[0].id   # 👈 or paste a warehouse ID you can use

# Some workspaces (e.g. Free Edition) don't expose Git details to MLflow; hide that one harmless warning
logging.getLogger("mlflow.tracking.context.registry").addFilter(lambda r: "extraContext" not in r.getMessage())
mlflow.set_experiment(
    f"/Users/{user}/churn_workshop",
    trace_location=UnityCatalog(catalog_name=CATALOG, schema_name=SCHEMA, table_prefix="churn_workshop"),
)

@mlflow.trace(span_type="TOOL")
def get_churn_risk(customer_id: str) -> float:
    row = spark.table(f"{CATALOG}.{SCHEMA}.churn_scores").filter(f"customer_id = '{customer_id}'").first()
    return round(row.p_churn, 3)

@mlflow.trace(span_type="CHAIN")
def retention_assistant(customer_id: str) -> str:
    risk = get_churn_risk(customer_id)
    offer = "personal call + fee waiver" if risk > 0.6 else "app voucher" if risk > 0.3 else "no action"
    return f"Customer {customer_id}: churn risk {risk:.0%} → {offer}"

top = spark.table(f"{CATALOG}.{SCHEMA}.churn_scores").orderBy("p_churn", ascending=False).first().customer_id
retention_assistant(top)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The output cell shows a **trace**: the parent `retention_assistant` span with the `get_churn_risk` tool call nested inside, each with inputs, outputs, and timing.
# MAGIC - The trace was written to **Unity Catalog**, in the location we set in Module 03. Writing needs nothing extra, but reading traces back (this viewer, the Experiments UI, `mlflow.search_traces`) runs SQL on a warehouse. That's what `MLFLOW_TRACING_SQL_WAREHOUSE_ID` is for, and MLflow starts the warehouse if it's stopped.
# MAGIC - In a real agent, the spans would be LLM calls, retrievals, and tool calls. Tracing, evaluation, and monitoring for GenAI build on everything you learned today.
# MAGIC
# MAGIC Because traces are tables, the SQL you used in Module 02 works on them too:

# COMMAND ----------

import time
trace_id = mlflow.get_last_active_trace_id().split("/")[-1]   # the trace we just created
query = f"""
  SELECT name, parent_span_id IS NULL AS is_root,
         round((end_time_unix_nano - start_time_unix_nano) / 1e6, 2) AS duration_ms
  FROM {CATALOG}.{SCHEMA}.churn_workshop_otel_spans
  WHERE trace_id = '{trace_id}'
  ORDER BY start_time_unix_nano"""

for _ in range(12):                      # spans land a few seconds after the call
    if spark.sql(query).count():
        break
    time.sleep(5)
display(spark.sql(query))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Each row is one **span**. The root span (`retention_assistant`) has no parent, and the tool call points to it through `parent_span_id`.
# MAGIC - Spans are written asynchronously, so they reach the table a few seconds after the call. The loop waits for the trace we just created, which `get_last_active_trace_id()` identifies.
# MAGIC - Because they're in UC, traces get the same governance as any table: grants, lineage, and dashboards.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧹 Clean up (optional)

# COMMAND ----------

CLEANUP = False      # 👈 set to True to delete everything you created today

if CLEANUP:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    try:
        t = w.tables.get(f"{CATALOG}.{SCHEMA}.churn_inference_log")
        w.data_quality.delete_monitor(object_type="table", object_id=t.table_id)
    except Exception as e:
        print(f"No monitor to delete ({e})")
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")   # tables, volumes, models
    print("Schema dropped. Also run: databricks bundle destroy -t dev")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Keep learning
# MAGIC
# MAGIC - Cheat sheet: legacy → Databricks, a one-page summary of today
# MAGIC - Glossary, every term from today in one sentence each
# MAGIC - Databricks Academy: *Get Started with Databricks for Machine Learning*, then *Machine Learning with Databricks*
# MAGIC - Docs: [MLflow on Databricks](https://docs.databricks.com/aws/en/mlflow/) · [Feature Engineering in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/feature-store/) · [Data quality monitoring](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/) · [Asset Bundles](https://docs.databricks.com/aws/en/dev-tools/bundles/)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *If you had to explain to your manager in one sentence what changed today, what would it be?*
# MAGIC >
# MAGIC > We didn't change the model, which is still scikit-learn. We changed everything *around* it: the data, experiments, features, models, monitoring, and deployments are now versioned, governed, connected by lineage, and automated, so every production question has an answer.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 09 — From Notebook to Production | Back to README
