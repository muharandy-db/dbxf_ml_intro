# Databricks notebook source

# MAGIC %md
# MAGIC # Module 06 — Features as Shared Products (Feature Store)
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Explain what a **feature table** is, and why a primary key plus a **timestamp key** matter
# MAGIC > - Build a training set with **point-in-time lookups** instead of hand-written joins
# MAGIC > - Log a model that **fetches its own features** at scoring time
# MAGIC >
# MAGIC > ⏱ **Time:** 40 minutes  ·  Answers production question **#5**: *Why does every team compute "average balance" differently?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC The churn team computes "average transaction amount over 90 days" in one SQL script. The cross-sell team computes it in another, with a 3-month calendar window instead of 90 days. The fraud team uses a third version. All three are called `avg_amount`. When a model goes to production, its features are rebuilt *again* by IT. Nobody can say which models depend on which feature, so nobody dares to change one.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: a shared pantry with labelled jars
# MAGIC
# MAGIC Picture a restaurant group where every kitchen makes its own stock from scratch, each slightly differently. Now picture a **central pantry**: the stock is made once, by a known recipe, in labelled jars with a date on each. Any kitchen takes what it needs, and the pantry knows which kitchens use which jars.
# MAGIC
# MAGIC A **Feature Store** is that pantry. In Databricks it's built into Unity Catalog: a **feature table** is simply a Delta table with a **primary key**, so it gets the same permissions, lineage, and search as any other table. Three ideas carry most of the value:
# MAGIC
# MAGIC 1. **Compute once, reuse everywhere.** A feature is defined in one place and discovered by everyone, so `avg_amount_90d` means one thing across the bank.
# MAGIC 2. **Point-in-time lookups.** The feature table has a **timestamp key** (`snapshot_date`). When you ask for a customer's features "as of 30 June," the store returns the latest values *on or before* 30 June, never later ones. That is the point-in-time correctness from Module 02, enforced by the platform instead of by careful SQL.
# MAGIC 3. **The model remembers where its features come from.** When you log a model through the Feature Store client, it records *which* feature tables and columns it was trained on. At scoring time you pass only the keys, and the model **looks up its own features**. Training/serving skew from Module 04 disappears for features too.
# MAGIC
# MAGIC ### What a point-in-time lookup does
# MAGIC
# MAGIC For each row of your training data, called the **spine**, the lookup finds the right feature row:
# MAGIC
# MAGIC ```text
# MAGIC feature_row(customer, t)  =  the row in the feature table with
# MAGIC                              customer_id = customer  AND  the largest snapshot_date ≤ t
# MAGIC ```
# MAGIC
# MAGIC Reading it: `customer` and `t` come from your spine, for example *(C00042, 2026-06-30)*. Among all the feature rows for C00042, take the most recent one that was **already known at time `t`**. Rows from the future are invisible. That one rule is what keeps July's transactions out of June's training set, however the tables are joined.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a notebook `06_feature_store`.
# MAGIC
# MAGIC ### Cell 0 — Libraries

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.14" scikit-learn==1.6.1 databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 1 — Config

# COMMAND ----------

import logging, re, mlflow
from pyspark.sql import functions as F
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
# Some workspaces (e.g. Free Edition) don't expose Git details to MLflow; hide that one harmless warning
logging.getLogger("mlflow.tracking.context.registry").addFilter(lambda r: "extraContext" not in r.getMessage())
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/churn_workshop")

PROFILE = ["branch_region", "age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d"]
TXN_FEATURES = ["txn_count_30d", "txn_count_90d", "avg_amount_90d",
                "mobile_share_90d", "days_since_last_txn"]
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_txn_features"
FS_MODEL_NAME = f"{CATALOG}.{SCHEMA}.churn_model_fs"

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - We split the columns into **profile** attributes (what the CRM knows about a customer) and **behavioural features** (what we derive from transactions). Only the behavioural ones go into the Feature Store today.
# MAGIC
# MAGIC ### Cell 2 — Create the feature table

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {FEATURE_TABLE} (
  customer_id          STRING NOT NULL,
  snapshot_date        DATE   NOT NULL,
  txn_count_30d        BIGINT,
  txn_count_90d        BIGINT,
  avg_amount_90d       DOUBLE,
  mobile_share_90d     DOUBLE,
  days_since_last_txn  INT,
  CONSTRAINT customer_txn_features_pk PRIMARY KEY (customer_id, snapshot_date TIMESERIES)
)
COMMENT 'Behavioural features from the 90 days of transactions before each snapshot_date.'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - That's all a feature table is: a Delta table with a **primary key**. The `TIMESERIES` keyword marks `snapshot_date` as the **timestamp key** that point-in-time lookups use.
# MAGIC - Primary-key columns must be `NOT NULL`, since a feature without an identity can't be looked up.
# MAGIC - The comment becomes searchable documentation in Catalog Explorer.
# MAGIC
# MAGIC ### Cell 3 — Publish the features

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

features_df = spark.table("customer_features").select("customer_id", "snapshot_date", *TXN_FEATURES)
fe.write_table(name=FEATURE_TABLE, df=features_df, mode="merge")
display(spark.table(FEATURE_TABLE).groupBy("snapshot_date").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - We *promote* the features we already built in Module 02. The logic isn't rewritten; the table just gets an identity.
# MAGIC - `mode="merge"` upserts on the primary key, so the nightly refresh job can re-run safely without creating duplicates.
# MAGIC - Open **Catalog → your schema → customer_txn_features**. The table carries a feature-table badge, and the left sidebar's **Features** page lets anyone in the bank search for it.
# MAGIC
# MAGIC ### Cell 4 — Build a training set from a "spine" + lookups

# COMMAND ----------

spine = (spark.table("customers")
         .withColumn("snapshot_date", F.to_date("snapshot_date"))
         .select("customer_id", "snapshot_date", *PROFILE, "churned"))

training_set = fe.create_training_set(
    df=spine,
    feature_lookups=[FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key="customer_id",
        timestamp_lookup_key="snapshot_date",          # 👈 point-in-time: no peeking into the future
        feature_names=TXN_FEATURES)],
    label="churned",
    exclude_columns=["customer_id", "snapshot_date"])

train_pdf = training_set.load_df().toPandas()
print(train_pdf.shape)
train_pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The **spine** holds only what *we* know: who, when, their profile, and the label. It contains **no transaction features**.
# MAGIC - `FeatureLookup` says *"for each spine row, fetch these features for this customer as of this date."* Nobody writes a join.
# MAGIC - `exclude_columns` drops the keys from the model inputs. Keys identify rows; they are not predictors.
# MAGIC
# MAGIC ### Cell 5 — Train and log a feature-aware model

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

NUMERIC = [c for c in PROFILE if c != "branch_region"] + TXN_FEATURES
X, y = train_pdf.drop(columns=["churned"]), train_pdf["churned"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["branch_region"])])),
    ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42)),
]).fit(X_train, y_train)

with mlflow.start_run(run_name="feature_store_model"):
    mlflow.set_tag("module", "06")
    mlflow.log_metric("val_auc", roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1]))
    fe.log_model(model=pipe, artifact_path="feature_store_pipeline", flavor=mlflow.sklearn,
                 training_set=training_set, registered_model_name=FS_MODEL_NAME,
                 infer_input_example=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - It's the same scikit-learn pipeline as Module 04. The Feature Store doesn't change *how* you model.
# MAGIC - `fe.log_model` (instead of `mlflow.sklearn.log_model`) stores the **feature lookup spec** inside the model: which table, which keys, which columns.
# MAGIC - It also **registers** the model in Unity Catalog. Open the model page and check **Lineage**: feature table → model, recorded automatically.
# MAGIC
# MAGIC ### Cell 6 — Score with keys only

# COMMAND ----------

from mlflow import MlflowClient
latest = max(int(v.version) for v in MlflowClient().search_model_versions(f"name='{FS_MODEL_NAME}'"))

next_quarter = (spark.table("customers_next_quarter")
                .withColumn("snapshot_date", F.to_date("snapshot_date"))
                .select("customer_id", "snapshot_date", *PROFILE))      # 👈 no transaction features here

scored = fe.score_batch(model_uri=f"models:/{FS_MODEL_NAME}/{latest}", df=next_quarter)
display(scored.select("customer_id", "snapshot_date", *TXN_FEATURES, "prediction").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC > 🤔 **Predict first:** `next_quarter` has no transaction columns. Where will `txn_count_90d` come from, and *which* quarter's values will it use?
# MAGIC
# MAGIC **What to point out**
# MAGIC - `score_batch` read the lookup spec from the model, **fetched the transaction features** for each customer **as of 30 September**, and then predicted.
# MAGIC - The scoring code doesn't need to know a single feature definition. Change the feature pipeline once, and every model that consumes it gets the fix.
# MAGIC - For real-time APIs, the same feature tables can be published to an **online store** so Model Serving can look features up in milliseconds. Same definitions, different speed.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC Discovery is half the value. Without writing a join, find out **which models use `customer_txn_features`**:
# MAGIC
# MAGIC 1. Open **Catalog → customer_txn_features → Lineage → See lineage graph**.
# MAGIC 2. Then try it in code: list the feature tables in your schema.
# MAGIC
# MAGIC
# MAGIC **Solution**

# COMMAND ----------

display(spark.sql(f"""
    SELECT table_name, constraint_name
    FROM {CATALOG}.information_schema.table_constraints
    WHERE table_schema = '{SCHEMA}' AND constraint_type = 'PRIMARY KEY'
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC Feature tables are ordinary Unity Catalog tables with a primary key, so the standard `information_schema` finds them. The lineage graph shows `churn_model_fs` as a downstream consumer.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks Feature Store |
# MAGIC |---|---|---|
# MAGIC | Feature definitions | Copy-pasted SQL per team | One feature table, discoverable in Unity Catalog |
# MAGIC | Time correctness | Depends on each join | Point-in-time lookups on the timestamp key |
# MAGIC | Scoring | Rebuild features in production code | Model looks up its own features (`score_batch`, serving) |
# MAGIC | Impact analysis | "Who uses this column?" → nobody knows | Lineage: feature table → models → endpoints |
# MAGIC | Real-time | Separate cache, separate code | Same tables published to an online store |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *Our feature table has rows for 30 June and 30 September. A training spine row says (C00042, 2026-08-15). Which feature row does the lookup return, and why?*
# MAGIC >
# MAGIC > The **30 June** row: the latest snapshot on or before 15 August. The 30 September row didn't exist yet on 15 August, so a point-in-time lookup must not use it.
# MAGIC
# MAGIC > *Why did `fe.log_model` matter more than training the model itself?*
# MAGIC >
# MAGIC > It packaged the feature lookup spec *with* the model. Any consumer, whether a batch job or a serving endpoint, now only supplies keys, and the model fetches the exact features it was trained on. That removes feature re-implementation, which is the main source of training/serving skew.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 05 — Scale Out Your scikit-learn | Next: Module 07 — Ship It with Governance →
