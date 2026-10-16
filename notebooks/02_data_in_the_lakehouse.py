# Databricks notebook source

# MAGIC %md
# MAGIC # Module 02 — Your Data Lives in the Lakehouse
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Explain the difference between a **pandas DataFrame** and a **Spark DataFrame**: single machine vs a cluster, eager vs lazy
# MAGIC > - Engineer behavioural features from raw transactions **point-in-time correctly**
# MAGIC > - Use **Delta history and time travel** to answer *"which version of the data trained this model?"*
# MAGIC >
# MAGIC > ⏱ **Time:** 45 minutes  ·  Answers production question **#1**: *Where did the data come from? Which version? Will it scale?*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC The CSV in Module 01 was a dead end. Nobody knows which query produced it, whether it was filtered, or what changed between `customers_may.csv` and `customers_jun.csv`. When the data grows past what one machine can hold, the notebook simply crashes with `MemoryError`. The usual fix is "sample 10% and hope."
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: one chef vs a kitchen brigade
# MAGIC
# MAGIC **pandas is a single brilliant chef.** Everything happens in one kitchen, the memory of one machine, and it happens *immediately*: you ask for a chopped onion and the onion is chopped. For a dinner for ten, nothing beats it.
# MAGIC
# MAGIC **Spark is a kitchen brigade.** A head chef, the **driver**, reads the order and splits the work into tasks. Many line cooks, the **executors**, each prepare their own share of the ingredients, called **partitions**, in parallel. Feeding ten thousand guests stops being a problem of one bigger kitchen. You add more cooks.
# MAGIC
# MAGIC Two consequences follow from the brigade model. Both are worth stating plainly, because they are the whole of what you need to know to start:
# MAGIC
# MAGIC 1. **Spark is lazy.** When you write `df.filter(...).groupBy(...)`, nothing is cooked. Spark writes the order down as a **plan**. Only an **action** (`display`, `count`, `write`, `toPandas`) sends the order to the kitchen. This sounds like an inconvenience, but it has a purpose: because Spark sees the *whole* order before starting, its optimiser can reorder steps, skip unused columns, and push filters down to the storage layer.
# MAGIC 2. **Spark DataFrames are not in your notebook's memory.** They are a *description* of data that lives in tables. `toPandas()` brings the result back to the driver as a pandas DataFrame, which is fine for small results and fatal for a billion rows.
# MAGIC
# MAGIC The practical takeaway is simple. **Use Spark (or SQL) to move and shape data at any size. Use pandas when the result is small enough to fit comfortably on one machine.** On Databricks, both run in the same notebook, on the same data.
# MAGIC
# MAGIC > 📘 **Refresher: point-in-time correctness.** To predict churn *after* 30 June, a feature may only use information *available on* 30 June. If "transactions in the last 90 days" accidentally included July, the model would learn from the future. It would look brilliant in testing and fail in production. This is leakage across *time* rather than across rows, and it is the most common silent bug in behavioural features.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a notebook `02_data_in_the_lakehouse`.
# MAGIC
# MAGIC ### Cell 1 — Config

# COMMAND ----------

import re
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — A Spark DataFrame is a plan, not data

# COMMAND ----------

from pyspark.sql import functions as F

customers = spark.table("customers")
churn_by_region = (customers
    .groupBy("branch_region")
    .agg(F.count("*").alias("customers"),
         F.round(F.avg("churned"), 3).alias("churn_rate"))
    .orderBy(F.desc("churn_rate")))

print(type(churn_by_region))   # nothing has been computed yet
churn_by_region.explain()      # ...this is the plan Spark will run

# COMMAND ----------

# MAGIC %md
# MAGIC > 🤔 **Predict first:** how long will this cell take on 10k rows? On 10 billion?
# MAGIC
# MAGIC **What to point out**
# MAGIC - The cell returns *instantly* at any data size, because it only **builds a plan**. `explain()` prints that plan: a scan, an aggregation, and an exchange (the "shuffle" where cooks swap ingredients).
# MAGIC - `F` (`pyspark.sql.functions`) is the toolbox: `F.col`, `F.when`, `F.avg`, `F.datediff`, and so on. Each one becomes part of the plan, not a Python loop.
# MAGIC
# MAGIC ### Cell 3 — Run it (the action)

# COMMAND ----------

display(churn_by_region)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `display()` is the **action** that actually executes the plan.
# MAGIC - Medan churns noticeably more. That's a business insight in one line of code, and also a hint that one model per region might be worth trying (Module 05).
# MAGIC - Toggle the result into a **bar chart** with the **+** next to the table.
# MAGIC
# MAGIC ### Cell 4 — Same question, in SQL

# COMMAND ----------

display(spark.sql("""
    SELECT branch_region,
           COUNT(*)                AS customers,
           ROUND(AVG(churned), 3)  AS churn_rate
    FROM customers
    GROUP BY branch_region
    ORDER BY churn_rate DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - SQL and the DataFrame API compile to the **same plan** and run on the same engine. Analysts who think in SQL and data scientists who think in Python can work on the same tables.
# MAGIC - You can also start a cell with `%sql` and write pure SQL.
# MAGIC
# MAGIC ### Cell 5 — Behavioural features, point-in-time correct

# COMMAND ----------

def txn_features(snapshot_date: str):
    """Aggregate the 90 days of transactions *before* snapshot_date."""
    snap = F.lit(snapshot_date).cast("date")
    return (spark.table("transactions")
        .withColumn("days_before", F.datediff(snap, F.to_date("txn_ts")))
        .filter(F.col("days_before").between(1, 90))           # 👈 only the past, never the future
        .groupBy("customer_id")
        .agg(
            F.count(F.when(F.col("days_before") <= 30, 1)).alias("txn_count_30d"),
            F.count("*").alias("txn_count_90d"),
            F.round(F.avg("amount"), 3).alias("avg_amount_90d"),
            F.round(F.avg((F.col("channel") == "mobile").cast("int")), 3).alias("mobile_share_90d"),
            F.min("days_before").alias("days_since_last_txn"))
        .withColumn("snapshot_date", snap))

display(txn_features("2026-06-30"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The `between(1, 90)` filter is **point-in-time correctness** in one line. Features for the 30 June snapshot can never see July.
# MAGIC - It's an ordinary Python function returning a Spark plan. We can call it for *any* snapshot date, and we will, because "next quarter" needs the same features computed as of 30 September.
# MAGIC - `F.count(F.when(...))` is the idiomatic Spark way to write *"count rows where…"*.
# MAGIC
# MAGIC ### Cell 6 — Assemble the modelling table

# COMMAND ----------

profiles = (spark.table("customers")
    .unionByName(spark.table("customers_next_quarter"))
    .withColumn("snapshot_date", F.to_date("snapshot_date")))

features = txn_features("2026-06-30").unionByName(txn_features("2026-09-30"))

customer_features = (profiles
    .join(features, ["customer_id", "snapshot_date"], "left")
    .fillna({"txn_count_30d": 0, "txn_count_90d": 0,
             "mobile_share_90d": 0.0, "days_since_last_txn": 91}))

(customer_features.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable("customer_features"))

spark.sql("COMMENT ON TABLE customer_features IS "
          "'One row per customer per quarterly snapshot. Profile + 90-day transaction features. Label: churned.'")
display(spark.table("customer_features").groupBy("snapshot_date").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - The table has **one row per customer per snapshot**: 10,000 rows for June (training) and 3,000 for September (our "future").
# MAGIC - Customers with no transactions get explicit defaults rather than silent nulls. That's a modelling decision, written down in code.
# MAGIC - `COMMENT ON TABLE` is documentation that **lives with the data**. It shows up in Catalog Explorer and in search.
# MAGIC
# MAGIC ### Cell 7 — Which version of the data? (Delta time travel)

# COMMAND ----------

display(spark.sql("DESCRIBE HISTORY customer_features")
          .select("version", "timestamp", "operation", "userName"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Every write to a Delta table creates a new **version**, with who did it, when, and how.
# MAGIC - Run Cell 6 again and then this cell: you'll see version 1, 2, 3…
# MAGIC - This is the answer to *"which version of the data trained the model?"* In Module 03, MLflow records the table and version automatically when we train.
# MAGIC
# MAGIC ### Cell 8 — Read the past

# COMMAND ----------

v0 = spark.read.option("versionAsOf", 0).table("customer_features")
print(f"Version 0 has {v0.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - **Time travel** reads the table exactly as it was at a version (or a timestamp). Reproducing last quarter's training set takes one line, not a hunt for `customers_mar_FINAL.csv`.
# MAGIC - Now open **Catalog → customer_features → Lineage**. Databricks has already recorded that this table was built from `customers`, `customers_next_quarter`, and `transactions`. Nobody had to draw that diagram.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC The retention team thinks recent spending matters more than 90-day spending. Add an **`avg_amount_30d`** feature to `txn_features`, rebuild the table, and check the new column exists.
# MAGIC
# MAGIC
# MAGIC **Solution**
# MAGIC
# MAGIC Add this line inside `.agg(...)` in Cell 5:
# MAGIC
# MAGIC ```py
# MAGIC F.round(F.avg(F.when(F.col("days_before") <= 30, F.col("amount"))), 3).alias("avg_amount_30d"),
# MAGIC ```
# MAGIC
# MAGIC Then re-run Cells 5, 6 and 7. `F.avg` ignores nulls, so `F.when` without an `otherwise` averages only the last 30 days. The table history now shows a new version, and the old one is still readable.
# MAGIC
# MAGIC > Later modules list the feature columns explicitly, so leaving the new column in is harmless.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks |
# MAGIC |---|---|---|
# MAGIC | Scale | One machine's RAM; sample if it doesn't fit | Same code on 10k or 10bn rows, more executors |
# MAGIC | Language | pandas *or* SQL, different tools | DataFrame API and SQL on the same engine and tables |
# MAGIC | Versioning | `customers_jun_FINAL_v2.csv` | Delta history and time travel built in |
# MAGIC | Lineage | A diagram on someone's wiki (out of date) | Captured automatically in Unity Catalog |
# MAGIC | Time-correct features | Depends on who wrote the query | A reusable, tested function with an explicit window |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *You run Cell 2 on a 5-billion-row table and it finishes in a second. Did Spark just process 5 billion rows in a second?*
# MAGIC >
# MAGIC > No. Cell 2 only **built a plan**, because transformations are lazy. The work happens when an action such as `display`, `count`, `write`, or `toPandas` runs. That is also when you'd see the real duration.
# MAGIC
# MAGIC > *Why does `txn_features` take the snapshot date as a parameter instead of using today's date?*
# MAGIC >
# MAGIC > Features must be computed **as of the moment the prediction would have been made**. Using a parameter guarantees training features (30 June) and scoring features (30 September) are built with the same logic and never peek into the future.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 01 — The Whole Game | Next: Module 03 — Training That Remembers →
