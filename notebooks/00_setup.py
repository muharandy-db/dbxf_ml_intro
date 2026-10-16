# Databricks notebook source

# MAGIC %md
# MAGIC # Module 00 — Setup: Bring the Data Into the Lakehouse
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Clone this repo into Databricks as a **Git folder**
# MAGIC > - Create your personal **schema** and **volume** in Unity Catalog
# MAGIC > - Land the three CSV files as governed **Delta tables**
# MAGIC >
# MAGIC > ⏱ **Time:** 15 minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea
# MAGIC
# MAGIC Before any modelling, every attendee needs the same three things: a place to work, the raw files, and tables built from them. Unity Catalog organises everything in a three-level namespace, **catalog → schema → object**, the way a library is organised into building, shelf, and book. Your instructor has already created the *building*, a catalog called `ml_workshop`. You will create your own *shelf*, a schema named after you, so nobody overwrites anyone else's work.
# MAGIC
# MAGIC Two kinds of objects will live on that shelf:
# MAGIC
# MAGIC - **Volumes** hold *files*: CSVs, images, model artifacts. Think of a volume as a governed folder.
# MAGIC - **Tables** hold *data with a schema*: rows and columns stored in the Delta format, with versioning, ACID transactions, and access control.
# MAGIC
# MAGIC The flow for this module is simple: **Git folder (CSV) → Volume (raw file) → Delta table**. Every later module reads only from tables.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Before you start
# MAGIC
# MAGIC 1. In Databricks, go to **Workspace → Users → *your name* → Create → Git folder** and paste this repo's URL, `https://github.com/muharandy-db/dbxf_ml_intro`. The repo lands at `/Workspace/Users/<you>/dbxf_ml_intro`.
# MAGIC 2. Create a new notebook (**New → Notebook**) and name it `00_setup`.
# MAGIC 3. In the compute dropdown at the top right, select **Serverless**.
# MAGIC 4. Open the **Environment** panel (the 🧩 icon on the right side of the notebook) and set **Environment version** to **4** or later. Spark ML (Module 04) needs it on serverless. Do this for **every notebook you create today**.
# MAGIC
# MAGIC > 💡 **How to follow along:** copy each code cell below into your notebook, one cell at a time, and run it with **Shift + Enter**. If you fall behind, the complete notebook is in [`notebooks/00_setup.py`](../notebooks/00_setup.py).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC ### Cell 1 — Workshop config

# COMMAND ----------

import re

CATALOG = "ml_workshop"                                   # 👈 the catalog your instructor created
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())   # one schema per attendee

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"You are working in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `spark` is already there. Every Databricks notebook comes with a ready-made `SparkSession`, so there is no setup boilerplate.
# MAGIC - The schema name comes from your login, so everyone in the room gets an isolated sandbox inside the shared catalog.
# MAGIC - `USE CATALOG` / `USE SCHEMA` work like `cd`: later cells can write `customers` instead of `ml_workshop.<you>.customers`.
# MAGIC
# MAGIC ### Cell 2 — Create a volume and copy the raw files into it

# COMMAND ----------

import os, shutil

DATA_DIR = f"/Workspace/Users/{user}/dbxf_ml_intro/data"   # 👈 the data/ folder of your Git folder
spark.sql("CREATE VOLUME IF NOT EXISTS raw")
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

for f in ["customers.csv", "transactions.csv", "customers_next_quarter.csv"]:
    shutil.copy(f"{DATA_DIR}/{f}", f"{RAW}/{f}")

print(os.listdir(RAW))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - A volume path (`/Volumes/<catalog>/<schema>/<volume>`) looks like a normal folder to Python, so `shutil`, `pandas`, and `open()` all work on it.
# MAGIC - Unlike a shared network drive, a volume has **permissions and an audit log**. You can `GRANT READ VOLUME` to a team, and Unity Catalog records who read what.
# MAGIC - If you cloned the repo somewhere else, change `DATA_DIR`. Alternatively, upload the CSVs through **Catalog → your schema → raw → Upload to this volume**.
# MAGIC
# MAGIC ### Cell 3 — Turn files into Delta tables

# COMMAND ----------

for name in ["customers", "transactions", "customers_next_quarter"]:
    (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{RAW}/{name}.csv")
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(name))
    print(f"✅ {name}: {spark.table(name).count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `saveAsTable` writes a **managed Delta table**. Unity Catalog decides where the files live, and you never handle storage paths.
# MAGIC - Spark read the CSV *in parallel*. With 75k rows that doesn't matter, but the identical code would work on 75 billion rows.
# MAGIC - From now on nobody should touch the CSV. The **table is the source of truth**, and it is versioned, which matters in Module 02.
# MAGIC
# MAGIC ### Cell 4 — Take a look

# COMMAND ----------

display(spark.table("customers"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `display()` gives you an interactive grid. Click **+ → Visualization** to build a chart, or **+ → Data Profile** for instant summary statistics on every column.
# MAGIC - Open **Catalog** in the left sidebar and find your schema. You'll see the three tables and the volume, each with its schema, sample data, and (later) lineage.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🗂 The dataset
# MAGIC
# MAGIC All three tables describe customers of a fictional Indonesian retail bank. Money is in **IDR millions**.
# MAGIC
# MAGIC | Table | Rows | Grain | Purpose |
# MAGIC |---|---|---|---|
# MAGIC | `customers` | 10,000 | one row per customer, snapshot **2026-06-30** | training data; `churned` = left within the next quarter |
# MAGIC | `transactions` | ~75,000 | one row per transaction, Apr–Sep 2026 | raw material for behavioural features |
# MAGIC | `customers_next_quarter` | 3,000 | the same customers, snapshot **2026-09-30** | "the future": used for scoring and drift detection |
# MAGIC
# MAGIC See the README for the column dictionary.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | Legacy | Databricks |
# MAGIC |---|---|---|
# MAGIC | Where data lives | CSV exports on a shared drive | Delta tables in Unity Catalog |
# MAGIC | Isolation | "Please don't overwrite my file" | One schema per person, enforced by permissions |
# MAGIC | Discoverability | Ask around | Search the Catalog; see schema, owner, and sample data |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *Why do we copy the CSV into a volume and then build a table, instead of reading the CSV straight from the Git folder in every module?*
# MAGIC >
# MAGIC > Git folders are for *code*, while volumes and tables are for *data*. A volume gives the raw file governance (permissions and audit). The Delta table gives it a schema, versioning, and parallel reads. Every later module reads the same governed table, so there is exactly one source of truth.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Back to README | Next: Module 01 — The Whole Game, the Legacy Way →
