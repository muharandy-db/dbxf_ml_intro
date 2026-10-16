# Module 00 — Setup: Bring the Data Into the Lakehouse

> 🎯 **Learning objectives**
> - Clone this repo into Databricks as a **Git folder**
> - Create your personal **schema** and **volume** in Unity Catalog
> - Land the three CSV files as governed **Delta tables**
>
> ⏱ **Time:** 15 minutes

---

## 💡 The idea

Before any modelling, every attendee needs the same three things: a place to work, the raw files, and tables built from them. Unity Catalog organises everything in a three-level namespace, **catalog → schema → object**, the way a library is organised into building, shelf, and book. Your instructor has already created the *building*, a catalog called `ml_workshop`. You will create your own *shelf*, a schema named after you, so nobody overwrites anyone else's work.

Two kinds of objects will live on that shelf:

- **Volumes** hold *files*: CSVs, images, model artifacts. Think of a volume as a governed folder.
- **Tables** hold *data with a schema*: rows and columns stored in the Delta format, with versioning, ACID transactions, and access control.

The flow for this module is simple: **Git folder (CSV) → Volume (raw file) → Delta table**. Every later module reads only from tables.

---

## ✅ Before you start

1. In Databricks, go to **Workspace → Users → *your name* → Create → Git folder** and paste this repo's URL. The repo lands at `/Workspace/Users/<you>/dbx_ml_intro`.
2. Create a new notebook (**New → Notebook**) and name it `00_setup`.
3. In the compute dropdown at the top right, select **Serverless**.
4. Open the **Environment** panel (the 🧩 icon on the right side of the notebook) and set **Environment version** to **4** or later. Spark ML (Module 04) needs it on serverless. Do this for **every notebook you create today**.

> 💡 **How to follow along:** copy each code cell below into your notebook, one cell at a time, and run it with **Shift + Enter**. If you fall behind, the complete notebook is in [`notebooks/00_setup.py`](../notebooks/00_setup.py).

---

## 🛠 Hands-on

### Cell 1 — Workshop config

```python
import re

CATALOG = "ml_workshop"                                   # 👈 the catalog your instructor created
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())   # one schema per attendee

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"You are working in {CATALOG}.{SCHEMA}")
```

**What to point out**
- `spark` is already there. Every Databricks notebook comes with a ready-made `SparkSession`, so there is no setup boilerplate.
- The schema name comes from your login, so everyone in the room gets an isolated sandbox inside the shared catalog.
- `USE CATALOG` / `USE SCHEMA` work like `cd`: later cells can write `customers` instead of `ml_workshop.<you>.customers`.

### Cell 2 — Create a volume and copy the raw files into it

```python
import os, shutil

DATA_DIR = f"/Workspace/Users/{user}/dbx_ml_intro/data"   # 👈 the data/ folder of your Git folder
spark.sql("CREATE VOLUME IF NOT EXISTS raw")
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

for f in ["customers.csv", "transactions.csv", "customers_next_quarter.csv"]:
    shutil.copy(f"{DATA_DIR}/{f}", f"{RAW}/{f}")

print(os.listdir(RAW))
```

**What to point out**
- A volume path (`/Volumes/<catalog>/<schema>/<volume>`) looks like a normal folder to Python, so `shutil`, `pandas`, and `open()` all work on it.
- Unlike a shared network drive, a volume has **permissions and an audit log**. You can `GRANT READ VOLUME` to a team, and Unity Catalog records who read what.
- If you cloned the repo somewhere else, change `DATA_DIR`. Alternatively, upload the CSVs through **Catalog → your schema → raw → Upload to this volume**.

### Cell 3 — Turn files into Delta tables

```python
for name in ["customers", "transactions", "customers_next_quarter"]:
    (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(f"{RAW}/{name}.csv")
        .write.mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(name))
    print(f"✅ {name}: {spark.table(name).count():,} rows")
```

**What to point out**
- `saveAsTable` writes a **managed Delta table**. Unity Catalog decides where the files live, and you never handle storage paths.
- Spark read the CSV *in parallel*. With 75k rows that doesn't matter, but the identical code would work on 75 billion rows.
- From now on nobody should touch the CSV. The **table is the source of truth**, and it is versioned, which matters in Module 02.

### Cell 4 — Take a look

```python
display(spark.table("customers"))
```

**What to point out**
- `display()` gives you an interactive grid. Click **+ → Visualization** to build a chart, or **+ → Data Profile** for instant summary statistics on every column.
- Open **Catalog** in the left sidebar and find your schema. You'll see the three tables and the volume, each with its schema, sample data, and (later) lineage.

---

## 🗂 The dataset

All three tables describe customers of a fictional Indonesian retail bank. Money is in **IDR millions**.

| Table | Rows | Grain | Purpose |
|---|---|---|---|
| `customers` | 10,000 | one row per customer, snapshot **2026-06-30** | training data; `churned` = left within the next quarter |
| `transactions` | ~75,000 | one row per transaction, Apr–Sep 2026 | raw material for behavioural features |
| `customers_next_quarter` | 3,000 | the same customers, snapshot **2026-09-30** | "the future": used for scoring and drift detection |

See the [README](../README.md#-the-dataset) for the column dictionary.

---

## ⚖️ Legacy vs Databricks

| | Legacy | Databricks |
|---|---|---|
| Where data lives | CSV exports on a shared drive | Delta tables in Unity Catalog |
| Isolation | "Please don't overwrite my file" | One schema per person, enforced by permissions |
| Discoverability | Ask around | Search the Catalog; see schema, owner, and sample data |

---

**Check your understanding**

> *Why do we copy the CSV into a volume and then build a table, instead of reading the CSV straight from the Git folder in every module?*
>
> Git folders are for *code*, while volumes and tables are for *data*. A volume gives the raw file governance (permissions and audit). The Delta table gives it a schema, versioning, and parallel reads. Every later module reads the same governed table, so there is exactly one source of truth.

---

[← Back to README](../README.md) | [Next: Module 01 — The Whole Game, the Legacy Way →](01-the-whole-game-legacy.md)
