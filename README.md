# From Legacy ML to Databricks ML — A One-Day Hands-On Workshop

> *The retention team asks a simple question: **"Which customers will leave us next quarter?"***
> *By the end of today you'll have answered it twice: once the way most teams do it now, and once the way it holds up in production.*

This workshop is for practitioners who **already know some machine learning**. You've trained a scikit-learn model before. Now you're moving from a platform with little ML support to Databricks. We use **one small, synthetic banking dataset** all day, so the attention stays on the concepts and the platform rather than on the data.

The model itself barely changes during the day. What changes is everything *around* it.

---

## 🧭 How this workshop is designed

**1. Play the whole game first.** In [Module 01](modules/01-the-whole-game-legacy.md) you build a complete churn model in 15 minutes the familiar way, with pandas, scikit-learn, and a pickle file. Then we ask the questions a regulator, a manager, or a new teammate would ask six months later. None of them can be answered. Those **eight production questions** are the map for the rest of the day.

**2. One story, one dataset.** Every module improves the *same* churn model, so the only new thing in each module is the idea it teaches.

**3. Follow the lifecycle.** Data → features → training & tracking → packaging → governance & deployment → monitoring → automation. Each module answers one production question and contrasts *the legacy way* with *the Databricks way*.

**4. Every module has the same shape:**

| Section | What it's for |
|---|---|
| 🎯 Learning objectives · ⏱ time | What you'll be able to do |
| 🏚 **The legacy way** | A short, familiar pain story |
| 💡 **The idea** | The concept from first principles: intuition first, then mechanics |
| 🛠 **Hands-on** | Numbered cells to copy and run, each followed by **What to point out** |
| ✍️ **Your turn** | A 5-minute exercise, solution included |
| ⚖️ **Legacy vs Databricks** | One-table recap |
| **Check your understanding** | One or two questions that test the key insight |

---

## 📅 Agenda (≈ 6 hours)

| # | Module | Production question it answers | Key capabilities | Time |
|---|---|---|---|---|
| 00 | [Setup](modules/00-setup.md) | — | Git folders, Unity Catalog schema & volume, Delta tables | 15m |
| 01 | [The Whole Game, the Legacy Way](modules/01-the-whole-game-legacy.md) | *"What's missing?"* | pandas + scikit-learn + pickle; the accuracy trap; **the 8 questions** | 30m |
| 02 | [Your Data Lives in the Lakehouse](modules/02-data-in-the-lakehouse.md) | Where did the data come from? Will it scale? | **pandas vs Spark**, lazy evaluation, point-in-time features, Delta time travel, lineage | 45m |
| 03 | [Training That Remembers](modules/03-training-that-remembers.md) | Which params/code/data produced the model? | **MLflow Tracking**, autologging, dataset lineage, Optuna tuning with nested runs | 45m |
| 04 | [Pipelines, Not Scripts](modules/04-pipelines-not-scripts.md) | Is scoring using the same preprocessing? | Training/serving skew, scikit-learn `Pipeline`, **`spark.ml.Pipeline`**, how Spark ML distributes training, **sklearn vs Spark ML decision guide** | 45m |
| | ☕ *Lunch* | | | 45m |
| 05 | [Bring Your scikit-learn, Scale It Out](modules/05-scale-out-sklearn.md) | Do we have to rewrite scikit-learn? | **`groupBy().applyInPandas`** (many models in parallel), **`spark_udf`** & `mapInPandas` (distributed inference) | 35m |
| 06 | [Features as Shared Products](modules/06-feature-store.md) | Why does every team compute features differently? | **Feature Store** in Unity Catalog, point-in-time lookups, `score_batch` | 40m |
| 07 | [Ship It with Governance](modules/07-ship-it-with-governance.md) | Which model is live? Can we roll back? | **UC Model Registry**, aliases, champion/challenger, batch scoring, Model Serving | 30m |
| 08 | [Keep It Healthy](modules/08-keep-it-healthy.md) | Is it still accurate? When to retrain? | **Scheduled retraining vs drift detection**, PSI, Data Quality Monitoring, drift-triggered retraining | 40m |
| 09 | [From Notebook to Production](modules/09-notebook-to-production.md) | How do we promote dev → prod safely? | **Jobs + Databricks Asset Bundles**, dev/staging/prod targets, CI/CD | 35m |
| 10 | [Wrap-Up](modules/10-wrap-up.md) | Revisit all 8 questions | Lifecycle on one page; AutoML, Assistant, MLflow Tracing preview | 15m |

> 🧑‍🏫 **Running short on time?** Modules 05 and 09 work well as instructor demos. See the [instructor guide](docs/instructor-guide.md) for suggested cuts.

---

## ✅ Prerequisites

**For attendees**
- Comfortable with Python and pandas; has trained at least one scikit-learn model
- A Databricks workspace login with **serverless notebooks** enabled, using serverless **environment version 4 or later** (set it in each notebook's Environment panel; Spark ML needs it)

**For the instructor (before the day)**
- A Unity Catalog catalog for the workshop (default: `ml_workshop`). Attendees need `USE CATALOG` and `CREATE SCHEMA` on it:
  ```sql
  CREATE CATALOG IF NOT EXISTS ml_workshop;
  GRANT USE CATALOG, CREATE SCHEMA ON CATALOG ml_workshop TO `account users`;
  ```
- This repo pushed to a Git provider the workspace can reach (for Git folders)
- *(Optional, Module 09)* `ml_workshop_staging` / `ml_workshop_prod` catalogs if you want to demo promotion
- *(Optional, Module 07)* permission to create Model Serving endpoints for the serving demo

---

## 🧑‍💻 How to use this repo

**Option 1: copy and paste cell by cell (recommended for the live workshop).**
Open each module page, create a new notebook on **Serverless**, and copy each code cell into it in order. The **What to point out** notes under each cell are what the instructor will talk through. Reading them as you go is the self-paced version of the workshop.

**Option 2: import the complete notebooks.**
The [`notebooks/`](notebooks/) folder has every module as a ready-to-run Databricks notebook. They're generated from the module pages, so the code is identical. Clone the repo as a **Git folder** and open `notebooks/` directly, which is handy for catching up or reviewing afterwards.

> In the module pages, ` ```python ` blocks are cells to run, while ` ```py ` blocks are snippets to read or edit into an existing cell.

---

## 🗂 Repository layout

```text
.
├── README.md                 ← you are here
├── modules/                  ← the workshop: one page per module (source of truth)
├── notebooks/                ← complete notebooks, generated from modules/
├── data/                     ← the three CSV files
├── bundle/                   ← Databricks Asset Bundle used in Module 09
├── docs/
│   ├── cheat-sheet.md        ← legacy → Databricks on one page
│   ├── glossary.md           ← every term from the day
│   └── instructor-guide.md   ← timing, talking points, common errors
└── scripts/
    ├── generate_data.py      ← regenerates data/ (seeded, reproducible)
    └── build_notebooks.py    ← rebuilds notebooks/ from modules/
```

---

## 📊 The dataset

A fictional Indonesian retail bank. All amounts are in **IDR millions**. The data is synthetic and generated by [`scripts/generate_data.py`](scripts/generate_data.py) with a fixed seed, but it has realistic structure: class imbalance, missing values, behavioural signal, and a **deliberately drifted** next quarter.

### `customers.csv` — 10,000 rows, snapshot 2026-06-30

| Column | Description |
|---|---|
| `customer_id` | Unique ID (`C00001` …) |
| `branch_region` | Jakarta, Surabaya, Bandung, Medan, Makassar |
| `age` | Age in years |
| `tenure_months` | Months as a customer |
| `num_products` | Number of bank products held (1–4) |
| `balance` | Current balance (IDR m); 0 for ~22% of customers |
| `credit_score` | 300–850; **~3% missing** |
| `has_credit_card` | 1 if the customer holds a credit card |
| `is_active_member` | 1 if flagged active by the CRM |
| `estimated_salary` | Monthly salary (IDR m); **~2% missing** |
| `digital_logins_30d` | Mobile/web logins in the last 30 days |
| `complaints_90d` | Complaints filed in the last 90 days |
| `snapshot_date` | Date the row describes |
| `churned` | **Label**: 1 if the customer left in the following quarter (~19%) |

### `transactions.csv` — ~77,000 rows, Apr–Sep 2026

| Column | Description |
|---|---|
| `txn_id` | Unique transaction ID |
| `customer_id` | Who transacted |
| `txn_ts` | Timestamp |
| `amount` | Amount (IDR m) |
| `channel` | `mobile`, `web`, `atm`, `branch` |

### `customers_next_quarter.csv` — 3,000 rows, snapshot 2026-09-30

The same columns as `customers.csv`, for 3,000 of the same customers one quarter later. In between, three things happened: a **mobile-app relaunch** (logins up), a **competitor rate cut** (balances down), and a **service outage** (complaints up). A digital-only competitor also started poaching heavy app users, which changes the churn relationship itself. That's your data drift *and* concept drift for Module 08.

---

## 🔁 Regenerating data and notebooks

```bash
python scripts/generate_data.py     # rewrites data/*.csv (identical output every run)
python scripts/build_notebooks.py   # rewrites notebooks/*.py from modules/*.md
```

Edit the **module pages**, never the notebooks, then rebuild.
