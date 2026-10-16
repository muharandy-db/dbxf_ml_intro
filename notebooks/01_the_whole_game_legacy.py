# Databricks notebook source

# MAGIC %md
# MAGIC # Module 01 — The Whole Game, the Legacy Way
# MAGIC
# MAGIC > 🎯 **Learning objectives**
# MAGIC > - Build a complete churn model in 15 minutes the way most teams do today: pandas, scikit-learn, pickle
# MAGIC > - Recall why **accuracy** misleads on imbalanced data, and what to measure instead
# MAGIC > - Produce the **list of production questions** that the rest of the day answers
# MAGIC >
# MAGIC > ⏱ **Time:** 30 minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏚 The legacy way
# MAGIC
# MAGIC Here is a story most of the room will recognise. The retention team emails a CSV called `customers_jun.csv` and asks: *"Can you tell us who is likely to leave next quarter?"* A data scientist opens a notebook on a laptop or a shared VM, cleans the file by hand, trains a model, prints an accuracy score, and saves `churn_model_final_v3.pkl`. The pickle gets emailed to IT, and IT is asked to "run it every month."
# MAGIC
# MAGIC That workflow is not *wrong*. The model may even be good. The trouble starts on day two, so we'll play the whole game once, quickly, and then look at what is missing.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 The idea: play the whole game first
# MAGIC
# MAGIC You learn football by playing a scrappy match, not by spending a semester on the physics of a kick. This module is the scrappy match: an end-to-end model, warts and all, in a few cells. The point is not the model. **The point is to notice everything the code cannot tell you**, and that list becomes the map for the rest of the day.
# MAGIC
# MAGIC We use plain pandas and scikit-learn here on purpose. Everyone in the room already knows them, and Databricks runs them happily. Changing the *tools* is not the lesson. The lesson is what the *platform* adds around them.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠 Hands-on
# MAGIC
# MAGIC Create a new notebook called `01_whole_game`, attached to **Serverless**.
# MAGIC
# MAGIC ### Cell 1 — Config

# COMMAND ----------

import re
CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
RAW = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - This module reads the **raw CSV** deliberately, the way a legacy pipeline does. From Module 02 onwards we read the tables instead.
# MAGIC
# MAGIC ### Cell 2 — "The CSV arrived by email"

# COMMAND ----------

import pandas as pd

pdf = pd.read_csv(f"{RAW}/customers.csv")
print(pdf.shape)
print(f"Churn rate: {pdf['churned'].mean():.1%}")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - 10,000 rows, 14 columns: a typical tabular problem.
# MAGIC - The churn rate is about **19%**. Keep that number in mind; it matters in two cells.
# MAGIC - `pdf` lives entirely in the memory of one machine, the notebook's **driver**. That's fine for 10k rows; Module 02 explains what changes when it isn't.
# MAGIC
# MAGIC ### Cell 3 — Hand-rolled preprocessing

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = pdf.drop(columns=["customer_id", "snapshot_date", "churned"])
y = pdf["churned"]

X = X.fillna(X.median(numeric_only=True))          # fill missing credit_score / salary
X = pd.get_dummies(X, columns=["branch_region"])   # one-hot encode region
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)   # 🤔 spot the bug?

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - Three manual steps (impute, encode, scale) that someone must **repeat exactly** when scoring new customers. Hold that thought until Module 04.
# MAGIC - 🤔 **Spot the bug:** the medians and the scaler are computed on *all* rows, *before* the split, so information from the test set leaks into training. This is called **data leakage**. It is small here, but it is a classic. Module 04 makes it impossible by design.
# MAGIC
# MAGIC ### Cell 4 — Train and score

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC > 🤔 **Predict first:** the model scores about **81.5% accuracy**. Before running the next cell, decide: is that good?
# MAGIC
# MAGIC **What to point out**
# MAGIC - 81% sounds respectable, and it would look great on a slide.
# MAGIC
# MAGIC ### Cell 5 — The accuracy trap

# COMMAND ----------

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

dumb = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"'Nobody churns' accuracy: {accuracy_score(y_test, dumb.predict(X_test)):.1%}")

proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)
print(f"ROC AUC:   {roc_auc_score(y_test, proba):.3f}")
print(f"Precision: {precision_score(y_test, pred):.1%}")
print(f"Recall:    {recall_score(y_test, pred):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - A model that predicts *"nobody ever churns"* scores **~80.5% accuracy**, within one point of ours. Our accuracy was mostly the class imbalance talking.
# MAGIC - **Recall** is the number the retention team cares about: *of the customers who actually left, how many did we flag?* At the default 0.5 threshold it is only **~15%**, so we miss 85% of the people who leave.
# MAGIC - **ROC AUC** (~0.75) measures how well the model *ranks* churners above non-churners, regardless of threshold. That makes it a good headline metric for a ranking problem like "who should we call first?"
# MAGIC
# MAGIC > 📘 **Refresher: picking a metric for imbalanced classes.** Accuracy answers *"how often am I right?"*, which is dominated by the majority class. **Precision** asks *"when I raise an alarm, how often is it real?"* (cost of wasted retention calls). **Recall** asks *"of the real churners, how many did I catch?"* (cost of lost customers). **AUC** is threshold-free ranking quality. Choose the one that matches the business cost, and never report accuracy alone on a 19/81 split.
# MAGIC
# MAGIC ### Cell 6 — Ship it… the legacy way

# COMMAND ----------

import pickle

with open("/tmp/churn_model_final_v3.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved. Emailing it to IT... 📧")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to point out**
# MAGIC - `/tmp` vanishes when the session ends. In real life this is a laptop folder or a shared drive, which is not much better.
# MAGIC - The pickle holds the model's *coefficients* only. The median values, the dummy columns, and the scaler are **not** in it, so IT has to re-implement Cell 3 perfectly.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✍️ Your turn (5 min)
# MAGIC
# MAGIC The retention team says: *"We'd rather call a few too many people than miss real churners."* Lower the decision threshold from 0.5 to **0.3** and compare precision and recall.
# MAGIC
# MAGIC
# MAGIC **Solution**

# COMMAND ----------

pred_03 = (proba >= 0.3).astype(int)
print(f"Precision @0.3: {precision_score(y_test, pred_03):.1%}")
print(f"Recall    @0.3: {recall_score(y_test, pred_03):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Recall roughly triples (~15% → ~45%) while precision drops (~60% → ~43%). The threshold is a **business decision**, not a modelling one, and it should be recorded alongside the model (Module 03 shows where).
# MAGIC
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🗺 The production questions (the map for the day)
# MAGIC
# MAGIC The model works. Now imagine it is six months later and the regulator, your manager, or a new teammate asks the questions below. With the code above, **every answer is "I don't know" or "ask the person who wrote it."** Each module of the day answers one of them.
# MAGIC
# MAGIC | # | The question nobody can answer today | Answered in |
# MAGIC |---|---|---|
# MAGIC | 1 | *Where did this data come from? Which version trained the model? Will this still work at 100× the data?* | Module 02 — Data in the Lakehouse |
# MAGIC | 2 | *Which parameters, code, and metrics produced `final_v3`? What did we try before it?* | Module 03 — Training That Remembers |
# MAGIC | 3 | *Is scoring using exactly the same preprocessing as training?* | Module 04 — Pipelines, Not Scripts |
# MAGIC | 4 | *We love scikit-learn. Do we have to rewrite everything to scale?* | Module 05 — Scale Out Your scikit-learn |
# MAGIC | 5 | *Why does every team compute "average balance" differently?* | Module 06 — Features as Shared Products |
# MAGIC | 6 | *Which model is in production right now? Can we roll back in a minute?* | Module 07 — Ship It with Governance |
# MAGIC | 7 | *Is the model still accurate? When should we retrain?* | Module 08 — Keep It Healthy |
# MAGIC | 8 | *How do we move this from my notebook to production safely?* | Module 09 — From Notebook to Production |
# MAGIC
# MAGIC > 🧑‍🏫 **Instructor tip:** ask the room to call out *their own* version of these questions before you show the table. Most groups produce six of the eight unprompted, and the list then belongs to them.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Legacy vs Databricks
# MAGIC
# MAGIC | | What we just did | What's coming |
# MAGIC |---|---|---|
# MAGIC | Data | CSV read into pandas | Versioned Delta tables with lineage |
# MAGIC | Experiments | `print()` statements | Every run logged automatically (MLflow) |
# MAGIC | Preprocessing | Manual, repeated at scoring | One pipeline object, trained and shipped together |
# MAGIC | Model artifact | `final_v3.pkl` emailed | Versioned, governed model in Unity Catalog |
# MAGIC | Monitoring | None | Drift and quality metrics, automated alerts |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Check your understanding**
# MAGIC
# MAGIC > *Our model and a model that always says "no churn" have the same accuracy. Is our model useless?*
# MAGIC >
# MAGIC > No. Accuracy hides the difference because 81% of customers don't churn anyway. The real model has an AUC well above 0.5 and catches some churners, which the dummy can never do. On imbalanced problems, judge models with recall, precision, or AUC, chosen to match the business cost.
# MAGIC
# MAGIC > *Why is a pickle file "not a deployable model" in this example?*
# MAGIC >
# MAGIC > It holds only the classifier. The preprocessing (medians, one-hot columns, scaler) lives in notebook cells that the deployer must re-create exactly. It also carries no record of the data, code, parameters, or metrics behind it.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ← Module 00 — Setup | Next: Module 02 — Data in the Lakehouse →
