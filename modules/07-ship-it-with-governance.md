# Module 07 — Ship It with Governance (Unity Catalog Model Registry)

> 🎯 **Learning objectives**
> - Register models in **Unity Catalog** and understand **versions vs aliases**
> - Run a **champion / challenger** comparison on the untouched test set and promote the winner
> - Score in batch by alias, roll back in one line, and (demo) expose the model as a REST endpoint
>
> ⏱ **Time:** 30 minutes  ·  Answers production question **#6**: *Which model is in production right now? Can we roll back in a minute?*

---

## 🏚 The legacy way

Production runs `churn_model_final_v3.pkl`. Or is it `v3_fixed.pkl`? The file on the scoring server was overwritten in March, and nobody kept the old one. To ship a new model, someone edits a path in a cron script on a Friday evening. If it goes wrong, "rolling back" means finding the old pickle in somebody's email.

---

## 💡 The idea: editions and a "staff pick" sticker

A library keeps every **edition** of a book, and editions never change once printed. On the front table sits a sticker: **"Staff pick."** Moving the sticker from the 2nd edition to the 3rd changes what visitors pick up, without reprinting anything. Moving it back is just as easy.

The **Unity Catalog Model Registry** works the same way:

- A **registered model** (`ml_workshop.<you>.churn_model`) is the book. It lives in a schema next to your tables, with the same permissions, lineage, and audit.
- A **version** (1, 2, 3, …) is an edition. It's **immutable** and linked to the exact MLflow run, code, and data that produced it.
- An **alias** (`@champion`, `@challenger`) is the sticker: a movable name pointing to one version.

Consumers never hard-code a version. Batch jobs, dashboards, and endpoints all load **`models:/ml_workshop.<you>.churn_model@champion`**. Promotion moves the alias, and **rollback** moves it back. Neither requires redeploying code.

The catch is that the registry only records decisions; it doesn't make them. Someone still has to decide whether the challenger is better. That decision should be **evidence-based and recorded**, which is what this module does on the test set we've kept untouched since Module 03.

---

## 🛠 Hands-on

Create a notebook `07_ship_it_with_governance`.

### Cell 0 — Libraries

```python
%pip install -q "mlflow>=3.1" scikit-learn==1.6.1
```

```python
dbutils.library.restartPython()
```

### Cell 1 — Config + the same data split as Modules 03/04

```python
import re, mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

CATALOG = "ml_workshop"
user = spark.sql("SELECT current_user()").first()[0]
SCHEMA = re.sub(r"[^a-z0-9_]", "_", user.split("@")[0].lower())
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/churn_workshop")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.churn_model"
client = MlflowClient()

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"

pdf = spark.table("customer_features").filter("snapshot_date = '2026-06-30'").orderBy("customer_id").toPandas()
X, y = pdf[NUMERIC + CATEGORICAL], pdf[LABEL]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
```

**What to point out**
- `X_test` is the held-out set we set aside in Module 03 and never looked at. This is the moment it's for.

### Cell 2 — Register the Module 04 pipeline as version 1, the champion

```python
candidate = mlflow.search_runs(filter_string="tags.candidate = 'true'",
                               order_by=["start_time DESC"], max_results=1).iloc[0]
v1 = mlflow.register_model(f"runs:/{candidate.run_id}/model", MODEL_NAME)

client.update_registered_model(MODEL_NAME, description=
    "Predicts probability that a retail customer churns in the next quarter. Owner: retention analytics.")
client.set_registered_model_alias(MODEL_NAME, "champion", v1.version)
client.set_model_version_tag(MODEL_NAME, v1.version, "approval_status", "approved")
print(f"{MODEL_NAME} v{v1.version} is @champion")
```

**What to point out**
- `register_model` creates the model in Unity Catalog on the first call and adds a new **version** on every later call.
- Open **Catalog → your schema → Models → churn_model**. You'll see the version, its alias, its tags, and a link back to the **run that produced it**, including the dataset version from Module 03.
- Models are securable objects like tables: `GRANT EXECUTE ON MODEL ... TO <group>` controls who can load it.

### Cell 3 — Train a challenger with the tuned parameters from Module 03

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from mlflow.models import infer_signature

tuning = mlflow.search_runs(filter_string="attributes.run_name = 'hist_gbm_tuning'",
                            order_by=["start_time DESC"], max_results=1).iloc[0]
best = {"learning_rate": float(tuning["params.best_learning_rate"]),
        "max_depth": int(tuning["params.best_max_depth"]),
        "max_iter": int(tuning["params.best_max_iter"]),
        "l2_regularization": float(tuning["params.best_l2_regularization"])}

challenger = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
    ("model", HistGradientBoostingClassifier(**best, random_state=42)),
]).fit(X_train, y_train)

with mlflow.start_run(run_name="challenger_tuned_pipeline"):
    mlflow.set_tag("module", "07")
    mlflow.log_params(best)
    info = mlflow.sklearn.log_model(challenger, name="model", input_example=X_train.head(3),
                                    signature=infer_signature(X_train, challenger.predict(X_train)),
                                    registered_model_name=MODEL_NAME)
v2 = info.registered_model_version
client.set_registered_model_alias(MODEL_NAME, "challenger", v2)
print(f"v{v2} is @challenger with {best}")
```

**What to point out**
- The best hyperparameters come **from the experiment log** (Module 03), not from someone's memory. That's tracking paying off.
- `registered_model_name=` logs and registers in one step, which creates **version 2**.

### Cell 4 — Champion vs challenger on the untouched test set

```python
from sklearn.metrics import roc_auc_score, recall_score

def evaluate(alias):
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    proba = model.predict_proba(X_test)[:, 1]
    return {"alias": alias, "test_auc": roc_auc_score(y_test, proba),
            "test_recall@0.3": recall_score(y_test, proba >= 0.3)}

results = pd.DataFrame([evaluate("champion"), evaluate("challenger")])
display(results)
```

**What to point out**
- Both models are loaded **by alias**, exactly as a production job would.
- This is the one honest comparison: neither model was tuned or chosen using this data.

### Cell 5 — Promote (or don't) and record why

```python
champ, chall = results.set_index("alias").loc[["champion", "challenger"], "test_auc"]

if chall > champ:
    old = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    client.set_registered_model_alias(MODEL_NAME, "champion", v2)
    client.set_model_version_tag(MODEL_NAME, v2, "approval_status", "approved")
    client.set_model_version_tag(MODEL_NAME, v2, "promotion_note",
                                 f"test_auc {chall:.4f} > {champ:.4f} (v{old})")
    print(f"🏆 Promoted v{v2} to @champion")
else:
    print(f"Champion stays: {champ:.4f} >= {chall:.4f}")
```

**What to point out**
- Promotion is **one alias move**. Nothing is copied or redeployed, and every consumer of `@champion` picks up the new version the next time it loads.
- The *reason* for promotion is stored as a tag on the version, which gives an audit trail written by the same code that made the decision.
- In Module 09 this exact logic runs inside a scheduled job, so promotion happens without anyone opening a notebook.

### Cell 6 — Batch scoring by alias

```python
from pyspark.sql import functions as F

champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
sk_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

def score_batches(batches):
    for pdf in batches:
        pdf["p_churn"] = sk_model.predict_proba(pdf[NUMERIC + CATEGORICAL])[:, 1]
        yield pdf[["customer_id", "snapshot_date", "p_churn"]]

(spark.table("customer_features").filter("snapshot_date = '2026-09-30'")
    .mapInPandas(score_batches, "customer_id string, snapshot_date date, p_churn double")
    .withColumn("model_version", F.lit(champion_version))
    .withColumn("scored_at", F.current_timestamp())
    .write.mode("overwrite").saveAsTable("churn_scores"))

display(spark.table("churn_scores").orderBy(F.desc("p_churn")).limit(10))
```

**What to point out**
- Every score row carries the **model version** that produced it. When someone asks *"why was this customer flagged in September?"*, you can trace it to the run, the code, and the data.
- Open the `churn_scores` table's **Lineage** tab: `customer_features` → `churn_model` → `churn_scores`, captured automatically.

### Cell 7 — (Instructor demo) Real-time serving

```python
DEPLOY_SERVING = False     # 👈 instructor only: creating endpoints takes ~10 min and costs money

if DEPLOY_SERVING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    w = WorkspaceClient()
    endpoint = f"churn-{SCHEMA}".replace("_", "-")
    w.serving_endpoints.create_and_wait(name=endpoint, config=EndpointCoreConfigInput(
        served_entities=[ServedEntityInput(entity_name=MODEL_NAME, entity_version=champion_version,
                                           workload_size="Small", scale_to_zero_enabled=True)]))
    response = w.serving_endpoints.query(name=endpoint,
        dataframe_records=X_test.head(2).astype(object).where(X_test.head(2).notna(), None).to_dict("records"))
    print(response.predictions)
```

**What to point out**
- The same registered model becomes an **autoscaling REST API**, with no Flask app or Docker image to build. The signature from Module 04 validates every request.
- `scale_to_zero_enabled=True` means you pay nothing while nobody calls it.
- Model Serving also logs every request and response to an **inference table** if you enable it, which feeds directly into monitoring (Module 08).

---

## ✍️ Your turn (3 min)

The call centre reports something odd with this morning's scores. **Roll back** `@champion` to version 1, then confirm which version the alias points to.

<details>
<summary>Solution</summary>

```python
client.set_registered_model_alias(MODEL_NAME, "champion", v1.version)
print(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
```

That's the whole rollback: one line, seconds, no deployment. Move it forward again once the issue is understood:

```python
client.set_registered_model_alias(MODEL_NAME, "champion", v2)
```

</details>

---

## ⚖️ Legacy vs Databricks

| | Legacy | Unity Catalog Model Registry |
|---|---|---|
| "Which model is live?" | Check the path in a cron script | `@champion` alias, visible in the UI |
| Versions | Overwritten files | Immutable versions linked to run, code, and data |
| Promotion | Edit a path, redeploy | Move an alias, based on recorded evidence |
| Rollback | Find the old pickle in email | Move the alias back: one line |
| Access control | Whoever can reach the server | `GRANT EXECUTE ON MODEL`, audited |
| Real-time | Build and host an API | Model Serving from the registry, scale to zero |

---

**Check your understanding**

> *Why should a batch job load `models:/…/churn_model@champion` instead of `models:/…/churn_model/2`?*
>
> A version number hard-codes one edition. Promotion or rollback would then need a code change and redeploy. An alias is resolved at load time, so moving `@champion` changes what every consumer uses, instantly and consistently.

> *We compared champion and challenger on the test set. Why not on the validation set?*
>
> The validation set was used to *choose* hyperparameters (Module 03), so both models are slightly optimistic on it. The test set has never influenced any decision, which makes it the only unbiased basis for a promotion decision.

---

[← Module 06 — Feature Store](06-feature-store.md) | [Next: Module 08 — Keep It Healthy →](08-keep-it-healthy.md)
