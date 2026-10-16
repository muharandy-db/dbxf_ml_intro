# Module 04 — Pipelines, Not Scripts (and scikit-learn vs Spark ML)

> 🎯 **Learning objectives**
> - Reproduce a real **training/serving skew** bug, then make it impossible with a **pipeline**
> - Understand the **Estimator / Transformer** idea that scikit-learn and Spark ML share
> - Know **how Spark ML trains on distributed data**, and when to choose it over scikit-learn
>
> ⏱ **Time:** 45 minutes  ·  Answers production question **#3**: *Is scoring using exactly the same preprocessing as training?*

---

## 🏚 The legacy way

Look back at Module 01: impute, one-hot encode, scale, then train. Each step is a separate cell. When IT scores next month's customers, they rewrite those steps from a Word document. They fill missing values with *this month's* median instead of the training median. Their one-hot encoder produces different columns because one region is missing from the batch. The model doesn't crash. It quietly returns wrong scores.

This is called **training/serving skew**: the model sees data prepared one way in training and another way in production. It is one of the most common and least visible failures in production ML.

---

## 💡 The idea: ship the recipe, not just the cake

Module 01's pickle was the cake. The *recipe* that turns raw ingredients into what the model expects (medians, category lists, scaling factors) stayed in a notebook, so someone had to bake it again from memory.

A **pipeline** packages the recipe *with* the cake. It's built on one abstraction that scikit-learn and Spark ML share:

- An **Estimator** *learns* something from data when you call `fit`: an imputer learns medians, an encoder learns the list of categories, a classifier learns its weights.
- A **Transformer** *applies* what was learned when you call `transform` or `predict`, without learning anything new.

A pipeline chains these steps. Calling `fit` on the pipeline fits each step **on the training data only**, in order, and returns one fitted object. Calling `predict` replays every step with the values learned during training. The medians, the category list, and the model weights travel together. Leakage (fitting on test data) and skew (re-implementing steps differently) both become *structurally impossible*, not just discouraged.

### How Spark ML trains on data spread across machines

scikit-learn assumes the whole dataset fits in one machine's memory. Spark ML's algorithms are *re-designed* so that each executor works on its own partition and only small summaries travel over the network.

Take logistic regression, which learns by gradient descent. The gradient of the loss over the full dataset is a sum over rows, and a sum over rows splits cleanly into a sum over partitions:

$$\nabla L(w) \;=\; \sum_{i=1}^{N} \nabla \ell(w; x_i, y_i) \;=\; \sum_{p=1}^{P} \underbrace{\sum_{i \in p} \nabla \ell(w; x_i, y_i)}_{\text{computed on executor } p}$$

Reading it: $w$ is the current set of model weights, $\ell$ is the error on one customer, and $\nabla$ is "which direction to nudge each weight to reduce that error." The left side asks for the nudge summed over all $N$ customers. The right side groups those customers by partition $p$. Each executor computes the sum for *its own* customers, which is just a vector the size of $w$, and sends that small vector to the driver. The driver adds the $P$ vectors, takes one step, and broadcasts the new weights. **The data never moves; only gradients do.**

Tree models use the same trick. Each executor builds a histogram of feature values for its partition, the driver merges the histograms to pick the best split, and the process repeats. In both cases a census is the right picture: each district counts its own residents, and the capital only adds up the totals.

**So why ever use scikit-learn?** Because the trick has a price. Each step needs a network round-trip, and only algorithms that decompose this way get a distributed version. Spark ML has the classics (linear models, trees, forests, gradient boosting, k-means, ALS), while scikit-learn has hundreds. When the data fits on one machine, which it does more often than people think, scikit-learn is faster and richer.

The practical takeaway is worth stating plainly: **the right tool is decided by data size, not by fashion.** Use scikit-learn up to a few million rows, Spark ML beyond that, and Module 05 covers the case where you want scikit-learn's algorithms on *many* datasets at once.

---

## 🛠 Hands-on

Create a notebook `04_pipelines_not_scripts`.

> ⚠️ **Check your serverless environment version.** Spark ML runs on serverless from **environment version 4** onwards. If Cell 6 fails with `Py4JSecurityException: ... is not whitelisted`, open the **Environment** panel, set version **4**, and re-run from Cell 0.

### Cell 0 — Libraries

```python
%pip install -q "mlflow[databricks]>=3.14" scikit-learn==1.6.1
```

```python
dbutils.library.restartPython()
```

### Cell 1 — Config

```python
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
MODEL_NAME = f"{CATALOG}.{SCHEMA}.churn_model"
```

### Cell 2 — Load raw features (no manual preprocessing this time)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"

train_sdf = spark.table("customer_features").filter("snapshot_date = '2026-06-30'")
pdf = train_sdf.orderBy("customer_id").toPandas()
X, y = pdf[NUMERIC + CATEGORICAL], pdf[LABEL]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
```

**What to point out**
- This is the same split as Module 03 (same seed, same ordering), so results are comparable across modules.
- `X` still contains the raw `branch_region` strings and missing values, because the pipeline will handle them.

### Cell 3 — Reproduce the skew bug 🐛

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Legacy style: preprocess by hand, then train
legacy_model = HistGradientBoostingClassifier(random_state=42).fit(
    pd.get_dummies(X_train, columns=CATEGORICAL, dtype=int), y_train)

# Next month IT scores a batch... that happens to contain only Medan customers
new_batch = X_test[X_test["branch_region"] == "Medan"].head(5)
try:
    legacy_model.predict_proba(pd.get_dummies(new_batch, columns=CATEGORICAL, dtype=int))
except ValueError as e:
    print("💥", str(e).splitlines()[0])
```

> 🤔 **Predict first:** what will `get_dummies` produce for a batch where every row is from Medan?

**What to point out**
- `get_dummies` creates columns from **whatever categories appear in this batch**. Training had five region columns; this batch produces one. The model refuses to score.
- Here we got a loud error, which is the *lucky* outcome. With a different mix of regions and a model that doesn't check column names, the columns can shift silently, and the predictions are simply wrong.

### Cell 4 — The fix: one pipeline object

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42)),
])
pipe
```

**What to point out**
- The last line renders the pipeline as an interactive **diagram**. Click a box to see its settings.
- `SimpleImputer` learns medians on **training data only** (that fixes Module 01's leakage). `OneHotEncoder` learns the five regions once and ignores unseen ones, which fixes the skew bug.
- The pipeline *is* a model: it has `fit`, `predict`, and `predict_proba`, so everything that accepts a model accepts the pipeline.

### Cell 5 — Train, log, and score the "impossible" batch

```python
from sklearn.metrics import roc_auc_score
from mlflow.models import infer_signature

with mlflow.start_run(run_name="sklearn_pipeline") as run:
    mlflow.set_tags({"module": "04", "candidate": "true"})
    pipe.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    mlflow.log_metric("val_auc", val_auc)
    mlflow.sklearn.log_model(
        pipe, name="model",
        input_example=X_train.head(3),
        signature=infer_signature(X_train, pipe.predict(X_train)))

print(f"val_auc = {val_auc:.3f}")
print(pipe.predict_proba(new_batch)[:, 1].round(3))      # the Medan-only batch now just works
```

**What to point out**
- The logged model contains preprocessing **and** classifier. Whoever loads it, whether a batch job, an API endpoint, or a colleague, passes **raw** columns and gets correct predictions.
- The **signature** records the expected input columns and types. Unity Catalog requires it to register a model, and serving uses it to validate requests.
- Tag `candidate=true` is how Module 07 will find this run to promote it.

### Cell 6 — The same idea in Spark ML

```python
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder as SparkOHE, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

train_s, test_s = train_sdf.randomSplit([0.8, 0.2], seed=42)

IMPUTED = [f"{c}_imp" for c in NUMERIC]
spark_pipe = SparkPipeline(stages=[
    Imputer(inputCols=NUMERIC, outputCols=IMPUTED, strategy="median"),
    StringIndexer(inputCol="branch_region", outputCol="region_idx", handleInvalid="keep"),
    SparkOHE(inputCols=["region_idx"], outputCols=["region_ohe"]),
    VectorAssembler(inputCols=IMPUTED + ["region_ohe"], outputCol="features"),
    RandomForestClassifier(labelCol=LABEL, featuresCol="features", numTrees=100, maxDepth=6, seed=42),
])

spark_model = spark_pipe.fit(train_s)                      # each stage fitted in order, on the cluster
predictions = spark_model.transform(test_s)
auc = BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC").evaluate(predictions)
print(f"Spark ML test AUC: {auc:.3f}")
```

**What to point out**
- It's the **same recipe** with different names: `Imputer` ≈ `SimpleImputer`, `StringIndexer` + `OneHotEncoder` ≈ `OneHotEncoder`, and the classifier is the last stage.
- **`VectorAssembler`** is the one genuinely new idea. Spark ML models expect *all* features packed into a single vector column (`features`). It's how the distributed algorithms receive one compact row per customer.
- `fit` returns a **`PipelineModel`**, a chain of fitted Transformers, exactly like the fitted sklearn `Pipeline`.
- Nothing was collected to the driver. This exact cell would train on 100 million customers by adding executors.

### Cell 7 — Persist and reload the Spark pipeline

```python
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

spark.sql("CREATE VOLUME IF NOT EXISTS models")
path = f"/Volumes/{CATALOG}/{SCHEMA}/models/churn_spark_pipeline"
spark_model.write().overwrite().save(path)

reloaded = PipelineModel.load(path)
display(reloaded.transform(test_s)
        .select("customer_id", LABEL,
                F.round(vector_to_array("probability")[1], 3).alias("p_churn"))
        .orderBy(F.desc("p_churn")))

with mlflow.start_run(run_name="spark_ml_pipeline"):
    mlflow.set_tag("module", "04")
    mlflow.log_params({"model": "RandomForestClassifier", "numTrees": 100, "maxDepth": 6})
    mlflow.log_metric("test_auc", auc)
    mlflow.log_param("pipeline_path", path)
```

**What to point out**
- The fitted pipeline is saved to a **volume**: governed storage, not a laptop folder. Reload it anywhere and the same preprocessing replays.
- `probability` is a vector `[p(stay), p(churn)]`. `vector_to_array(...)[1]` pulls out the churn probability for ranking.
- We still log the run to MLflow, so Spark ML experiments sit right next to scikit-learn ones.

---

## 🧭 Decision guide: scikit-learn or Spark ML?

| Situation | Reach for | Why |
|---|---|---|
| Training data fits in memory on one machine (≲ a few million rows) | **scikit-learn** (+ XGBoost/LightGBM) | Fastest iteration, biggest algorithm library |
| Training data is too big for one machine | **Spark ML** | Algorithms designed to run across partitions |
| Many *independent* models (per region, per product, per branch) | **scikit-learn inside `applyInPandas`** | Spark distributes the *models*, scikit-learn trains each one, [Module 05](05-scale-out-sklearn.md) |
| One small model, scoring a *huge* table | **scikit-learn + `spark_udf`** | Train small, score in parallel, [Module 05](05-scale-out-sklearn.md) |
| Deep learning | PyTorch / TensorFlow with **TorchDistributor** or Serverless GPU | Out of scope today |

For our 10k-row churn problem, **scikit-learn wins**, so the scikit-learn pipeline from Cell 5 is the model we carry forward.

---

## ✍️ Your turn (5 min)

Pipelines are modular. Replace the random forest in Cell 6 with a **`GBTClassifier`** (gradient-boosted trees in Spark ML). Change only the last stage and compare the AUC.

<details>
<summary>Solution</summary>

```python
from pyspark.ml.classification import GBTClassifier

gbt_pipe = SparkPipeline(stages=spark_pipe.getStages()[:-1] + [
    GBTClassifier(labelCol=LABEL, featuresCol="features", maxIter=50, maxDepth=4, seed=42)])
gbt_auc = BinaryClassificationEvaluator(labelCol=LABEL).evaluate(gbt_pipe.fit(train_s).transform(test_s))
print(f"GBT test AUC: {gbt_auc:.3f}")
```

Only one stage changed. Preprocessing is reused untouched, which is exactly why pipelines make experimentation cheap.

</details>

---

## ⚖️ Legacy vs Databricks

| | Legacy | Databricks |
|---|---|---|
| Preprocessing | Separate cells, re-implemented at scoring | Inside the pipeline, fitted once, shipped with the model |
| Leakage | Easy to fit on all data by accident | `fit` on train only, by construction |
| Unseen categories | Crash or silent column shift | `handle_unknown="ignore"` / `handleInvalid="keep"` |
| Bigger data | Rewrite, or sample | Same pipeline concept in Spark ML, runs on a cluster |
| What gets deployed | A model *and* a Word document | One pipeline artifact with a signature |

---

**Check your understanding**

> *In a pipeline, why is it impossible for the test set's median to leak into training?*
>
> `pipeline.fit(X_train, y_train)` is the only place anything is learned, and it only sees the training rows. On the test set we call `predict`, which *replays* the medians learned from training without recomputing them.

> *Spark ML trains on data spread across many machines. What actually travels over the network during training?*
>
> Small summaries, not the data. For linear models, each executor sends a gradient vector for its partition and the driver adds them. For trees, each executor sends feature histograms. The rows stay where they are.

---

[← Module 03 — Training That Remembers](03-training-that-remembers.md) | [Next: Module 05 — Scale Out Your scikit-learn →](05-scale-out-sklearn.md)
