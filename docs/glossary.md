# Glossary

Every term from the workshop in one or two sentences, grouped by where it first appears.

## Platform & data (Modules 00–02)

- **Unity Catalog (UC)**: Databricks' governance layer. One place for permissions, lineage, audit, and discovery across tables, volumes, models, and functions. Names have three levels: `catalog.schema.object`.
- **Catalog / Schema**: the first two levels of the namespace. Think *building* and *shelf*.
- **Volume**: a governed folder for *files* (CSVs, images, artifacts), addressed as `/Volumes/<catalog>/<schema>/<volume>/…`.
- **Delta table**: the default table format. Parquet files plus a transaction log, giving ACID writes, schema enforcement, history, and time travel.
- **Time travel**: reading a Delta table as of an earlier version or timestamp (`versionAsOf`, `timestampAsOf`).
- **Lineage**: the automatically captured graph of which tables, notebooks, jobs, and models read from and write to which.
- **Git folder**: a Git repository cloned into the workspace, with branches, commits, and pulls from the UI.
- **Serverless compute**: compute that Databricks provisions and scales for you; there are no clusters to configure.
- **Driver / Executor**: in Spark, the driver plans the work and executors run it in parallel, each on its own partitions.
- **Partition**: a chunk of a distributed dataset processed by one task.
- **Lazy evaluation**: Spark transformations only build a plan; an **action** (`display`, `count`, `write`, `toPandas`) runs it.
- **Shuffle**: moving rows between executors so rows with the same key end up together (for example, for `groupBy` or `join`).
- **Point-in-time correctness**: features for a prediction at time *t* use only information available at *t*.
- **Data leakage**: information from outside the training data (the test set, or the future) influences the model, making evaluation over-optimistic.

## Modelling & tracking (Modules 01–05)

- **Accuracy trap**: on imbalanced data, a model that always predicts the majority class gets high accuracy while being useless.
- **Precision / Recall**: precision is how many flagged customers really churn; recall is how many churners we flagged.
- **ROC AUC**: the probability that the model ranks a random churner above a random non-churner. 0.5 is random and 1.0 is perfect.
- **Decision threshold**: the probability above which we act (flag a customer). It's a business choice, not a model property.
- **Train / validation / test split**: fit on train, choose and tune on validation, and report once on the untouched test set.
- **MLflow experiment**: a container for related runs.
- **Run**: one training attempt, with parameters, metrics, artifacts, model, and dataset.
- **Autologging**: `mlflow.<flavor>.autolog()` records params, metrics, and models for every `fit` automatically.
- **Artifact**: any file attached to a run (plot, report, model files).
- **Nested run**: a child run under a parent, for example each Optuna trial under a tuning run.
- **Signature**: the declared input and output schema of a logged model. Unity Catalog requires one.
- **Estimator**: an object that *learns* from data with `fit` (imputer, encoder, classifier).
- **Transformer**: an object that *applies* what was learned, with `transform` or `predict`.
- **Pipeline**: a chain of estimators and transformers fitted and applied as one object, so preprocessing ships with the model.
- **Training/serving skew**: the model sees data prepared differently in production than in training.
- **`VectorAssembler`**: packs many columns into the single `features` vector column that Spark ML models expect.
- **`PipelineModel`**: the fitted result of a Spark ML `Pipeline`.
- **`applyInPandas`**: runs a pandas → pandas function on each group of a Spark DataFrame in parallel; you declare the output schema.
- **`mapInPandas`**: runs a function over batches (iterators of pandas DataFrames) of a Spark DataFrame.
- **`spark_udf`**: turns an MLflow model into a Spark function for distributed scoring.

## Features, registry & serving (Modules 06–07)

- **Feature table**: a Unity Catalog Delta table with a primary key, used as a shared, discoverable source of features.
- **Timestamp key (`TIMESERIES`)**: the primary-key column used for point-in-time lookups.
- **Spine**: the DataFrame of keys, timestamps, and labels to which features are joined.
- **`FeatureLookup`**: declares which features to fetch from which table, by which keys.
- **`fe.log_model` / `fe.score_batch`**: log a model with its feature lookup spec, then score by providing keys only.
- **Online store**: a low-latency copy of feature tables for real-time serving.
- **Registered model**: a named model in Unity Catalog (`catalog.schema.model`) that holds versions.
- **Model version**: an immutable, numbered snapshot of a registered model, linked to its run.
- **Alias**: a movable name (`@champion`) pointing to one version. Used for promotion and rollback.
- **Champion / challenger**: the current production model, and a candidate competing to replace it.
- **Model Serving**: managed, autoscaling REST endpoints for registered models.

## Monitoring & production (Modules 08–09)

- **Data drift**: the distribution of the *inputs* changes.
- **Concept drift**: the relationship between inputs and outcome changes.
- **PSI (Population Stability Index)**: a drift score summing $(a-e)\ln(a/e)$ over bins. Below 0.1 is stable; above 0.25 is a major shift.
- **Jensen–Shannon distance**: a symmetric distance between two distributions, from 0 (identical) to 1.
- **Inference table**: a log of model inputs, outputs, version, timestamp, and (later) labels.
- **Data Quality Monitor (data profiling)**: computes profile and drift metrics for a table per time window, with tables and a dashboard.
- **Baseline table**: the reference data that a monitor compares each window against.
- **Job**: a scheduled or triggered workflow of tasks on Databricks.
- **Databricks Asset Bundle (DAB)**: a Git folder of code + YAML resource definitions, deployed per target with the CLI or UI.
- **Target**: a named deployment environment in a bundle (`dev`, `staging`, `prod`) with its own variables.
- **`mode: development`**: a bundle target mode that prefixes resources with `[dev <user>]` and pauses schedules.
- **Service principal**: a non-human identity used by CI/CD and production jobs.
- **MLflow Tracing**: step-by-step recording of GenAI and agent executions (LLM calls, tools, retrieval).
