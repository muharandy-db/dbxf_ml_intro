# Cheat Sheet — Legacy → Databricks

One page, one row per habit worth replacing.

| Legacy habit | Databricks way | One-liner | Module |
|---|---|---|---|
| CSV exports on a shared drive | Delta tables in Unity Catalog | `df.write.saveAsTable("customers")` | [00](../modules/00-setup.md) |
| `customers_jun_FINAL_v2.csv` | Delta history & time travel | `spark.read.option("versionAsOf", 3).table("customer_features")` | [02](../modules/02-data-in-the-lakehouse.md) |
| `MemoryError` → sample 10% | Spark DataFrames (lazy, distributed) | `spark.table("t").groupBy("k").agg(...)` | [02](../modules/02-data-in-the-lakehouse.md) |
| Features that peek into the future | Point-in-time feature logic | `.filter(F.col("days_before").between(1, 90))` | [02](../modules/02-data-in-the-lakehouse.md) |
| Results in `print()` / a spreadsheet | MLflow Tracking + autolog | `mlflow.sklearn.autolog()` | [03](../modules/03-training-that-remembers.md) |
| "Which data trained this?" | Dataset logged per run | `mlflow.log_input(mlflow.data.from_spark(df, table_name=...))` | [03](../modules/03-training-that-remembers.md) |
| Nested `for` loops for tuning | Optuna + nested runs | `mlflow.start_run(nested=True)` | [03](../modules/03-training-that-remembers.md) |
| Preprocessing re-implemented at scoring | Pipeline = recipe + model | `Pipeline([("preprocess", ColumnTransformer(...)), ("model", ...)])` | [04](../modules/04-pipelines-not-scripts.md) |
| Data too big for one machine | Spark ML pipeline | `SparkPipeline(stages=[Imputer, StringIndexer, ..., VectorAssembler, RF])` | [04](../modules/04-pipelines-not-scripts.md) |
| `for region in regions: fit()` | Many models in parallel | `df.groupBy("region").applyInPandas(train_fn, schema)` | [05](../modules/05-scale-out-sklearn.md) |
| Chunked scoring loop overnight | Distributed inference | `mlflow.pyfunc.spark_udf(spark, model_uri)` / `mapInPandas` | [05](../modules/05-scale-out-sklearn.md) |
| Each team computes `avg_amount` differently | Feature tables (PK + timestamp key) | `PRIMARY KEY (customer_id, snapshot_date TIMESERIES)` | [06](../modules/06-feature-store.md) |
| Hand-written feature joins | Point-in-time lookups | `FeatureLookup(..., timestamp_lookup_key="snapshot_date")` | [06](../modules/06-feature-store.md) |
| Rebuild features in production | Model fetches its own features | `fe.score_batch(model_uri, df_with_keys_only)` | [06](../modules/06-feature-store.md) |
| `final_v3.pkl` emailed to IT | UC Model Registry | `mlflow.register_model(uri, "cat.schema.churn_model")` | [07](../modules/07-ship-it-with-governance.md) |
| Edit a path to promote | Aliases | `client.set_registered_model_alias(name, "champion", v)` | [07](../modules/07-ship-it-with-governance.md) |
| Find the old pickle to roll back | Move the alias back | same call, previous version | [07](../modules/07-ship-it-with-governance.md) |
| Build a Flask API | Model Serving | `w.serving_endpoints.create_and_wait(...)` | [07](../modules/07-ship-it-with-governance.md) |
| Retrain on the 1st of every month | Drift detection | PSI by hand; Data Quality Monitor on the inference table | [08](../modules/08-keep-it-healthy.md) |
| Nobody notices the model degrading | Quality per window + alerts | `…_profile_metrics`, `…_drift_metrics`, SQL alert | [08](../modules/08-keep-it-healthy.md) |
| Export/import notebooks, find-and-replace `dev_db` | Asset Bundles with targets | `databricks bundle deploy -t prod` | [09](../modules/09-notebook-to-production.md) |
| Jobs clicked together in a UI | Jobs as YAML in Git | `resources/churn_job.yml` | [09](../modules/09-notebook-to-production.md) |
| Prod runs as someone's account | Service principal via CI/CD | `run_as: service_principal_name: …` | [09](../modules/09-notebook-to-production.md) |

## Picking a tool

| Data / task | Use |
|---|---|
| Training data fits in memory | scikit-learn (+ XGBoost / LightGBM) |
| Training data doesn't fit | Spark ML |
| Many independent models | scikit-learn inside `applyInPandas` |
| Small model, huge table to score | scikit-learn + `spark_udf` / `mapInPandas` |
| Real-time predictions | Model Serving (+ online feature store) |
