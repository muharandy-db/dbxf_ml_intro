"""Score the latest customer snapshot with whatever model is @champion."""

import argparse

import mlflow
from mlflow import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--model-name", default="churn_model")
    return p.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    version = MlflowClient().get_model_version_by_alias(model_name, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    def score_batches(batches):
        for pdf in batches:
            pdf["p_churn"] = model.predict_proba(pdf[NUMERIC + CATEGORICAL])[:, 1]
            yield pdf[["customer_id", "snapshot_date", "p_churn"]]

    features = spark.table(f"{args.catalog}.{args.schema}.customer_features")
    latest = features.agg(F.max("snapshot_date")).first()[0]
    (features.filter(F.col("snapshot_date") == latest)
        .mapInPandas(score_batches, "customer_id string, snapshot_date date, p_churn double")
        .withColumn("model_version", F.lit(version))
        .withColumn("scored_at", F.current_timestamp())
        .write.mode("overwrite").saveAsTable(f"{args.catalog}.{args.schema}.churn_scores"))
    print(f"Scored snapshot {latest} with {model_name} v{version}")


if __name__ == "__main__":
    main()
