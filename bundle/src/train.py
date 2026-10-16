"""Retrain the churn pipeline and promote it if it beats the current champion.

This is Modules 04 + 07 condensed into one parameterised script, so the same
code runs unchanged in dev, staging and prod; only the arguments differ.
"""

import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["age", "tenure_months", "num_products", "balance", "credit_score",
           "has_credit_card", "is_active_member", "estimated_salary",
           "digital_logins_30d", "complaints_90d", "txn_count_30d", "txn_count_90d",
           "avg_amount_90d", "mobile_share_90d", "days_since_last_txn"]
CATEGORICAL = ["branch_region"]
LABEL = "churned"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--model-name", default="churn_model")
    p.add_argument("--experiment", required=True)
    p.add_argument("--training-snapshot", default="2026-06-30")
    return p.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()
    model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)
    client = MlflowClient()

    pdf = (spark.table(f"{args.catalog}.{args.schema}.customer_features")
           .filter(f"snapshot_date = '{args.training_snapshot}'")
           .orderBy("customer_id").toPandas())
    X, y = pdf[NUMERIC + CATEGORICAL], pdf[LABEL]
    # Same split as the workshop modules, so the champion is never scored on rows it trained on.
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    pipe = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42)),
    ]).fit(X_train, y_train)
    new_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    with mlflow.start_run(run_name="scheduled_retrain"):
        mlflow.set_tag("trigger", "job")
        mlflow.log_metric("test_auc", new_auc)
        info = mlflow.sklearn.log_model(
            pipe, name="model", input_example=X_train.head(3),
            signature=infer_signature(X_train, pipe.predict(X_train)),
            registered_model_name=model_name)
    new_version = info.registered_model_version
    client.set_registered_model_alias(model_name, "challenger", new_version)

    # Champion / challenger on the same test split (Module 07).
    try:
        champion = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
        champ_auc = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])
    except Exception:
        champ_auc = float("-inf")       # first run in this environment: no champion yet

    if new_auc >= champ_auc:
        client.set_registered_model_alias(model_name, "champion", new_version)
        client.set_model_version_tag(model_name, new_version, "promotion_note",
                                     f"test_auc {new_auc:.4f} >= champion {champ_auc:.4f}")
        print(f"Promoted v{new_version} to @champion (test_auc {new_auc:.4f})")
    else:
        print(f"Kept champion ({champ_auc:.4f}); v{new_version} stays @challenger ({new_auc:.4f})")


if __name__ == "__main__":
    main()
