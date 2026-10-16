"""Generate the synthetic bank-churn dataset used throughout the workshop.

Usage:
    python scripts/generate_data.py            # writes CSVs into ./data

Three files are produced:
    customers.csv               ~10k customers, snapshot 2026-06-30, labelled (churned)
    transactions.csv            90 days of transactions before each snapshot (Q2 and Q3 2026)
    customers_next_quarter.csv  ~3k of the same customers re-snapshotted on 2026-09-30,
                                with deliberately shifted distributions (data drift) and
                                a changed churn relationship (concept drift)

Everything is seeded, so the files are identical on every run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 10_000
N_NEXT_QUARTER = 3_000
SNAPSHOT = pd.Timestamp("2026-06-30")
NEXT_SNAPSHOT = pd.Timestamp("2026-09-30")
REGIONS = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar"]
REGION_WEIGHTS = [0.40, 0.20, 0.17, 0.13, 0.10]

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_profiles(rng, n, drift=False):
    """Customer profile columns. `drift=True` shifts the distributions."""
    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    age = np.clip(rng.normal(41, 12, n), 18, 80).round().astype(int)
    tenure = np.clip(rng.exponential(48, n), 0, 300).round().astype(int)
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.45, 0.40, 0.11, 0.04])
    has_balance = rng.random(n) > 0.22
    balance = np.where(has_balance, rng.lognormal(3.6, 1.0, n), 0.0)  # IDR millions
    credit_score = np.clip(rng.normal(650, 90, n), 300, 850).round()
    has_credit_card = (rng.random(n) < 0.68).astype(int)
    is_active = (rng.random(n) < 0.52).astype(int)
    salary = rng.lognormal(2.3, 0.55, n).round(2)  # monthly, IDR millions
    logins = rng.poisson(np.where(age < 35, 14, np.where(age < 55, 9, 4)))
    complaints = rng.poisson(0.35, n)

    if drift:
        # A mobile-app relaunch pushes digital usage up; a rate cut by a competitor
        # pulls balances down (applied to existing balances in main()); a service outage triggers a wave of complaints.
        logins = rng.poisson(np.where(age < 35, 24, np.where(age < 55, 16, 8)))
        complaints = rng.poisson(0.9, n)

    return pd.DataFrame(
        {
            "branch_region": region,
            "age": age,
            "tenure_months": tenure,
            "num_products": num_products,
            "balance": balance.round(2),
            "credit_score": credit_score,
            "has_credit_card": has_credit_card,
            "is_active_member": is_active,
            "estimated_salary": salary,
            "digital_logins_30d": logins,
            "complaints_90d": complaints,
        }
    )


def churn_logit(df, concept_drift=False):
    # Concept drift: a digital-only competitor launches, so heavy app users become
    # *more* likely to leave instead of less, and complaints bite harder.
    complaint_weight = 1.1 if concept_drift else 0.75
    login_weight = 0.11 if concept_drift else -0.045
    offset = -2.3 if concept_drift else 0.0
    return (
        -0.95
        + offset
        + complaint_weight * df["complaints_90d"]
        - 0.95 * df["is_active_member"]
        - 0.012 * df["tenure_months"]
        + 0.035 * (df["age"] - 41)
        + np.select([df["num_products"] == 1, df["num_products"] >= 3], [0.55, 1.6], -0.6)
        + login_weight * df["digital_logins_30d"]
        + 0.25 * (df["balance"] == 0)
        + 0.004 * (650 - df["credit_score"].fillna(650))
        + np.where(df["branch_region"] == "Medan", 0.35, 0.0)
    )


def make_transactions(rng, customers, snapshot):
    """90 days of transactions up to `snapshot`. Churners transact less, especially recently."""
    rows = []
    rate = (
        3.0
        + 4.0 * customers["is_active_member"]
        + 0.15 * customers["digital_logins_30d"]
        - 1.0 * customers["churned"]
    ).clip(lower=0.3)
    counts = rng.poisson(rate.to_numpy() * 0.9)
    for cid, n, churned, logins in zip(
        customers["customer_id"], counts, customers["churned"], customers["digital_logins_30d"]
    ):
        if n == 0:
            continue
        # Churners' activity is skewed towards the start of the 90-day window.
        u = rng.beta(1.5, 1.0, n) if churned else rng.random(n)
        days_before = np.minimum(u * 90, 89).astype(int)
        ts = snapshot - pd.to_timedelta(days_before, unit="D") - pd.to_timedelta(
            rng.integers(0, 86_400, n), unit="s"
        )
        p_mobile = min(0.15 + logins / 40, 0.8)
        channel = rng.choice(
            ["mobile", "web", "atm", "branch"],
            size=n,
            p=[p_mobile, 0.12, (1 - p_mobile - 0.12) * 0.65, (1 - p_mobile - 0.12) * 0.35],
        )
        amount = rng.lognormal(-0.3, 1.1, n).round(3)  # IDR millions
        rows.append(pd.DataFrame({"customer_id": cid, "txn_ts": ts, "amount": amount, "channel": channel}))
    return pd.concat(rows, ignore_index=True)


def add_missing(rng, df):
    df = df.copy()
    df.loc[rng.random(len(df)) < 0.03, "credit_score"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "estimated_salary"] = np.nan
    return df


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # --- customers (training snapshot) ---
    customers = make_profiles(rng, N_CUSTOMERS)
    customers.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)])
    customers = add_missing(rng, customers)
    customers["snapshot_date"] = SNAPSHOT.date()
    customers["churned"] = (rng.random(N_CUSTOMERS) < sigmoid(churn_logit(customers))).astype(int)

    # --- next quarter: same customers, drifted world ---
    ids = rng.choice(customers["customer_id"], size=N_NEXT_QUARTER, replace=False)
    base = customers.set_index("customer_id").loc[ids].reset_index()
    drifted = make_profiles(rng, N_NEXT_QUARTER, drift=True)
    nq = base[["customer_id", "branch_region", "has_credit_card", "estimated_salary"]].copy()
    nq["age"] = base["age"]
    nq["tenure_months"] = base["tenure_months"] + 3
    nq["num_products"] = base["num_products"]
    nq["balance"] = (base["balance"] * rng.uniform(0.4, 0.7, N_NEXT_QUARTER)).round(2)
    nq["credit_score"] = base["credit_score"]
    nq["is_active_member"] = base["is_active_member"]
    nq["digital_logins_30d"] = drifted["digital_logins_30d"]
    nq["complaints_90d"] = drifted["complaints_90d"]
    nq = nq[customers.columns.drop(["snapshot_date", "churned"])]
    nq["snapshot_date"] = NEXT_SNAPSHOT.date()
    nq["churned"] = (rng.random(N_NEXT_QUARTER) < sigmoid(churn_logit(nq, concept_drift=True))).astype(int)

    # --- transactions: Q2 for everyone, Q3 for the re-snapshotted customers ---
    txns = pd.concat(
        [make_transactions(rng, customers, SNAPSHOT), make_transactions(rng, nq, NEXT_SNAPSHOT)],
        ignore_index=True,
    ).sort_values("txn_ts").reset_index(drop=True)
    txns.insert(0, "txn_id", [f"T{i:07d}" for i in range(1, len(txns) + 1)])

    customers.to_csv(OUT_DIR / "customers.csv", index=False)
    txns.to_csv(OUT_DIR / "transactions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    nq.to_csv(OUT_DIR / "customers_next_quarter.csv", index=False)

    print(f"customers:              {len(customers):>7,} rows, churn rate {customers['churned'].mean():.1%}")
    print(f"transactions:           {len(txns):>7,} rows")
    print(f"customers_next_quarter: {len(nq):>7,} rows, churn rate {nq['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
