#!/usr/bin/env python3
"""
slim_model.py
=============
Retrain the wildfire RandomForest with size constraints so the saved model
shrinks from ~2.2 GB to tens of MB, with minimal accuracy loss.

The 2.2 GB size comes from fully-grown trees on ~562k rows -- leaves split down
to tiny groups, producing millions of nodes. Capping `min_samples_leaf` and
`max_depth` collapses that node count dramatically and usually *improves*
generalization (less overfitting), so it's close to a free win.

This replicates the preprocessing from wildfire-prediction.ipynb exactly, so the
retrained model is a drop-in replacement -- same 37 features, same order, same
`feature_names_in_` -- meaning daily_extract.py works against it unchanged.

Run locally (needs full_dataset.csv + daily_extract.py in the folder):
    python slim_model.py --dataset full_dataset.csv

Size/accuracy knobs (smaller leaf / shallower depth / fewer trees = smaller file):
    --min-samples-leaf 50 --max-depth 25 --n-estimators 100
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from daily_extract2 import FEATURE_ORDER

# raw columns the notebook preprocessing expects to find in full_dataset.csv
REQUIRED_RAW = [
    "station_code", "date", "DANGER_RATING", "mean_temp", "precipitation_sum",
    "mean_wind_east", "mean_wind_north", "max_wind_east", "max_wind_north",
    "EVI", "NDVI",
]


def preprocess(path):
    df = pd.read_csv(path)
    if "elevation" not in df.columns and "ELEVATION_M" in df.columns:
        df = df.rename(columns={"ELEVATION_M": "elevation"})

    missing_raw = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing_raw:
        raise SystemExit(
            f"full_dataset.csv is missing expected columns: {missing_raw}\n"
            f"Columns present: {list(df.columns)}"
        )

    # --- identical to wildfire-prediction.ipynb ---
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df.station_code.astype(str).isin(["964", "1239"])]
    df = df[(df.date != "2024-08-12") &
            (df.date >= "2015-04-01") &
            (df.date <= "2025-07-15")]
    df = df.sort_values(["station_code", "date"])

    df[["EVI", "NDVI"]] = (
        df.groupby("station_code")[["EVI", "NDVI"]]
          .transform(lambda g: g.interpolate(method="linear").ffill().bfill())
    )

    df["average_wind_speed"] = np.sqrt(df["mean_wind_east"]**2 + df["mean_wind_north"]**2)
    df["max_wind_speed"] = np.sqrt(df["max_wind_east"]**2 + df["max_wind_north"]**2)
    df = df.drop(["mean_wind_east", "mean_wind_north", "max_wind_east", "max_wind_north"], axis=1)

    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    g = df.groupby("station_code")
    df["precip_7day_sum"] = g["precipitation_sum"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["precip_14day_sum"] = g["precipitation_sum"].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df["precip_7day_mean"] = g["precipitation_sum"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["precip_14day_mean"] = g["precipitation_sum"].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["temp_7day_sum"] = g["mean_temp"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["temp_14day_sum"] = g["mean_temp"].transform(lambda x: x.rolling(14, min_periods=1).sum())
    df["temp_7day_mean"] = g["mean_temp"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["temp_14day_mean"] = g["mean_temp"].transform(lambda x: x.rolling(14, min_periods=1).mean())

    df = df.drop(["date", "station_name", "station_code"], axis=1, errors="ignore")
    df = df.dropna()

    y = df["DANGER_RATING"].astype(int)
    X = df.drop(columns="DANGER_RATING")

    # guarantee the exact 37-column set+order the daily pipeline asserts on
    missing = [c for c in FEATURE_ORDER if c not in X.columns]
    if missing:
        raise SystemExit(
            f"After preprocessing, these expected features are missing: {missing}\n"
            f"Produced columns: {list(X.columns)}"
        )
    X = X[FEATURE_ORDER]
    return X, y


def metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    within1 = np.mean(np.abs(y_true.to_numpy() - y_pred) <= 1)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"  {label:<10} exact={acc:.2%}  within±1={within1:.2%}  QWK={qwk:.4f}")
    return acc, within1, qwk


def total_nodes(forest):
    return sum(t.tree_.node_count for t in forest.estimators_)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrain a slim wildfire RandomForest")
    ap.add_argument("--dataset", required=True, help="full_dataset.csv")
    ap.add_argument("--output", default="rf_model_slim.pkl")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--min-samples-leaf", type=int, default=50)
    args = ap.parse_args(argv)

    print("Preprocessing (matching the training notebook)...")
    X, y = preprocess(args.dataset)
    print(f"  {len(X):,} rows, {X.shape[1]} features")

    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    print(f"\nConstraints: n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}")

    # --- validation split, to confirm accuracy holds ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    slim = RandomForestClassifier(**params)
    t0 = time.time()
    slim.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    print(f"\nTrained validation model in {time.time()-t0:.0f}s")
    print("Validation metrics:")
    metrics(y_val, slim.predict(X_val), "slim")
    print("  baseline   exact=~70%    within±1=~99%   QWK=~0.87   (original 2.2GB model)")

    # --- retrain on the FULL dataset for deployment (same as the notebook) ---
    print("\nRetraining on full dataset for the saved model...")
    final = RandomForestClassifier(**params)
    final.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    assert list(final.feature_names_in_) == FEATURE_ORDER, "feature order drift!"

    joblib.dump(final, args.output, compress=3)
    new_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output}  ({new_mb:,.1f} MB, {total_nodes(final):,} total nodes)")
    if os.path.exists("rf_model.pkl"):
        old_mb = os.path.getsize("rf_model.pkl") / 1e6
        print(f"Old rf_model.pkl: {old_mb:,.1f} MB  ->  {old_mb/new_mb:.0f}x smaller")
    print("\nFeature order matches daily_extract: drop-in replacement.")


if __name__ == "__main__":
    main()
