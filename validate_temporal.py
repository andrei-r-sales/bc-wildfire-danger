#!/usr/bin/env python3
"""
validate_temporal.py
====================
Honest evaluation of the wildfire model using a TEMPORAL split (train on early
years, test on a held-out recent period) instead of a leak-prone random split.

Also supports an A/B experiment on DROUGHT features via --drought, so you can
measure whether longer-memory dryness features improve the honest (temporal)
number before committing to retraining/serving them.

    # baseline (the 37 deployed features)
    python validate_temporal.py --dataset full_dataset.csv

    # with drought features added
    python validate_temporal.py --dataset full_dataset.csv --drought

Compare the "temporal test" lines: if --drought raises within±1 / QWK, the
features earn their place; then we mirror them into slim_model.py + daily_extract.py.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight

from daily_extract2 import FEATURE_ORDER

REQUIRED_RAW = [
    "station_code", "date", "DANGER_RATING", "mean_temp", "precipitation_sum",
    "mean_wind_east", "mean_wind_north", "max_wind_east", "max_wind_north",
    "EVI", "NDVI",
]

# --- drought feature config ---
# Final shipped set (after temporal A/B): the two precipitation-accumulation
# windows. days_since_rain (#36), humidity/vpd (redundant), and precip_90day_sum
# (#21, diminishing returns) were all tested and dropped. precip_30day_sum was
# the model's #1 feature.
DROUGHT_FEATURES = ["precip_30day_sum", "precip_60day_sum"]

# --- humidity / dryness-of-air config ---
# Computed from max_temp + mean_dewpoint, which the serving path already has, so
# these cost nothing extra at inference time. vpd = vapor pressure deficit (how
# "thirsty" the air is); we also test its simpler cousins to see which form wins.
HUMIDITY_FEATURES = ["vpd", "relative_humidity", "temp_dewpoint_spread"]


def add_humidity_features(df):
    """VPD (kPa), RH (%), and the raw T-Td spread, from Kelvin temp + dewpoint."""
    T = df["max_temp"] - 273.15          # peak daily temp, deg C
    Td = df["mean_dewpoint"] - 273.15    # dewpoint, deg C
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))     # sat. vapor pressure at T
    ea = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))   # actual vapor pressure
    df["vpd"] = (es - ea).clip(lower=0)
    df["relative_humidity"] = (ea / es * 100).clip(0, 100)
    df["temp_dewpoint_spread"] = df["max_temp"] - df["mean_dewpoint"]
    return df


def preprocess(path):
    df = pd.read_csv(path)
    if "elevation" not in df.columns and "ELEVATION_M" in df.columns:
        df = df.rename(columns={"ELEVATION_M": "elevation"})
    missing_raw = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing_raw:
        raise SystemExit(f"full_dataset.csv missing columns: {missing_raw}")

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

    # --- drought features (accumulation windows; days_since_rain dropped after A/B) ---
    df["precip_30day_sum"] = g["precipitation_sum"].transform(lambda x: x.rolling(30, min_periods=1).sum())
    df["precip_60day_sum"] = g["precipitation_sum"].transform(lambda x: x.rolling(60, min_periods=1).sum())

    # --- humidity features (free at serving time: derived from temp + dewpoint) ---
    df = add_humidity_features(df)

    dates = df["date"].copy()
    df = df.drop(["date", "station_name", "station_code"], axis=1, errors="ignore")
    df["__date__"] = dates.values
    df = df.dropna()

    y = df["DANGER_RATING"].astype(int)
    keep_dates = df["__date__"]
    base = [c for c in FEATURE_ORDER if c not in DROUGHT_FEATURES]
    cols = base + DROUGHT_FEATURES + HUMIDITY_FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing expected features: {missing}")
    X = df[cols]
    return X, y, keep_dates


def report(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    within1 = np.mean(np.abs(y_true.to_numpy() - y_pred) <= 1)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"  {label:<22} exact={acc:.2%}  within±1={within1:.2%}  QWK={qwk:.4f}")


def _region(lat, lon):
    """Rough BC bands so we can see WHERE the model's bias lives. Boundaries are
    deliberately simple heuristics, not official ecozones."""
    if lat <= 50.0 and lon <= -122.5:
        return "coast/SW"
    if lat >= 54.0:
        return "north"
    return "interior"


def compare_report(y_true, y_pred, X_te):
    """Quantify HOW the model disagrees with the official rating (the label),
    not just how often: confusion matrix + signed bias overall / by region /
    by official class. Positive bias = model predicts HIGHER than official."""
    yt = y_true.to_numpy()
    yp = np.asarray(y_pred)
    labels = [1, 2, 3, 4, 5]
    names = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High", 5: "Extreme"}

    print("\n=== Comparison vs official rating (temporal test set) ===")
    print(f"  signed bias (pred - official): {np.mean(yp - yt):+.3f} levels "
          f"({'over-calls' if np.mean(yp - yt) > 0 else 'under-calls'})")

    print("\n  Confusion (rows = official, cols = predicted), row %:")
    print("            " + "".join(f"{names[c][:4]:>7}" for c in labels))
    for r in labels:
        mask = yt == r
        n = mask.sum()
        if n == 0:
            print(f"  {names[r]:<10} (no test rows)")
            continue
        row = [(np.sum(yp[mask] == c) / n) * 100 for c in labels]
        print(f"  {names[r]:<10}" + "".join(f"{v:6.0f}%" for v in row) + f"   (n={n})")

    print("\n  Signed bias by region:")
    regions = np.array([_region(la, lo) for la, lo in
                        zip(X_te["lat"].to_numpy(), X_te["lon"].to_numpy())])
    for reg in ["coast/SW", "interior", "north"]:
        m = regions == reg
        if m.sum():
            print(f"    {reg:<10} {np.mean(yp[m] - yt[m]):+.3f}   (n={m.sum():,})")

    print("\n  When the official rating is X, what does the model say?")
    for r in labels:
        m = yt == r
        if m.sum():
            over = np.mean(yp[m] >= 3) * 100   # predicted High+ 
            print(f"    official {names[r]:<10} mean predicted={np.mean(yp[m]):.2f}, "
                  f"{over:4.0f}% called High+  (n={m.sum():,})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Temporal-split validation of the wildfire model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--test-from", default="2024-01-01")
    ap.add_argument("--drought", action="store_true", help="include the 2 drought features")
    ap.add_argument("--humidity", action="store_true", help="include the 3 humidity features (vpd etc.)")
    ap.add_argument("--compare", action="store_true",
                    help="print confusion matrix + bias breakdown vs the official rating")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=30)
    ap.add_argument("--min-samples-leaf", type=int, default=10)
    args = ap.parse_args(argv)

    print("Preprocessing...")
    X, y, dates = preprocess(args.dataset)
    # FEATURE_ORDER now ships WITH the drought features, so the true 37-feature
    # baseline is FEATURE_ORDER minus them; toggles add candidate groups back.
    BASE = [c for c in FEATURE_ORDER if c not in DROUGHT_FEATURES]
    feature_set = (BASE
                   + (DROUGHT_FEATURES if args.drought else [])
                   + (HUMIDITY_FEATURES if args.humidity else []))
    X = X[feature_set]
    tags = []
    if args.drought:
        tags.append("drought")
    if args.humidity:
        tags.append("humidity")
    label = ("+" + "+".join(tags)) if tags else "baseline"
    print(f"  feature set: {len(feature_set)} features ({label})")

    cutoff = pd.Timestamp(args.test_from)
    tr, te = dates < cutoff, dates >= cutoff
    print(f"  train: {tr.sum():,} rows (before {args.test_from})")
    print(f"  test:  {te.sum():,} rows (on/after {args.test_from})")
    if te.sum() == 0 or tr.sum() == 0:
        raise SystemExit("Empty split side; pick a --test-from inside the data range.")

    params = dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
                  min_samples_leaf=args.min_samples_leaf,
                  class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    model = RandomForestClassifier(**params)
    model.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))

    y_pred_te = model.predict(X[te])
    print("\nHonest (temporal) performance on held-out recent period:")
    report(y[te], y_pred_te, "temporal test")
    print("\nTraining-set fit (reference only):")
    report(y[tr], model.predict(X[tr]), "train")

    if args.compare:
        compare_report(y[te], y_pred_te, X[te])

    if args.drought or args.humidity:
        imp = pd.Series(model.feature_importances_, index=feature_set).sort_values(ascending=False)
        groups = {}
        if args.drought:
            groups["Drought"] = DROUGHT_FEATURES
        if args.humidity:
            groups["Humidity"] = HUMIDITY_FEATURES
        for name, feats in groups.items():
            print(f"\n{name} feature importances (rank out of {len(feature_set)}):")
            for f in feats:
                rank = list(imp.index).index(f) + 1
                print(f"  {f:<22} importance={imp[f]:.4f}  (#{rank})")


if __name__ == "__main__":
    main()
