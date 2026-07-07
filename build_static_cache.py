#!/usr/bin/env python3
"""
build_static_cache.py
=====================
Assemble grid_static.csv -- the cached, slow-changing satellite features that
daily_extract.py reads -- from the per-feature CSVs that geotiff_to_csv.py
produces.

The chain is:
    AppEEARS .tif bundle
      -> geotiff_to_csv.py        (one tidy CSV per layer: lat, lon, date, <layer>)
      -> build_static_cache.py    (this script: snap each layer to the BC grid)
      -> grid_static.csv          (lat, lon, + all satellite features per grid point)

For each grid point we snap to the NEAREST satellite pixel using a haversine
BallTree -- the same nearest-neighbour idea as merge_test__2_.ipynb. Different
layers have different resolutions (1 km / 9 km / 36 km), so each gets its own
tree. We take the most recent date available per layer (these features change
slowly, so "latest composite" is what you want for a live product).

Run it occasionally (weekly/monthly) to refresh the cache, not daily.

Usage
-----
    # typical: per-feature CSVs live in data/csv_output/, default BC grid
    python build_static_cache.py --features-dir data/csv_output/ \
        --mask BC_mask.geojson --output grid_static.csv

    # pin a specific date and a finer grid
    python build_static_cache.py --features-dir out/ --date 2024-06-24 --step 0.25
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Reuse the grid + elevation + feature list from the daily pipeline so the
# cache is built on exactly the points the daily job will query.
from daily_extract2 import build_grid, fetch_elevation, SATELLITE_FEATURES


def latest_slice(df, date=None):
    """Keep one date's worth of rows: the requested date, else the most recent."""
    if "date" not in df.columns:
        return df
    if date:
        return df[df["date"] == date]
    available = sorted(d for d in df["date"].dropna().unique())
    if not available:
        return df
    return df[df["date"] == available[-1]]


def value_column(df, feature):
    """The converter names the value column after the layer; fall back sensibly."""
    if feature in df.columns:
        return feature
    candidates = [c for c in df.columns if c not in ("lat", "lon", "date")]
    if not candidates:
        raise ValueError("no value column found")
    return candidates[0]


def nearest_join(grid, feat_df, value_col):
    """For each grid point, return the value of the nearest satellite pixel."""
    from sklearn.neighbors import BallTree
    feat_df = feat_df.dropna(subset=["lat", "lon", value_col])
    if feat_df.empty:
        return np.full(len(grid), np.nan)
    tree = BallTree(np.radians(feat_df[["lat", "lon"]].to_numpy()), metric="haversine")
    _, idx = tree.query(np.radians(grid[["lat", "lon"]].to_numpy()), k=1)
    return feat_df[value_col].to_numpy()[idx.ravel()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build grid_static.csv from converter output")
    ap.add_argument("--features-dir", required=True, help="dir of per-feature CSVs from geotiff_to_csv.py")
    ap.add_argument("--output", default="grid_static.csv")
    ap.add_argument("--step", type=float, default=0.5, help="grid spacing in degrees (match daily_extract)")
    ap.add_argument("--mask", default=None, help="BC_mask.geojson to clip the grid")
    ap.add_argument("--date", default=None, help="ISO date to pin; default = latest per layer")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.features_dir):
        sys.exit(f"--features-dir {args.features_dir!r} is not a directory")

    print(f"Building grid (step={args.step})...")
    grid = build_grid(args.step, mask=args.mask)
    out = grid.copy()
    print(f"  {len(grid)} grid points")

    satellite = [f for f in SATELLITE_FEATURES if f != "elevation"]
    missing = []
    for feat in satellite:
        path = os.path.join(args.features_dir, f"{feat}.csv")
        if not os.path.exists(path):
            missing.append(feat)
            out[feat] = 0.0
            continue
        df = pd.read_csv(path)
        df = latest_slice(df, args.date)
        col = value_column(df, feat)
        out[feat] = nearest_join(grid, df, col)
        date_note = df["date"].iloc[0] if "date" in df.columns and len(df) else "n/a"
        print(f"  {feat:<24} <- {os.path.basename(path)}  (date={date_note}, pixels={len(df)})")

    # elevation: prefer a converter CSV, else the free Open-Meteo elevation API
    elev_path = os.path.join(args.features_dir, "elevation.csv")
    if os.path.exists(elev_path):
        edf = latest_slice(pd.read_csv(elev_path), args.date)
        out["elevation"] = nearest_join(grid, edf, value_column(edf, "elevation"))
        print("  elevation                <- elevation.csv")
    else:
        try:
            out["elevation"] = fetch_elevation(grid)
            print("  elevation                <- Open-Meteo elevation API")
        except Exception as e:
            out["elevation"] = 0.0
            missing.append("elevation")
            print(f"  elevation                <- 0.0 (API failed: {e})")

    if missing:
        print("\n  NOTE: these features had no source CSV and were filled with 0.0:")
        print(f"        {missing}")
        print("        Rerun geotiff_to_csv.py for those layers to fill them properly.")

    cols = ["lat", "lon"] + SATELLITE_FEATURES
    out[cols].to_csv(args.output, index=False)
    print(f"\nwrote {args.output}  ({len(out)} rows, {len(SATELLITE_FEATURES)} features)")


if __name__ == "__main__":
    main()
