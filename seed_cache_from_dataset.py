#!/usr/bin/env python3
"""
seed_cache_from_dataset.py
==========================
Build grid_static.csv (the satellite-feature cache the daily job reads) from the
EXISTING full_dataset.csv training table -- no NASA re-download needed.

full_dataset.csv holds the satellite features (NDVI, EVI, surface temp, wetness,
evapotranspiration) at the ~200 BC weather stations across ~10 years. For each
station we take its most recent non-null value per feature (a complete "latest
snapshot"), then snap those station values onto the province-wide grid with a
haversine BallTree -- the same nearest-neighbour join the rest of the pipeline
uses.

Caveat: ~200 stations projected to a grid gives a COARSE satellite field (each
grid cell inherits its nearest station's value). It's a big step up from zeros,
not a true satellite raster. For a sharper field, run geotiff_to_csv.py on a
fresh AppEEARS bundle instead. The upside here: values come from the exact table
the model trained on, so units/scale match the model perfectly.

Usage:
    python seed_cache_from_dataset.py --dataset full_dataset.csv \
        --step 0.5 --mask BC_mask.geojson --output grid_static.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd

from daily_extract import build_grid, SATELLITE_FEATURES

# alternate column spellings seen in the training table
ALIASES = {
    "elevation": ["elevation", "ELEVATION_M"],
}
LAT_CANDIDATES = ["lat", "LATITUDE", "Latitude"]
LON_CANDIDATES = ["lon", "LONGITUDE", "Longitude"]
ID_CANDIDATES = ["STATION_CODE", "station_code"]
DATE_CANDIDATES = ["DATE_TIME", "date", "DATE"]


def first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def resolve_feature(cols, feature):
    for cand in [feature] + ALIASES.get(feature, []):
        if cand in cols:
            return cand
    return None


def last_valid(series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def representative(series):
    """Median over a station's FULL history -> a stable, in-distribution value.

    The previous approach (last reading per station) grabbed whatever the
    dataset's final dates happened to hold; because the data ends in summer and
    only some stations have late readings, that collapsed the cache to a few
    anomalously hot values (surface_temp_max ~301K vs a ~282K training median),
    which made the model predict extreme risk almost everywhere. These satellite
    features barely vary by season (<~4K month-to-month here), so a per-station
    median is both representative and unbiased.
    """
    s = series.dropna()
    return float(s.median()) if len(s) else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seed grid_static.csv from full_dataset.csv")
    ap.add_argument("--dataset", required=True, help="path to full_dataset.csv")
    ap.add_argument("--output", default="grid_static.csv")
    ap.add_argument("--step", type=float, default=0.5, help="grid spacing (match daily_extract)")
    ap.add_argument("--mask", default=None, help="BC_mask.geojson to clip the grid")
    ap.add_argument("--date", default=None, help="only use rows up to this ISO date (default: all)")
    args = ap.parse_args(argv)

    header = pd.read_csv(args.dataset, nrows=0).columns.tolist()
    lat_col = first_present(header, LAT_CANDIDATES)
    lon_col = first_present(header, LON_CANDIDATES)
    id_col = first_present(header, ID_CANDIDATES)
    date_col = first_present(header, DATE_CANDIDATES)
    if not (lat_col and lon_col):
        sys.exit(f"Could not find lat/lon columns. Header has: {header}")

    feat_cols = {f: resolve_feature(header, f) for f in SATELLITE_FEATURES}
    present = {f: c for f, c in feat_cols.items() if c}
    missing = [f for f, c in feat_cols.items() if not c]

    # read only the columns we need (full_dataset.csv is large)
    usecols = list(dict.fromkeys(
        [c for c in (id_col, date_col, lat_col, lon_col) if c] + list(present.values())
    ))
    print(f"Reading {len(usecols)} columns from {args.dataset}...")
    df = pd.read_csv(args.dataset, usecols=usecols)
    print(f"  {len(df):,} rows")

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if args.date:
            df = df[df[date_col] <= pd.Timestamp(args.date)]
        df = df.sort_values(date_col)

    group_key = id_col if id_col else [lat_col, lon_col]
    agg = {lat_col: "last", lon_col: "last"}
    for c in present.values():
        agg[c] = representative
    stations = df.groupby(group_key, as_index=False).agg(agg)
    stations = stations.rename(columns={c: f for f, c in present.items()})
    stations = stations.rename(columns={lat_col: "lat", lon_col: "lon"})
    stations = stations.dropna(subset=["lat", "lon"])
    print(f"  {len(stations)} stations with a recent snapshot")

    grid = build_grid(args.step, mask=args.mask)
    out = grid.copy()
    print(f"  {len(grid)} grid points")

    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    _, idx = tree.query(np.radians(grid[["lat", "lon"]].to_numpy()), k=1)
    idx = idx.ravel()

    for feat in present:
        vals = stations[feat].to_numpy()[idx]
        # any station that never observed this feature -> fill with overall median
        if np.isnan(vals).any():
            med = np.nanmedian(stations[feat].to_numpy())
            vals = np.where(np.isnan(vals), med if not np.isnan(med) else 0.0, vals)
        out[feat] = vals
        print(f"  {feat:<24} seeded (station range "
              f"{np.nanmin(stations[feat]):.3g}..{np.nanmax(stations[feat]):.3g})")

    for feat in missing:
        out[feat] = 0.0
    if missing:
        print(f"\n  NOTE: not found in dataset, filled with 0.0: {missing}")

    cols = ["lat", "lon"] + SATELLITE_FEATURES
    out[cols].to_csv(args.output, index=False)
    print(f"\nwrote {args.output}  ({len(out)} rows, {len(SATELLITE_FEATURES)} features)")


if __name__ == "__main__":
    main()
