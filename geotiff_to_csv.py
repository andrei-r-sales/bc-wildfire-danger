#!/usr/bin/env python3
"""
geotiff_to_csv.py
=================
Convert a directory of AppEEARS / GeoTIFF rasters into tidy long-format CSVs
with columns: lat, lon, date, <layer>.

This is the missing link in the wildfire pipeline. The downstream join
(merge_test__2_.ipynb) reads per-feature CSVs such as EVI.csv / root_wetness.csv
that each carry `lat`, `lon`, `date` plus a value column, then snaps every BC
weather station to its nearest satellite pixel with a haversine BallTree. This
script produces exactly those CSVs from the .tif files that nasa_extract.py /
download_all.py pull out of AppEEARS.

Two modes
---------
full    (default)  dump every valid (non-nodata) pixel -> for province-wide
                   grids and for the BallTree join, which keeps native pixels.
points  --points   sample the rasters only at supplied lat/lon points -> handy
                   if you want values straight at the station coordinates.

AppEEARS area outputs are typically named like
    MOD13Q1.061__250m_16_days_NDVI_doy2023145_aid0001.tif
so the date is parsed from the `doyYYYYDDD` token and the layer name is the part
before it. Use --layer-map to rename messy layer keys to the clean names the
training notebook expects (NDVI, EVI, surface_temp_max, ...).

Examples
--------
    # one CSV per layer, full pixel dump, downsampled 1:4 to keep size sane
    python geotiff_to_csv.py --input year/ --output data/csv_output/ --decimate 4

    # sample at the weather stations instead of dumping every pixel
    python geotiff_to_csv.py --input year/ --output out/ \
        --points 2023_BCWS_WX_STATIONS.csv --lat-col LATITUDE --lon-col LONGITUDE

    # rename AppEEARS layer keys to clean names
    python geotiff_to_csv.py --input year/ --output out/ \
        --layer-map layer_map.json
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as warp_transform

# AppEEARS encodes the date as a day-of-year token: _doyYYYYDDD
DOY_RE = re.compile(r"_doy(\d{4})(\d{3})")
# fallbacks: a plain YYYYMMDD or YYYY-MM-DD anywhere in the name
YMD_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def parse_layer_and_date(path):
    """Return (layer_name, iso_date_or_None) parsed from a filename."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    m = DOY_RE.search(stem)
    if m:
        year, doy = int(m.group(1)), int(m.group(2))
        date = (datetime(year, 1, 1) + timedelta(days=doy - 1)).date().isoformat()
        layer = stem[: m.start()].rstrip("_")
        # strip the trailing AppEEARS aid suffix if it leaked into the layer
        layer = re.sub(r"_aid\d+$", "", layer)
        return _clean_layer(layer), date

    m = YMD_RE.search(stem)
    if m:
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        layer = stem[: m.start()].rstrip("_-")
        return _clean_layer(layer or stem), date

    return _clean_layer(stem), None


def _clean_layer(layer):
    """Collapse an AppEEARS product+layer key down to a short token.

    'MOD13Q1.061__250m_16_days_NDVI' -> 'NDVI'. Falls back to the whole key if
    there is nothing obviously trailing. Override with --layer-map for control.
    """
    token = layer.split("__")[-1] if "__" in layer else layer
    token = token.split("_")[-1] if "_" in token else token
    return token or layer


def raster_full_to_long(path, date, decimate=1):
    """Dump every valid pixel of band 1 to a DataFrame[lat, lon, value]."""
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        rows, cols = band.shape
        r_idx = np.arange(0, rows, decimate)
        c_idx = np.arange(0, cols, decimate)
        cc, rr = np.meshgrid(c_idx, r_idx)
        rr_flat = rr.ravel()
        cc_flat = cc.ravel()

        vals = np.asarray(band[rr_flat, cc_flat], dtype="float64")
        valid = ~np.ma.getmaskarray(band)[rr_flat, cc_flat]

        # pixel-centre coordinates in the raster's own CRS
        xs, ys = transform_xy(src.transform, rr_flat.tolist(), cc_flat.tolist())
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        lon, lat = _to_lonlat(src.crs, xs, ys)

    df = pd.DataFrame({"lat": lat, "lon": lon, "value": vals})
    df = df[valid]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if date is not None:
        df["date"] = date
    return df


def raster_sample_points(path, date, pts, lat_col, lon_col):
    """Sample band 1 at the given lat/lon points -> DataFrame[lat, lon, value]."""
    with rasterio.open(path) as src:
        lon = pts[lon_col].to_numpy(dtype="float64")
        lat = pts[lat_col].to_numpy(dtype="float64")
        xs, ys = _from_lonlat(src.crs, lon, lat)
        coords = list(zip(xs, ys))
        sampled = np.array([v[0] for v in src.sample(coords)], dtype="float64")
        nodata = src.nodata

    df = pd.DataFrame({"lat": lat, "lon": lon, "value": sampled})
    if nodata is not None:
        df.loc[df["value"] == nodata, "value"] = np.nan
    df = df.replace([np.inf, -np.inf], np.nan)
    if date is not None:
        df["date"] = date
    return df


def _to_lonlat(crs, xs, ys):
    if crs is None:
        return xs, ys
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return xs, ys  # already lon/lat
    lon, lat = warp_transform(crs, "EPSG:4326", xs.tolist(), ys.tolist())
    return np.asarray(lon), np.asarray(lat)


def _from_lonlat(crs, lon, lat):
    if crs is None:
        return lon, lat
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return lon, lat
    xs, ys = warp_transform("EPSG:4326", crs, lon.tolist(), lat.tolist())
    return np.asarray(xs), np.asarray(ys)


def collect_inputs(input_arg):
    if os.path.isdir(input_arg):
        files = sorted(glob.glob(os.path.join(input_arg, "**", "*.tif"), recursive=True))
        files += sorted(glob.glob(os.path.join(input_arg, "**", "*.tiff"), recursive=True))
        return sorted(set(files))
    return sorted(glob.glob(input_arg))


def main(argv=None):
    ap = argparse.ArgumentParser(description="GeoTIFF -> tidy long CSV (lat, lon, date, value)")
    ap.add_argument("--input", required=True, help="directory of .tif files or a glob")
    ap.add_argument("--output", required=True, help="output directory for per-layer CSVs")
    ap.add_argument("--decimate", type=int, default=1, help="full mode: keep every Nth pixel per axis")
    ap.add_argument("--points", help="CSV of points for sampling mode")
    ap.add_argument("--lat-col", default="lat", help="latitude column in --points CSV")
    ap.add_argument("--lon-col", default="lon", help="longitude column in --points CSV")
    ap.add_argument("--layer-map", help="JSON file mapping parsed layer key -> clean name")
    ap.add_argument("--date", help="override the date for every file (ISO YYYY-MM-DD)")
    ap.add_argument("--combine", action="store_true",
                    help="also write a single combined.csv with a 'layer' column")
    args = ap.parse_args(argv)

    files = collect_inputs(args.input)
    if not files:
        sys.exit(f"No .tif files found under {args.input!r}")

    layer_map = {}
    if args.layer_map:
        with open(args.layer_map) as fh:
            layer_map = json.load(fh)

    pts = None
    if args.points:
        pts = pd.read_csv(args.points)
        missing = {args.lat_col, args.lon_col} - set(pts.columns)
        if missing:
            sys.exit(f"--points CSV is missing columns: {sorted(missing)}")
        pts = pts.dropna(subset=[args.lat_col, args.lon_col])

    os.makedirs(args.output, exist_ok=True)
    per_layer = {}

    for path in files:
        layer, date = parse_layer_and_date(path)
        layer = layer_map.get(layer, layer)
        if args.date:
            date = args.date
        try:
            if pts is not None:
                df = raster_sample_points(path, date, pts, args.lat_col, args.lon_col)
            else:
                df = raster_full_to_long(path, date, decimate=args.decimate)
        except Exception as e:  # keep going on a bad tile
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        df = df.rename(columns={"value": layer})
        per_layer.setdefault(layer, []).append(df)
        print(f"  {os.path.basename(path)} -> layer={layer} date={date} rows={len(df)}")

    combined = []
    for layer, frames in per_layer.items():
        out = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(args.output, f"{layer}.csv")
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({len(out)} rows)")
        if args.combine:
            c = out.rename(columns={layer: "value"}).copy()
            c["layer"] = layer
            combined.append(c)

    if args.combine and combined:
        allc = pd.concat(combined, ignore_index=True)
        allc_path = os.path.join(args.output, "combined.csv")
        allc.to_csv(allc_path, index=False)
        print(f"wrote {allc_path}  ({len(allc)} rows)")


if __name__ == "__main__":
    main()
