#!/usr/bin/env python3
"""
daily_extract.py
================
Produce the *latest* province-wide wildfire-risk prediction for British Columbia,
end to end, with no slow batch downloads:

    grid -> Open-Meteo API (weather, instant) + cached satellite features
         -> feature engineering (identical to wildfire-prediction.ipynb)
         -> rf_model.pkl
         -> predictions CSV + GeoJSON + PNG heatmap

Why Open-Meteo instead of the original ERA5/AppEEARS jobs
---------------------------------------------------------
ERA5 (Copernicus) lags ~5 days and AppEEARS area requests are asynchronous and
can take hours-to-days. Neither can drive a "today" product. Open-Meteo is a
free, no-key JSON API that returns recent history + forecast instantly, and its
`past_days` parameter hands back the trailing window the 7/14-day rolling
features need. The satellite-derived features (NDVI, EVI, surface/rootzone
wetness, surface temp, evapotranspiration) change slowly, so they live in a
static cache you rebuild occasionally with geotiff_to_csv.py.

The model expects EXACTLY these 37 columns, in this order (taken from
testset_visualization.ipynb). We build the frame in this order so the model's
internal feature-name assertion passes.

Mapping notes (model feature  <-  source):
  EXACT (Open-Meteo daily/hourly):
    max_temp <- temperature_2m_max          mean_temp <- temperature_2m_mean
    precipitation_sum <- precipitation_sum  sum_snowfall <- snowfall_sum
    average_wind_speed <- wind_speed_10m_mean   max_wind_speed <- wind_speed_10m_max
    mean_dewpoint <- dew_point_2m_mean      mean_total_cloud_cover <- cloud_cover_mean
    mean_low_cloud_cover <- hourly cloud_cover_low (daily mean)
    mean_snow_depth <- hourly snow_depth (daily mean)
    max_solar_radiation <- hourly shortwave_radiation (daily max)
    mean_soil_water <- hourly soil_moisture_3_to_9cm (daily mean)
    precip_/temp_ 7|14 day sum|mean <- rolling over the past-14-day daily series
  APPROXIMATE (close proxy, flagged):
    mean_evaporation <- et0_fao_evapotranspiration  (reference ET, not actual)
    sum_snowmelt     <- 0.0  (not exposed by Open-Meteo; low model importance)
  CACHED (satellite, from geotiff_to_csv.py -> --static grid_static.csv):
    EVI NDVI rootzone_wetness_mean rootzone_wetness_max surface_temp_mean
    surface_temp_max surface_wetness_mean surface_wetness_max
    evapotranspiration_mean evapotranspiration_max  (+ elevation)

Usage
-----
    python daily_extract.py                       # today, default BC grid
    python daily_extract.py --date 2024-06-24 --step 0.5
    python daily_extract.py --static grid_static.csv --mask BC_mask.geojson
"""
import argparse
import json
import os
import tempfile
import time
from datetime import date as date_cls

import numpy as np
import pandas as pd
import requests

import storage  # local-or-S3 file access (set STORAGE_BACKEND=s3 for the cloud)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Exact column order the trained RandomForestClassifier expects.
FEATURE_ORDER = [
    "lat", "lon", "elevation", "EVI", "NDVI", "max_temp", "mean_temp",
    "precipitation_sum", "mean_snow_depth", "sum_snowfall", "sum_snowmelt",
    "rootzone_wetness_mean", "rootzone_wetness_max", "surface_temp_mean",
    "surface_temp_max", "surface_wetness_mean", "surface_wetness_max",
    "evapotranspiration_mean", "evapotranspiration_max", "max_solar_radiation",
    "mean_dewpoint", "mean_evaporation", "mean_low_cloud_cover", "mean_soil_water",
    "mean_total_cloud_cover", "average_wind_speed", "max_wind_speed",
    "sin_doy", "cos_doy", "precip_7day_sum", "precip_14day_sum",
    "precip_7day_mean", "precip_14day_mean", "temp_7day_sum", "temp_14day_sum",
    "temp_7day_mean", "temp_14day_mean",
]

# Satellite features filled from the static cache (everything not weather-derived).
SATELLITE_FEATURES = [
    "elevation", "EVI", "NDVI", "rootzone_wetness_mean", "rootzone_wetness_max",
    "surface_temp_mean", "surface_temp_max", "surface_wetness_mean",
    "surface_wetness_max", "evapotranspiration_mean", "evapotranspiration_max",
]

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_mean", "precipitation_sum",
    "snowfall_sum", "wind_speed_10m_max", "wind_speed_10m_mean",
    "dew_point_2m_mean", "cloud_cover_mean", "et0_fao_evapotranspiration",
]
HOURLY_VARS = [
    "snow_depth", "cloud_cover_low", "soil_moisture_3_to_9cm", "shortwave_radiation",
]

BC_BBOX = dict(lat_min=48.3, lat_max=60.0, lon_min=-139.0, lon_max=-114.0)


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def build_grid(step, bbox=BC_BBOX, mask=None):
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + 1e-9, step)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + 1e-9, step)
    grid = pd.DataFrame(
        [(round(la, 4), round(lo, 4)) for la in lats for lo in lons],
        columns=["lat", "lon"],
    )
    if mask:
        grid = clip_to_mask(grid, mask)
    return grid.reset_index(drop=True)


def clip_to_mask(grid, mask_path):
    """Keep only points inside the BC polygon. Requires geopandas/shapely."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception:
        print("  (geopandas not installed: skipping BC mask clip)")
        return grid
    mask_local = storage.local_copy(mask_path)  # downloads from S3 if needed
    gdf = gpd.read_file(mask_local)
    poly = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    inside = grid.apply(lambda r: poly.contains(Point(r["lon"], r["lat"])), axis=1)
    return grid[inside]


# --------------------------------------------------------------------------- #
# weather (Open-Meteo)
# --------------------------------------------------------------------------- #
def get_with_retry(url, params, max_retries=6, label="request"):
    """GET with exponential backoff on HTTP 429 (rate limit).

    Honors a Retry-After header if the server sends one, else backs off
    5/10/20/40/80/160 seconds. Raises after max_retries straight 429s.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 0)) or (2 ** attempt) * 5
            print(f"  rate limited (429) on {label}, waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(
        f"Open-Meteo kept returning 429 for {label} after {max_retries} retries. "
        "Try a smaller chunk size, a larger pause, or wait a few minutes "
        "before rerunning (free-tier quota resets over time)."
    )


def _grid_key(df):
    """Stable (lat, lon) key for matching grid points to checkpoint rows.

    build_grid rounds coords to 4 decimals and we store those same requested
    coords, so rounding here makes the match robust to float noise.
    """
    return list(zip(df["lat"].round(4), df["lon"].round(4)))


def fetch_weather(grid, target_date, chunk=40, pause=2.0, max_retries=6,
                  checkpoint_path=None):
    """Return one weather-feature row per grid point for target_date.

    Incremental checkpointing: after each chunk succeeds, its rows are appended
    to checkpoint_path immediately. On a rerun, points already in the checkpoint
    are skipped and only the missing ones are fetched. So a rate-limited or
    interrupted run loses nothing -- just rerun the same command to resume, and
    a fully-cached run does zero API calls. Delete the checkpoint to force a
    clean refetch (e.g. after changing the unit-conversion logic).

    Open-Meteo's free tier rate-limits per minute AND per day; small chunks plus
    a pause plus exponential backoff on 429 keep each run inside the per-minute
    budget, and checkpointing keeps you from re-spending the daily budget.
    """
    target = pd.Timestamp(target_date)

    # what's already done?
    done_rows = []
    done_keys = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        prev = prev.drop_duplicates(subset=["lat", "lon"], keep="last")
        # keep only points that belong to the current grid
        grid_keys = set(_grid_key(grid))
        prev = prev[[k in grid_keys for k in _grid_key(prev)]]
        done_rows = prev.to_dict("records")
        done_keys = set(_grid_key(prev))
        if done_keys:
            print(f"  resuming: {len(done_keys)}/{len(grid)} points already cached")

    remaining = grid[[k not in done_keys for k in _grid_key(grid)]].reset_index(drop=True)
    if len(remaining) == 0:
        print(f"  all {len(grid)} points cached, no API calls needed")
        return _align_to_grid(pd.DataFrame(done_rows), grid)

    new_rows = []
    try:
        for start in range(0, len(remaining), chunk):
            sub = remaining.iloc[start:start + chunk]
            params = {
                "latitude": ",".join(f"{v:.4f}" for v in sub["lat"]),
                "longitude": ",".join(f"{v:.4f}" for v in sub["lon"]),
                "daily": ",".join(DAILY_VARS),
                "hourly": ",".join(HOURLY_VARS),
                "timezone": "auto",
                "past_days": 14,
                "forecast_days": 2,
                "cell_selection": "nearest",
                # The model trained on ERA5 wind COMPONENTS in m/s, then took the
                # magnitude. Open-Meteo defaults to km/h, so ask for m/s directly;
                # then our speed maps straight onto the model's units (see UNIT
                # NOTES in _features_for_location for the rest).
                "wind_speed_unit": "ms",
            }

            resp = get_with_retry(FORECAST_URL, params, max_retries=max_retries, label="weather")

            payload = resp.json()
            if isinstance(payload, dict):
                payload = [payload]
            if len(payload) != len(sub):
                raise RuntimeError(
                    f"Open-Meteo returned {len(payload)} results for {len(sub)} "
                    "requested points; can't safely align them."
                )
            # cell_selection=nearest echoes back the model-grid lat/lon, jittered
            # from what we asked for; keep the REQUESTED coords as identity so the
            # final pivot is a clean rectangle, not a sparse NaN matrix.
            chunk_rows = []
            for (_, greq), loc in zip(sub.iterrows(), payload):
                row = _features_for_location(loc, target)
                row["lat"] = round(greq["lat"], 4)
                row["lon"] = round(greq["lon"], 4)
                chunk_rows.append(row)

            # checkpoint THIS chunk immediately, before the next request
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                write_header = not os.path.exists(checkpoint_path)
                pd.DataFrame(chunk_rows).to_csv(
                    checkpoint_path, mode="a", header=write_header, index=False)

            new_rows.extend(chunk_rows)
            fetched = len(done_keys) + len(new_rows)
            print(f"  weather {fetched}/{len(grid)}")
            time.sleep(pause)
    except RuntimeError as e:
        saved = len(done_keys) + len(new_rows)
        raise RuntimeError(
            f"{e}\nProgress saved: {saved}/{len(grid)} points are in "
            f"{checkpoint_path}. Rerun the same command to resume from there."
        )

    return _align_to_grid(pd.DataFrame(done_rows + new_rows), grid)


def _align_to_grid(weather, grid):
    """Return weather rows in grid order, one per grid point."""
    weather = weather.drop_duplicates(subset=["lat", "lon"], keep="last")
    g = grid.copy()
    g["lat"] = g["lat"].round(4)
    g["lon"] = g["lon"].round(4)
    merged = g.merge(weather, on=["lat", "lon"], how="left")
    return merged


def _features_for_location(loc, target):
    daily = pd.DataFrame(loc["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.sort_values("time").reset_index(drop=True)

    # daily means of hourly-only variables
    hourly = pd.DataFrame(loc["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["day"] = hourly["time"].dt.floor("D")
    agg = hourly.groupby("day").agg(
        mean_snow_depth=("snow_depth", "mean"),
        mean_low_cloud_cover=("cloud_cover_low", "mean"),
        mean_soil_water=("soil_moisture_3_to_9cm", "mean"),
        max_solar_radiation=("shortwave_radiation", "max"),
    ).reset_index()

    # the row for the target day (fall back to the latest available)
    day_mask = daily["time"] == target
    if not day_mask.any():
        target = daily["time"].iloc[-1]
        day_mask = daily["time"] == target
    drow = daily[day_mask].iloc[0]
    arow = agg[agg["day"] == target]
    arow = arow.iloc[0] if len(arow) else pd.Series(dtype="float64")

    # === UNIT CONVERSION: Open-Meteo -> ERA5, the units the model trained on ===
    # Confirmed from full_dataset.csv describe():
    #   temps    : Kelvin (medians ~279-283)  -> Open-Meteo Celsius + 273.15
    #   precip   : meters (max 0.13)           -> Open-Meteo mm / 1000
    #   cloud    : fraction 0-1                 -> Open-Meteo percent / 100
    #   wind     : m/s                          -> requested in m/s (no change)
    #   solar    : J/m^2 (med ~2.1e6)           -> Open-Meteo W/m^2 * 3600
    # Approximate (low-importance features, no clean live equivalent):
    #   snowfall : m water-equiv               -> Open-Meteo cm * ~0.001 (10:1)
    #   snow_depth: m water-equiv (max 0.51)    -> Open-Meteo m depth * ~0.1 (10:1)
    # Constants (no usable live source; set to training-median so they sit
    # in-distribution and contribute ~nothing, matching their near-zero importance):
    #   sum_snowmelt    = 0.0   (Open-Meteo has no snowmelt; training median 0)
    #   mean_evaporation= -7e-5 (ERA5 actual evaporation, training median)
    #   mean_soil_water = 0.0   (this column was ALL zero in training)
    C_TO_K = 273.15
    MM_TO_M = 1.0 / 1000.0
    PCT_TO_FRAC = 1.0 / 100.0
    WM2_TO_JM2 = 3600.0
    SNOW_CM_TO_M_WE = 0.001
    SNOWDEPTH_TO_WE = 0.1
    EVAP_CONST = -7e-5
    SOIL_WATER_CONST = 0.0

    # trailing windows, converted to ERA5 units BEFORE aggregating
    hist = daily[daily["time"] <= target]
    precip = hist["precipitation_sum"].to_numpy(dtype="float64") * MM_TO_M
    temp = hist["temperature_2m_mean"].to_numpy(dtype="float64") + C_TO_K

    def tail_sum(a, n):
        return float(np.nansum(a[-n:])) if len(a) else 0.0

    def tail_mean(a, n):
        seg = a[-n:]
        return float(np.nanmean(seg)) if len(seg) else 0.0

    return {
        "lat": round(loc["latitude"], 4),
        "lon": round(loc["longitude"], 4),
        "max_temp": float(drow["temperature_2m_max"]) + C_TO_K,
        "mean_temp": float(drow["temperature_2m_mean"]) + C_TO_K,
        "precipitation_sum": float(drow["precipitation_sum"]) * MM_TO_M,
        "sum_snowfall": float(drow["snowfall_sum"]) * SNOW_CM_TO_M_WE,
        "sum_snowmelt": 0.0,
        "average_wind_speed": float(drow["wind_speed_10m_mean"]),   # already m/s
        "max_wind_speed": float(drow["wind_speed_10m_max"]),        # already m/s
        "mean_dewpoint": float(drow["dew_point_2m_mean"]) + C_TO_K,
        "mean_total_cloud_cover": float(drow["cloud_cover_mean"]) * PCT_TO_FRAC,
        "mean_evaporation": EVAP_CONST,
        "mean_snow_depth": float(arow.get("mean_snow_depth", 0.0)) * SNOWDEPTH_TO_WE,
        "mean_low_cloud_cover": float(arow.get("mean_low_cloud_cover", 0.0)) * PCT_TO_FRAC,
        "mean_soil_water": SOIL_WATER_CONST,
        "max_solar_radiation": float(arow.get("max_solar_radiation", 0.0)) * WM2_TO_JM2,
        "precip_7day_sum": tail_sum(precip, 7),
        "precip_14day_sum": tail_sum(precip, 14),
        "precip_7day_mean": tail_mean(precip, 7),
        "precip_14day_mean": tail_mean(precip, 14),
        "temp_7day_sum": tail_sum(temp, 7),
        "temp_14day_sum": tail_sum(temp, 14),
        "temp_7day_mean": tail_mean(temp, 7),
        "temp_14day_mean": tail_mean(temp, 14),
    }


def fetch_elevation(grid, chunk=50, pause=1.0, max_retries=6):
    """Elevation never changes, so this is meant to run ONCE via
    build_static_cache.py, not on every daily run. Retries on 429 like
    fetch_weather."""
    elev = np.full(len(grid), np.nan)
    for start in range(0, len(grid), chunk):
        sub = grid.iloc[start:start + chunk]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in sub["lat"]),
            "longitude": ",".join(f"{v:.4f}" for v in sub["lon"]),
        }
        resp = get_with_retry(ELEVATION_URL, params, max_retries=max_retries, label="elevation")
        elev[start:start + len(sub)] = resp.json()["elevation"]
        time.sleep(pause)
    return elev


# --------------------------------------------------------------------------- #
# satellite cache
# --------------------------------------------------------------------------- #
def attach_static(grid, static_path):
    """Nearest-join cached satellite features (+elevation) onto the grid."""
    out = grid.copy()
    if static_path and storage.exists(static_path):
        static = storage.read_csv(static_path)
        from sklearn.neighbors import BallTree
        tree = BallTree(np.radians(static[["lat", "lon"]].to_numpy()), metric="haversine")
        _, idx = tree.query(np.radians(grid[["lat", "lon"]].to_numpy()), k=1)
        idx = idx.ravel()
        for col in SATELLITE_FEATURES:
            if col in static.columns:
                out[col] = static[col].to_numpy()[idx]
    missing = [c for c in SATELLITE_FEATURES if c not in out.columns]
    if missing:
        print(f"  WARNING: satellite features not in cache, filling neutral defaults: {missing}")
        if "elevation" in missing:
            try:
                out["elevation"] = fetch_elevation(grid)
                missing.remove("elevation")
            except Exception as e:
                print(f"  elevation API failed ({e}); using 0")
                out["elevation"] = 0.0
                missing.remove("elevation")
        for col in missing:
            out[col] = 0.0
    return out


# --------------------------------------------------------------------------- #
# assemble + predict
# --------------------------------------------------------------------------- #
def assemble(weather, static, target_date):
    df = weather.merge(static, on=["lat", "lon"], how="left")
    doy = pd.Timestamp(target_date).dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0.0
    return df[["lat", "lon"]].copy(), df[FEATURE_ORDER].copy()


def predict(features, model_path):
    import joblib
    model = joblib.load(storage.local_copy(model_path))  # downloads from S3 if needed
    expected = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    assert list(features.columns) == expected, (
        f"Feature mismatch.\n got: {list(features.columns)}\n exp: {expected}"
    )
    preds = model.predict(features.fillna(0.0))
    return np.clip(np.round(preds.astype(float)), 1, 5).astype(int)


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
LABELS = ["Low", "Moderate", "High", "Very High", "Extreme"]
HEX = ["#0001fe", "#00e001", "#ffff00", "#e29f01", "#fe0000"]


def write_outputs(coords, preds, target_date, outdir, features=None):
    res = coords.copy()
    res["prediction"] = preds

    # CSV
    csv_key = f"{outdir}/pred_{target_date}.csv"
    storage.write_dataframe(csv_key, res)

    # GeoJSON points -- ideal for a Leaflet/Mapbox web map for your audience.
    # Attach a few human-readable "why" values per cell (converted from the
    # model's ERA5 units back to friendly units: precip m->mm, temp K->C; wind
    # is already m/s). These let the map explain what the model is seeing.
    # Guarded with round/try so a missing column never breaks output.
    def explain(i):
        props = {}
        if features is not None:
            try:
                props["rain14_mm"] = round(float(features["precip_14day_sum"].iloc[i]) * 1000, 1)
                props["maxtemp_c"] = round(float(features["max_temp"].iloc[i]) - 273.15, 1)
                props["wind_ms"] = round(float(features["max_wind_speed"].iloc[i]), 1)
            except Exception:
                pass
        return props

    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
        "properties": {"risk": int(r.prediction), "label": LABELS[int(r.prediction) - 1],
                       **explain(i)},
    } for i, r in enumerate(res.itertuples())]
    gj_text = json.dumps({"type": "FeatureCollection", "date": str(target_date),
                          "features": feats})
    gj_key = f"{outdir}/geojson/{target_date}.geojson"
    storage.write_text(gj_key, gj_text, content_type="application/json")
    # stable pointer the web map reads ("today's" prediction). Short cache so
    # CloudFront re-fetches it at least hourly -> the public map updates within
    # an hour of each daily run with no manual invalidation needed. (The dated
    # files above are immutable, so they keep the default long cache.)
    storage.write_text(f"{outdir}/latest.geojson", gj_text,
                       content_type="application/json",
                       cache_control="public, max-age=3600")

    # PNG heatmap (same look as testset_visualization.ipynb). matplotlib needs a
    # real path, so render to a temp file, then hand the bytes to storage.
    png_key = f"{outdir}/pngs/{target_date}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        lons = np.sort(res["lon"].unique())
        lats = np.sort(res["lat"].unique())
        grid = (res.pivot(index="lat", columns="lon", values="prediction")
                   .reindex(index=lats[::-1], columns=lons).values)
        cmap = ListedColormap(HEX)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(grid, origin="upper", aspect="equal", cmap=cmap, norm=norm)
        plt.axis("off")
        cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5])
        cbar.ax.set_yticklabels(LABELS)
        cbar.set_label("Fire Danger Level")
        plt.title(f"Fire Danger Map - {target_date}")
        tmp_png = os.path.join(tempfile.gettempdir(), f"pred_{target_date}.png")
        plt.savefig(tmp_png, dpi=200, bbox_inches="tight")
        plt.close()
        storage.upload_file(tmp_png, png_key, content_type="image/png")
    except Exception as e:
        print(f"  PNG render skipped: {e}")
        png_key = None

    return csv_key, gj_key, png_key


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Latest BC wildfire-risk prediction")
    ap.add_argument("--date", default=date_cls.today().isoformat(), help="ISO date (default: today)")
    ap.add_argument("--step", type=float, default=0.5, help="grid spacing in degrees (smaller = finer/slower)")
    ap.add_argument("--static", default="grid_static.csv", help="cached satellite features CSV")
    ap.add_argument("--mask", default=None, help="BC_mask.geojson to clip the grid to BC")
    ap.add_argument("--grid-file", default=None,
                    help="pre-clipped grid CSV (lat,lon) from make_grid.py; "
                         "skips bounding-box generation, no geopandas needed")
    ap.add_argument("--model", default="rf_model.pkl", help="trained model")
    ap.add_argument("--out", default="predictions", help="output directory")
    ap.add_argument("--chunk", type=int, default=40,
                    help="locations per Open-Meteo request (lower = safer for rate limits)")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds to sleep between Open-Meteo requests")
    ap.add_argument("--weather-cache", default=None,
                    help="path to save/resume the raw weather fetch as CSV "
                         "(useful if a run gets rate-limited partway through)")
    args = ap.parse_args(argv)

    if args.grid_file:
        print(f"Loading pre-clipped grid from {args.grid_file}...")
        grid = storage.read_csv(args.grid_file)[["lat", "lon"]].reset_index(drop=True)
    else:
        print(f"Building grid (step={args.step})...")
        grid = build_grid(args.step, mask=args.mask)
    print(f"  {len(grid)} grid points")

    # Per-(date, step) checkpoint. fetch_weather resumes from this automatically
    # and only fetches missing points. Delete it to force a clean refetch.
    cache_path = args.weather_cache or os.path.join(
        args.out, f"weather_{args.date}_step{args.step}.csv")
    print("Fetching weather from Open-Meteo (resumable)...")
    weather = fetch_weather(grid, args.date, chunk=args.chunk, pause=args.pause,
                            checkpoint_path=cache_path)

    print("Attaching satellite cache...")
    static = attach_static(weather[["lat", "lon"]].drop_duplicates(), args.static)

    print("Assembling features...")
    coords, features = assemble(weather, static, args.date)

    if not storage.exists(args.model):
        feat_key = f"{args.out}/features_{args.date}.csv"
        storage.write_dataframe(feat_key, features.assign(lat=coords["lat"], lon=coords["lon"]))
        print(f"Model {args.model!r} not found. Wrote feature frame to {feat_key} "
              f"so you can predict once the model is in place.")
        return

    print("Predicting...")
    preds = predict(features, args.model)
    csv_path, gj_path, png_path = write_outputs(coords, preds, args.date, args.out, features=features)
    print("Done:")
    print(f"  {csv_path}")
    print(f"  {gj_path}")
    if png_path:
        print(f"  {png_path}")


if __name__ == "__main__":
    main()
