#!/usr/bin/env python3
"""
make_grid.py
============
One-off: generate the BC prediction grid, clip it to the actual province
outline in BC_mask.geojson, and save the surviving lat/lon points to a CSV.

The daily job then loads this fixed point list (via --grid-file) instead of
generating a rectangular bounding box -- so the map is BC-shaped, and the
container never needs geopandas/GDAL.

Clipping uses matplotlib's point-in-polygon test (matplotlib is already a
dependency), so this needs NO geopandas install.

Run locally:
    python make_grid.py --step 0.5 --mask BC_mask.geojson --output bc_grid.csv
"""
import argparse
import json

import numpy as np
from matplotlib.path import Path

from daily_extract2 import build_grid


def load_polygons(geojson_path):
    """Return a list of (exterior_ring, [hole_rings]) as arrays of [lon, lat]."""
    gj = json.load(open(geojson_path))
    geoms = []
    feats = gj.get("features", [gj]) if gj.get("type") == "FeatureCollection" else [gj]
    for f in feats:
        geom = f.get("geometry", f)
        t = geom["type"]
        if t == "Polygon":
            polys = [geom["coordinates"]]
        elif t == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for rings in polys:
            ext = np.array(rings[0])
            holes = [np.array(r) for r in rings[1:]]
            geoms.append((ext, holes))
    return geoms


def clip(grid, geojson_path):
    pts = grid[["lon", "lat"]].to_numpy()  # matplotlib wants (x=lon, y=lat)
    inside = np.zeros(len(pts), dtype=bool)
    for ext, holes in load_polygons(geojson_path):
        in_ext = Path(ext).contains_points(pts)
        for h in holes:                      # remove lakes / interior holes
            in_ext &= ~Path(h).contains_points(pts)
        inside |= in_ext
    return grid[inside].reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-clip the BC grid to the province outline")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--mask", default="BC_mask.geojson")
    ap.add_argument("--output", default="bc_grid.csv")
    args = ap.parse_args(argv)

    full = build_grid(args.step, mask=None)
    clipped = clip(full, args.mask)
    clipped[["lat", "lon"]].to_csv(args.output, index=False)
    print(f"full grid:    {len(full)} points")
    print(f"clipped grid: {len(clipped)} points  ({len(full) - len(clipped)} removed)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
