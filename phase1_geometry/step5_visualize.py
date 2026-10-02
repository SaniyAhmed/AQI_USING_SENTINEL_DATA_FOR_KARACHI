"""
PHASE 1 - STEP 5: Draw maps to check everything by eye.

Creates outputs/figures/phase1_overview.png with four panels:
  1. District polygons, centroids, the proposed box and the download box
  2. The 1 km district-id grid (which district each cell belongs to)
  3. The 1 km fraction mask for Karachi Central (shows partial edge cells)
  4. The fake NO2 test from step 4, averaged per district

Open the PNG and check: are the six districts in the right place? Do the
grid colours line up with the polygon outlines?

Run:
    python phase1_geometry/step5_visualize.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to a file; Anaconda's default Qt window backend crashes here
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

COLORS = ["#ffffff", "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def add_box(ax, box, color, label):
    ax.add_patch(Rectangle((box["west"], box["south"]), box["east"] - box["west"],
                           box["north"] - box["south"], fill=False, edgecolor=color,
                           linestyle="--", linewidth=1.5, label=label))


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    wgs = gpd.read_file(config.BOUNDARY_DIR / "karachi_districts_wgs84.geojson")
    utm = gpd.read_file(config.BOUNDARY_DIR / "karachi_districts_utm42n.gpkg")
    cent = pd.read_csv(config.BOUNDARY_DIR / "district_centroids.csv")
    bbox = pd.read_json(config.BOUNDARY_DIR / "master_bbox.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))

    # Panel 1: polygons + boxes (lat/lon)
    ax = axes[0, 0]
    wgs.plot(ax=ax, color=[COLORS[i] for i in wgs["district_id"]], edgecolor="black", alpha=0.6)
    ax.scatter(cent["centroid_lon"], cent["centroid_lat"], c="black", s=20, zorder=3)
    for _, r in cent.iterrows():
        ax.annotate(r["district"], (r["centroid_lon"], r["centroid_lat"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    add_box(ax, bbox["proposed_bbox"], "red", "Proposed bbox")
    add_box(ax, bbox["download_bbox"], "blue", "Download bbox (used)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("1. Districts (WGS84) and bounding boxes")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # Panel 2: 1 km district id grid (UTM)
    ax = axes[0, 1]
    with rasterio.open(config.GRID_DIR / "district_id_1000m.tif") as src:
        ids = src.read(1)
        b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]
    ax.imshow(ids, extent=extent, cmap=ListedColormap(COLORS), vmin=0, vmax=6,
              interpolation="nearest")
    utm.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title("2. 1 km district-id grid (UTM 42N)")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")

    # Panel 3: fraction mask for Karachi Central, zoomed in
    ax = axes[1, 0]
    with rasterio.open(config.GRID_DIR / "district_fraction_1000m.tif") as src:
        frac = src.read(1)  # band 1 = Karachi Central
    im = ax.imshow(np.where(frac > 0, frac, np.nan), extent=extent, cmap="viridis",
                   vmin=0, vmax=1, interpolation="nearest")
    central = utm[utm["district_id"] == 1]
    central.boundary.plot(ax=ax, color="red", linewidth=1)
    x0, y0, x1, y1 = central.total_bounds
    ax.set_xlim(x0 - 3000, x1 + 3000); ax.set_ylim(y0 - 3000, y1 + 3000)
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of 1 km cell inside district")
    ax.set_title("3. Fraction mask - Karachi Central (zoomed)")

    # Panel 4: result of the fake NO2 test
    ax = axes[1, 1]
    test_file = config.GRID_DIR / "test_zonal_no2.csv"
    if test_file.exists():
        res = pd.read_csv(test_file)
        merged = wgs.merge(res[["district_id", "mean", "valid_fraction"]], on="district_id")
        merged.plot(ax=ax, column="mean", cmap="YlOrRd", edgecolor="black", legend=True,
                    legend_kwds={"label": "fake NO2 district mean", "shrink": 0.8})
        for _, r in merged.iterrows():
            p = r.geometry.representative_point()
            ax.annotate(f"{r['mean']:.0f}\n({r['valid_fraction']:.0%} valid)", (p.x, p.y),
                        fontsize=7, ha="center")
        ax.set_title("4. Zonal-stats test (fake NO2 from step 4)")
    else:
        ax.text(0.5, 0.5, "Run step4_test_zonal_stats.py first", ha="center")

    fig.tight_layout()
    out = config.FIGURE_DIR / "phase1_overview.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
