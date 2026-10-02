"""
PHASE 2 - STEP 5: Quality report - how much real data do we have, and does
the gap filling look sensible?

Creates in outputs/figures/:
  phase2_coverage.png      share of district area with REAL observations,
                           per product and month (monsoon gaps show up here)
  phase2_timeseries.png    district daily series per product (after filling)
  phase2_gapfill_<p>.png   one day's map: raw vs gap-filled vs flags

Run:
    python phase2_ingestion/step5_quality_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

import config
from products import PRODUCTS


def coverage_figure(tables):
    fig, axes = plt.subplots(len(tables), 1, figsize=(14, 2.2 * len(tables)), squeeze=False)
    for ax, (key, t) in zip(axes[:, 0], tables.items()):
        m = (t.assign(month=t["date"].dt.to_period("M").dt.to_timestamp())
             .pivot_table(index="district", columns="month", values="observed_fraction"))
        im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(m.index)), m.index, fontsize=7)
        step = max(1, len(m.columns) // 12)
        ax.set_xticks(range(0, len(m.columns), step),
                      [c.strftime("%Y-%m") for c in m.columns[::step]], fontsize=7)
        ax.set_title(f"{key.upper()}: share of district with real observations (monthly mean)", fontsize=9)
        fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "phase2_coverage.png", dpi=110)
    plt.close(fig)


def timeseries_figure(tables):
    fig, axes = plt.subplots(len(tables), 1, figsize=(14, 2.6 * len(tables)), squeeze=False)
    for ax, (key, t) in zip(axes[:, 0], tables.items()):
        for name, g in t.groupby("district"):
            ax.plot(g["date"], g["mean"].rolling(7, min_periods=1).mean(), lw=0.9, label=name)
        ax.set_title(f"{key.upper()} district mean (7-day smoothed for display)", fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "phase2_timeseries.png", dpi=110)
    plt.close(fig)


def gapfill_figure(key, districts):
    files = sorted((config.SATELLITE_DIR / key).glob(f"{key}_*.nc"))
    if not files:
        return
    ds = xr.concat([xr.load_dataset(f) for f in files], dim="time")
    flags = ds["flag"].values
    inside = flags != 9
    # Pick the day where the most cells were gap-filled, to show the method at work.
    filled_share = ((flags == 1) | (flags == 2)).sum(axis=(1, 2)) / inside[0].sum()
    t = int(np.argmax(filled_share))
    value, flag = ds["value"].values[t], flags[t]
    raw = np.where(flag == 0, value, np.nan)
    extent = [float(ds.x.min()) - 500, float(ds.x.max()) + 500,
              float(ds.y.min()) - 500, float(ds.y.max()) + 500]
    vmin, vmax = np.nanpercentile(value[inside[t]], [2, 98]) if np.isfinite(value).any() else (0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, arr, title in [(axes[0], raw, "Raw (after quality filter)"),
                           (axes[1], np.where(inside[t], value, np.nan), "After gap filling")]:
        im = ax.imshow(arr, extent=extent, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.7, label=ds.attrs.get("units", ""))
        ax.set_title(title)
    cmap = ListedColormap(["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
    axes[2].imshow(np.where(inside[t], flag, np.nan), extent=extent, cmap=cmap, vmin=-0.5, vmax=3.5,
                   interpolation="nearest")
    axes[2].set_title("Flags: green observed, blue IDW,\norange temporal, red still missing")
    for ax in axes:
        districts.boundary.plot(ax=ax, color="white" if ax is not axes[2] else "black", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
    day = pd.Timestamp(ds.time.values[t]).date()
    fig.suptitle(f"{key.upper()} on {day}: {filled_share[t]:.0%} of district cells gap-filled")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"phase2_gapfill_{key}.png", dpi=110)
    plt.close(fig)


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for key in PRODUCTS:                          # daily satellite products only (landcover.csv has other columns)
        path = config.DISTRICT_DAILY_DIR / f"{key}.csv"
        if path.exists():
            tables[key] = pd.read_csv(path, parse_dates=["date"])
    if not tables:
        sys.exit("No district tables yet - run steps 3 and 4 first.")

    print("Average share of district area per day (all districts, whole period):")
    print(f"  {'product':8s} {'observed':>9s} {'+IDW':>7s} {'+temporal':>10s} {'final':>7s}  days")
    for key, t in tables.items():
        print(f"  {key:8s} {t['observed_fraction'].mean():9.0%} {t['idw_fraction'].mean():7.0%} "
              f"{t['temporal_fraction'].mean():10.0%} {t['valid_fraction'].mean():7.0%}  "
              f"{t['date'].nunique()}")

    districts = gpd.read_file(config.BOUNDARY_DIR / "karachi_districts_utm42n.gpkg")
    coverage_figure(tables)
    timeseries_figure(tables)
    for key in tables:
        gapfill_figure(key, districts)
    print(f"\nFigures saved in {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
