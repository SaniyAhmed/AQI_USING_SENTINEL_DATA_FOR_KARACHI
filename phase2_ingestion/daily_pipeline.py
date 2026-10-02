"""
daily_pipeline.py — the engine shared by the Sentinel-5P and MODIS steps.

For one product and a date range it:
  1. downloads daily 1 km grids from Earth Engine, one month per request
     (cached in data/cache/gee/ so interrupted runs can resume)
  2. gap-fills them (spatial IDW, then 3-day temporal median)
  3. averages them per district (normal and built-up-weighted)
  4. saves
       data/processed/satellite/<product>/<product>_YYYY-MM.nc   1 km grids
       data/processed/district_daily/<product>.csv               district table
"""
import calendar
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import config
from cache_io import atomic_csv, atomic_netcdf, load_month_cache, save_month_cache
from gapfill import FLAG_IDW, FLAG_OBSERVED, FLAG_TEMPORAL, gap_fill_cube
from gee_utils import fetch_array, region
from spatial_utils import load_grid, zonal_timeseries

RES_M = 1000


# ----------------------------------------------------------------------------
# Download (one month = one Earth Engine request)
# ----------------------------------------------------------------------------
def month_days(year, month):
    """All dates of a month up to yesterday."""
    yesterday = dt.date.today() - dt.timedelta(days=1)
    last = min(dt.date(year, month, calendar.monthrange(year, month)[1]), yesterday)
    return pd.date_range(dt.date(year, month, 1), last, freq="D")


def apply_valid_range(cube, product):
    """Turn pixels outside the product's physical range into NaN (missing)."""
    lo, hi = product["valid_range"]
    cube = cube.copy()
    cube[(cube < lo) | (cube > hi)] = np.nan
    return cube


def fetch_month(key, product, year, month, geom):
    """Daily grids for one month -> (dates, cube[days, rows, cols]), out-of-range pixels removed."""
    dates, cube = _fetch_month_raw(key, product, year, month, geom)
    return dates, (apply_valid_range(cube, product) if cube.size else cube)


def _fetch_month_raw(key, product, year, month, geom):
    cache = config.GEE_CACHE_DIR / key / f"{key}_{year}-{month:02d}.npz"
    dates = month_days(year, month)
    if len(dates) == 0:
        return dates, np.empty((0,) + load_grid(RES_M)["shape"], dtype="float32")
    grid_shape = load_grid(RES_M)["shape"]

    # A cached month is used only if it passes every check in cache_io
    # (readable, right days, right grid size); otherwise it is quarantined.
    cached = load_month_cache(cache, dates, grid_shape)
    if cached is not None:
        return cached

    # One band per day, named d20240131 etc., stacked into a single image.
    bands = [product["build_day"](ee.Date(d.strftime("%Y-%m-%d")), geom)
             .rename(d.strftime("d%Y%m%d")) for d in dates]
    cube, names = fetch_array(ee.Image.cat(bands), RES_M)
    if cube.shape != (len(dates), *grid_shape) or len(names) != len(dates):
        raise RuntimeError(f"{key} {year}-{month:02d}: Earth Engine returned {cube.shape}, "
                           f"expected {(len(dates), *grid_shape)}")

    month_end = dt.date(year, month, calendar.monthrange(year, month)[1])
    if (dt.date.today() - month_end).days >= config.CACHE_AFTER_DAYS:
        save_month_cache(cache, dates, cube)            # atomic: never leaves a partial file
    return dates, cube


def load_days(key, product, start, end):
    """Daily cube covering [start, end], fetching months in parallel."""
    months = pd.period_range(start, end, freq="M")
    geom = region()
    print(f"  Fetching {len(months)} month(s) from Earth Engine ...")

    def job(period):
        dates, cube = fetch_month(key, product, period.year, period.month, geom)
        valid = np.isfinite(cube).mean() if cube.size else 0
        print(f"    {period}: {len(dates)} days, {valid:.0%} of grid cells observed")
        return dates, cube

    with ThreadPoolExecutor(max_workers=config.GEE_WORKERS) as pool:
        parts = list(pool.map(job, months))
    parts = [p for p in parts if len(p[0])]
    dates = pd.DatetimeIndex(np.concatenate([p[0] for p in parts]))
    cube = np.concatenate([p[1] for p in parts])
    order = np.argsort(dates)
    dates, cube = dates[order], cube[order]
    # Guard against duplicated or missing days before anything is computed.
    if dates.has_duplicates:
        raise RuntimeError(f"{key}: duplicated dates after joining months: {dates[dates.duplicated()][:5]}")
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected):
        raise RuntimeError(f"{key}: missing days {expected.difference(dates)[:5]} after joining months")
    keep = (dates >= start) & (dates <= end)
    return dates[keep], cube[keep]


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def grid_coords(grid):
    t = grid["transform"]
    rows, cols = grid["shape"]
    x = t.c + t.a * (np.arange(cols) + 0.5)
    y = t.f + t.e * (np.arange(rows) + 0.5)
    return x, y


def save_grids(key, product, dates, filled, flags, grid):
    """
    One small NetCDF per month: gap-filled value + flag for every cell.
    If the month file already exists (an earlier run covered other days of the
    same month) the days are merged, never truncated. Written atomically.
    """
    x, y = grid_coords(grid)
    out_dir = config.SATELLITE_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    periods = dates.to_period("M")
    for period in periods.unique():
        sel = periods == period
        ds = xr.Dataset(
            {"value": (("time", "y", "x"), filled[sel]),
             "flag": (("time", "y", "x"), flags[sel])},
            coords={"time": dates[sel], "y": y, "x": x},
            attrs={"product": key, "units": product["units"], "crs": config.CRS_UTM,
                   "description": product["description"],
                   "flag_meaning": "0 observed, 1 IDW spatial fill, 2 temporal median fill, "
                                   "3 missing, 9 outside districts"},
        )
        path = out_dir / f"{key}_{period}.nc"
        if path.exists():
            try:
                with xr.open_dataset(path) as old:
                    old = old.load()
                old = old.sel(time=~old.time.isin(ds.time))
                if old.sizes["time"]:
                    ds = xr.concat([old, ds], dim="time", combine_attrs="override").sortby("time")
            except Exception:
                pass                                     # unreadable old file: the new one replaces it
        enc = {"value": {"zlib": True, "complevel": 4}, "flag": {"zlib": True, "complevel": 4}}
        atomic_netcdf(ds, path, encoding=enc)


def save_district_table(key, table):
    """Merge new rows into data/processed/district_daily/<key>.csv."""
    config.DISTRICT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DISTRICT_DAILY_DIR / f"{key}.csv"
    if path.exists():
        old = pd.read_csv(path, parse_dates=["date"])
        old = old[~old["date"].isin(table["date"])]
        table = pd.concat([old, table])
    table = table.sort_values(["date", "district_id"]).reset_index(drop=True)
    if table.duplicated(["date", "district_id"]).any():
        raise RuntimeError(f"{key}: duplicate (date, district) rows - refusing to save")
    atomic_csv(table, path, float_format="%.6g")
    return path


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def run_product(key, product, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    print(f"\n=== {key.upper()} ({product['source']}: {product['description']}) "
          f"{start.date()} -> {end.date()} ===")
    grid = load_grid(RES_M)
    targets = grid["weights"].sum(axis=0) > 0          # cells inside any district

    # Fetch a few extra days before `start` so the backward temporal window
    # has donors on the first day (and after `end` for the centered mode).
    # In "backward" mode nothing after `end` is ever read, so no future day is fetched.
    pad = config.TEMPORAL_FILL_WINDOW
    pad_after = pad if config.TEMPORAL_FILL_MODE == "centered" else 0
    dates, raw = load_days(key, product, start - pd.Timedelta(days=pad),
                           end + pd.Timedelta(days=pad_after))
    if len(dates) == 0:
        print("  No days to process.")
        return

    print("  Gap filling (IDW + temporal median) ...")
    filled, flags = gap_fill_cube(raw, targets, RES_M, product["idw_radius_km"] * 1000,
                                  mode=config.TEMPORAL_FILL_MODE,
                                  window=config.TEMPORAL_FILL_WINDOW)
    keep = (dates >= start) & (dates <= end)
    dates, raw, filled, flags = dates[keep], raw[keep], filled[keep], flags[keep]

    # District statistics
    mean, std, final_frac = zonal_timeseries(filled, grid)
    _, _, obs_frac = zonal_timeseries(raw, grid)
    idw_frac = zonal_timeseries(np.where(flags == FLAG_IDW, 1.0, np.nan), grid)[2]
    tmp_frac = zonal_timeseries(np.where(flags == FLAG_TEMPORAL, 1.0, np.nan), grid)[2]
    if not config.BUILT_WEIGHT_FILE.exists():
        raise SystemExit(f"{config.BUILT_WEIGHT_FILE} is missing. Run step2_sentinel2_landcover.py first "
                         "(run_phase2.py does this automatically).")
    with rasterio.open(config.BUILT_WEIGHT_FILE) as src:
        built = src.read(1)
    urban_mean = zonal_timeseries(filled, grid, extra_weight=built)[0]

    rows = []
    for i, d in enumerate(dates):
        for k, (did, name) in enumerate(zip(grid["ids"], grid["names"])):
            rows.append({
                "date": d, "district_id": did, "district": name,
                "mean": mean[i, k], "std": std[i, k],
                "urban_mean": urban_mean[i, k],
                "observed_fraction": obs_frac[i, k],
                "idw_fraction": idw_frac[i, k],
                "temporal_fraction": tmp_frac[i, k],
                "valid_fraction": final_frac[i, k],
            })
    table = pd.DataFrame(rows)
    save_grids(key, product, dates, filled, flags, grid)
    path = save_district_table(key, table)

    # Summary
    s = table.groupby("district")[["observed_fraction", "valid_fraction"]].mean()
    s["days_with_value"] = table.dropna(subset=["mean"]).groupby("district").size()
    print(f"  Average share of each district covered: observed (raw) vs after gap filling")
    print((s.round(2)).to_string())
    print(f"  Saved {path.name} and monthly grids in {config.SATELLITE_DIR / key}")
    return table
