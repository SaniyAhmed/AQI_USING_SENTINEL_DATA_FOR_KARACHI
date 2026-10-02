"""
PHASE 2 - STEP 2: Sentinel-2 land cover - NDVI, NDBI and % built-up per district.

Why monthly? Sentinel-2 passes over Karachi only every 5 days, and vegetation
and buildings change slowly. A monthly cloud-free composite is stable and still
captures seasons (monsoon greening, dry winter).

What happens on Earth Engine for each month
  1. Take every Sentinel-2 L2A scene over Karachi in that month.
  2. Cloud masking with the Scene Classification Layer (SCL). Keep only:
        4 vegetation, 5 bare/built, 6 water, 7 unclassified
     Drop: 0 no data, 1 saturated, 2 dark/shadow, 3 cloud shadow,
           8 cloud (medium prob.), 9 cloud (high prob.), 10 thin cirrus, 11 snow
  3. Per scene: NDVI = (B8 - B4) / (B8 + B4)     vegetation
                NDBI = (B11 - B8) / (B11 + B8)   built-up / bare surfaces
  4. Median over the month's clear pixels.
  5. % built-up from Google Dynamic World (share of observations labelled "built").
  6. Averaged from 100 m to our 1 km grid (with the share of clear 100 m pixels).

WHY THIS STEP IS LEAK-FREE (no future information in any daily row)
  * A composite is only complete when its month is over, so day D uses the
    composite of the PREVIOUS calendar month (landcover.csv, column source_month).
  * A 1 km cell with < S2_MIN_CELL_CLEAR clear pixels (monsoon cloud) is filled
    from EARLIER months only (at most S2_MAX_FILL_MONTHS old), never by
    interpolating towards a later month. `filled_fraction` / `lc_age_days` say so.
  * The built-up weights for urban_mean use a fixed reference period that ends
    before START_DATE (config.BUILT_WEIGHT_PERIOD), so they never change when
    new data arrives.

The monsoon cloud gap is physical: Sentinel-2's own SCL and the independent
s2cloudless detector agree that ~78 % of Karachi's August pixels are cloudy,
so there is nothing more to recover from the sensor. Months are fetched once and cached
(data/cache/gee/sentinel2/) so this step is quick when repeated.

Outputs
  data/processed/district_monthly/sentinel2.csv      one row per month and district
  data/processed/district_daily/landcover.csv        daily table (previous month's composite)
  data/processed/satellite/sentinel2/sentinel2_monthly.nc   1 km monthly grids
  data/processed/grid/built_fraction_1000m.tif       weights for urban_mean

Run:
    python phase2_ingestion/step2_sentinel2_landcover.py
(--start/--end are accepted for compatibility but ignored: this step always
covers the whole period so its outputs are never cut short by a partial run.)
"""
import datetime as dt
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ee
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import config
from cache_io import atomic_csv, atomic_netcdf, atomic_write, pipeline_lock, quarantine
from cli import parse_dates
from gee_utils import fetch_array, init_ee, region
from spatial_utils import load_grid, zonal_timeseries

BANDS = ["ndvi", "ndbi", "built_pct", "clear_obs"]
KEEP_SCL = [4, 5, 6, 7]
FACTOR = 10               # 100 m -> 1 km
S2_SCHEMA = 2
GRID_NAMES = ["ndvi", "ndbi", "built_pct", "clear_share"]


# ----------------------------------------------------------------------------
# Earth Engine: one month
# ----------------------------------------------------------------------------
def s2_month(start, end, geom):
    def prep(img):
        clear = img.select("SCL").remap(KEEP_SCL, [1] * len(KEEP_SCL), 0)
        refl = img.select(["B4", "B8", "B11"]).updateMask(clear)
        ndvi = refl.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndbi = refl.normalizedDifference(["B11", "B8"]).rename("ndbi")
        return ndvi.addBands(ndbi).addBands(clear.rename("clear"))

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom).filterDate(start, end).map(prep))
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(geom).filterDate(start, end)
          .map(lambda img: img.select("label").eq(6).rename("built")))
    return (s2.select(["ndvi", "ndbi"]).median()
            .addBands(dw.mean().multiply(100).rename("built_pct"))
            .addBands(s2.select("clear").sum().rename("clear_obs"))
            .rename(BANDS))


def block_mean(arr, factor=FACTOR):
    """Average factor x factor blocks (100 m -> 1 km), ignoring NaN."""
    r, c = arr.shape
    blocks = arr.reshape(r // factor, factor, c // factor, factor)
    with warnings.catch_warnings():          # all-NaN blocks are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def cache_file(period):
    return config.GEE_CACHE_DIR / "sentinel2" / f"s2_{period}.npz"


def load_cached(period, shape):
    path = cache_file(period)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh, np.load(fh, allow_pickle=False) as z:   # own handle: always closed
            ok = int(z["schema"]) == S2_SCHEMA and str(z["month"]) == str(period)
            grids = z["grids"]
    except Exception as err:
        quarantine(path, f"unreadable ({type(err).__name__})")
        return None
    if not ok or grids.shape != (4, *shape) or grids.dtype != np.float32:
        quarantine(path, "wrong layout")
        return None
    return grids


def fetch_month_grids(period, geom, shape):
    """4 x rows x cols float32 at 1 km: ndvi, ndbi, built_pct, clear_share."""
    cached = load_cached(period, shape)
    if cached is not None:
        return cached
    start = period.start_time.strftime("%Y-%m-%d")
    end = (period.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cube, _ = fetch_array(s2_month(start, end, geom), 100)
    ndvi, ndbi, built, clear_obs = cube
    has_clear = np.isfinite(clear_obs) & (clear_obs > 0)
    ndvi = np.where(has_clear, ndvi, np.nan)
    ndbi = np.where(has_clear, ndbi, np.nan)
    grids = np.stack([block_mean(ndvi), block_mean(ndbi), block_mean(built),
                      block_mean(has_clear.astype("float32"))]).astype("float32")
    if grids.shape != (4, *shape):
        raise RuntimeError(f"Sentinel-2 {period}: got {grids.shape}, expected {(4, *shape)}")
    if (dt.date.today() - period.end_time.date()).days >= config.CACHE_AFTER_DAYS:
        atomic_write(cache_file(period), lambda p: np.savez_compressed(
            p, schema=np.int16(S2_SCHEMA), month=np.str_(str(period)), grids=grids))
    return grids


# ----------------------------------------------------------------------------
# Causal cell-level fill (earlier months only)
# ----------------------------------------------------------------------------
def causal_fill(monthly):
    """
    monthly: list of (4, rows, cols) arrays in time order.
    A cell is "real" when it has enough clear pixels this month. Cloudy cells
    take the last real value, if it is at most S2_MAX_FILL_MONTHS months old.
    Returns filled (months, 3, r, c), real (months, r, c) bool, age (months, r, c).
    """
    n = len(monthly)
    shape = monthly[0].shape[1:]
    last_val = np.full((3, *shape), np.nan, dtype="float32")
    last_age = np.full((3, *shape), np.inf)
    filled = np.full((n, 3, *shape), np.nan, dtype="float32")
    real_all = np.zeros((n, *shape), dtype=bool)
    age_all = np.full((n, *shape), np.nan, dtype="float32")
    for i, g in enumerate(monthly):
        optical_ok = g[3] >= config.S2_MIN_CELL_CLEAR
        for v in range(3):
            real = np.isfinite(g[v]) & (optical_ok if v < 2 else True)
            last_val[v] = np.where(real, g[v], last_val[v])
            last_age[v] = np.where(real, 0, last_age[v] + 1)
            usable = last_age[v] <= config.S2_MAX_FILL_MONTHS
            filled[i, v] = np.where(usable, last_val[v], np.nan)
            if v == 0:
                real_all[i] = real
                age_all[i] = np.where(usable, last_age[v], np.nan)
    return filled, real_all, age_all


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def last_complete_month():
    today = pd.Timestamp(dt.date.today())
    return (today.to_period("M") - 1)


def main():
    parse_dates(__doc__.splitlines()[1])             # only for --help / compatibility
    init_ee()
    geom = region()
    grid1k = load_grid(1000)
    shape = grid1k["shape"]

    first = min(pd.Period(config.BUILT_WEIGHT_PERIOD[0], "M"),
                pd.Timestamp(config.START_DATE).to_period("M") - config.S2_WARMUP_MONTHS)
    last = last_complete_month()
    months = pd.period_range(first, last, freq="M")
    print(f"Sentinel-2 monthly composites: {months[0]} -> {months[-1]} ({len(months)} months, "
          f"cached months are reused)")

    def job(period):
        grids = fetch_month_grids(period, geom, shape)
        print(f"  {period}: cells with enough clear sky: {np.mean(grids[3] >= config.S2_MIN_CELL_CLEAR):.0%}")
        return grids

    with ThreadPoolExecutor(max_workers=config.GEE_WORKERS_S2) as pool:
        monthly = list(pool.map(job, months))

    filled, real, age = causal_fill(monthly)
    filled_cube = [filled[:, v] for v in range(3)]
    valid = [np.isfinite(c) for c in filled_cube]

    # ---- district table (monthly), 1 km fraction weights like every daily product
    stats = [zonal_timeseries(c, grid1k) for c in filled_cube]                 # mean, std, valid_fraction
    clear_frac = zonal_timeseries(np.where(real, 1.0, np.nan).astype("float32"), grid1k)[2]
    rows = []
    for i, period in enumerate(months):
        for k, (did, name) in enumerate(zip(grid1k["ids"], grid1k["names"])):
            rows.append({
                "month": period.start_time, "district_id": did, "district": name,
                "ndvi": stats[0][0][i, k], "ndbi": stats[1][0][i, k], "built_pct": stats[2][0][i, k],
                "clear_fraction": clear_frac[i, k],                  # area with real observations
                "valid_fraction": stats[0][2][i, k],                 # area with a value after causal fill
                "filled_fraction": max(stats[0][2][i, k] - clear_frac[i, k], 0.0),
            })
    table = pd.DataFrame(rows)
    if table.duplicated(["month", "district_id"]).any():
        raise RuntimeError("duplicate month/district rows in the Sentinel-2 table")
    config.DISTRICT_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, config.DISTRICT_MONTHLY_DIR / "sentinel2.csv", float_format="%.5g")

    # ---- daily table: day D gets the composite of the previous calendar month
    by_month = table.set_index(["month", "district_id"])
    end_day = min(pd.Timestamp(dt.date.today()) - pd.Timedelta(days=1),
                  (last + 1).end_time.normalize())
    days = pd.date_range(config.START_DATE, end_day, freq="D")
    daily = []
    for d in days:
        src = d.to_period("M") - 1
        src_start = src.start_time
        src_end = src.end_time.normalize()
        for did, name in zip(grid1k["ids"], grid1k["names"]):
            r = by_month.loc[(src_start, did)]
            daily.append({
                "date": d, "district_id": did, "district": name,
                "ndvi": r["ndvi"], "ndbi": r["ndbi"], "built_pct": r["built_pct"],
                "source_month": str(src), "lc_age_days": (d - src_end).days,
                "lc_valid_fraction": r["valid_fraction"], "lc_filled_fraction": r["filled_fraction"],
            })
    daily = pd.DataFrame(daily)
    config.DISTRICT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(daily, config.DISTRICT_DAILY_DIR / "landcover.csv", float_format="%.5g")

    # ---- monthly 1 km grids: ONE file (no overlapping range files)
    t = grid1k["transform"]
    rows_n, cols_n = shape
    x = t.c + t.a * (np.arange(cols_n) + 0.5)
    y = t.f + t.e * (np.arange(rows_n) + 0.5)
    ds = xr.Dataset(
        {"ndvi": (("month", "y", "x"), filled[:, 0]), "ndbi": (("month", "y", "x"), filled[:, 1]),
         "built_pct": (("month", "y", "x"), filled[:, 2]),
         "age_months": (("month", "y", "x"), age)},
        coords={"month": [p.start_time for p in months], "y": y, "x": x},
        attrs={"crs": config.CRS_UTM, "source": "Sentinel-2 L2A (SCL masked), Dynamic World",
               "note": "cloudy cells filled from earlier months only; age_months = 0 means observed"})
    out_dir = config.SATELLITE_DIR / "sentinel2"
    atomic_netcdf(ds, out_dir / "sentinel2_monthly.nc",
                  encoding={v: {"zlib": True} for v in ds.data_vars})
    for old in out_dir.glob("sentinel2_[0-9]*.nc"):     # superseded range files from earlier versions
        old.unlink()

    # ---- built-up weights from the fixed reference period (ends before START_DATE)
    ref = [i for i, p in enumerate(months)
           if config.BUILT_WEIGHT_PERIOD[0] <= str(p) <= config.BUILT_WEIGHT_PERIOD[1]]
    if not ref or pd.Period(config.BUILT_WEIGHT_PERIOD[1], "M").end_time >= pd.Timestamp(config.START_DATE):
        raise SystemExit("BUILT_WEIGHT_PERIOD must be a period that ends before START_DATE.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        built_frac = np.nan_to_num(np.nanmean(filled[ref, 2], axis=0) / 100, nan=0.0).astype("float32")
    profile = {"driver": "GTiff", "height": rows_n, "width": cols_n, "count": 1,
               "dtype": "float32", "crs": config.CRS_UTM, "transform": t, "compress": "deflate"}

    def write_tif(path):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(built_frac, 1)
    atomic_write(config.BUILT_WEIGHT_FILE, write_tif)

    summary = table.groupby("district")[["ndvi", "ndbi", "built_pct", "clear_fraction",
                                         "valid_fraction", "filled_fraction"]].mean()
    print("\nDistrict averages over all months:")
    print(summary.round(3).to_string())
    print(f"\nSaved {config.DISTRICT_MONTHLY_DIR / 'sentinel2.csv'}\n      "
          f"{config.DISTRICT_DAILY_DIR / 'landcover.csv'}\n      {out_dir}\n      {config.BUILT_WEIGHT_FILE}")


if __name__ == "__main__":
    with pipeline_lock():
        main()
