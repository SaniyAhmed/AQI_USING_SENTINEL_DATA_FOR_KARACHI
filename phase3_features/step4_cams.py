"""
PHASE 3 - STEP 4: CAMS model aerosol (AOD, dust AOD, PM2.5, PM10) - gap-free, leak-free.

Why: in July-August MODIS AOD has real retrievals on only ~5 % of the area (clouds). CAMS
(Copernicus Atmosphere Monitoring Service, ECMWF, ~0.4 deg) is a global aerosol model that
assimilates satellite AOD, so it has NO gaps, and its forecasts exist for the days we must
predict. Bands read from Earth Engine (ECMWF/CAMS/NRT):
    aod550       total aerosol optical depth at 550 nm
    dust_aod550  dust aerosol optical depth at 550 nm   (Karachi's dust storms)
    pm25, pm10   surface particulate matter, ug/m3

STRICT TIME-STAMPING (the leakage rule)
    The forecast is issued on the morning of day D. Then the CAMS 12Z run of day D-1 exists (ready
    ~22:00 UTC the evening before) but the 00Z run of day D does NOT (ready ~10:00 UTC).
    So every value is keyed by the RUN it came from:

        init_date I  = the date of a 12Z run
        lead_day k   = 0 .. 3    ->  the local day  I + 1 + k   (target_date)

    and a feature for issue day D and target day D+h always uses  I = D-1, k = h.
    A CAMS value for day D-j (a lag) uses the run I = D-j-1, k = 0 - a run that started before D.
    Because the run date is stored in the table, the validation step can prove no feature
    uses a run issued on or after the issue date.

    Local day L = UTC [L-1 19:00, L 19:00) contains 8 three-hourly steps valid at
    21:00, 00, 03, 06, 09, 12, 15, 18 UTC, i.e. lead hours 24k+9 ... 24k+30 after the 12Z start.

Output: data/processed/met/cams_daily.csv
        init_date, lead_day, target_date, district_id, district, aod550, dust_aod550, pm25, pm10, n_steps

Run:  python phase3_features/step4_cams.py
"""
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase2_ingestion"))
import ee
import numpy as np
import pandas as pd

import config
import sources
from cache_io import atomic_csv, load_month_cache, pipeline_lock, save_month_cache
from cli import parse_dates
from gee_utils import fetch_array, init_ee

BANDS = {"aod550": "total_aerosol_optical_depth_at_550nm_surface",
         "dust_aod550": "dust_aerosol_optical_depth_at_550nm_surface",
         "pm25": "particulate_matter_d_less_than_25_um_surface",
         "pm10": "particulate_matter_d_less_than_10_um_surface"}
SCALE = {"aod550": 1.0, "dust_aod550": 1.0, "pm25": 1e9, "pm10": 1e9}      # kg/m3 -> ug/m3 for PM
STATS = list(BANDS) + ["n_steps"]
LEADS = config.CAMS_LEADS
CAMS = "ECMWF/CAMS/NRT"
STEPS_PER_DAY = 8


def lead_day_image(init_date, k):
    """Mean over the 8 three-hourly steps of local day init_date+1+k, from the 12Z run of init_date."""
    run = f"{init_date:%Y-%m-%d}T{config.CAMS_INIT_HOUR:02d}:00:00"
    start = ee.Date(init_date.strftime("%Y-%m-%d")).advance(24 * k + 24 - config.LOCAL_UTC_OFFSET_H, "hour")
    coll = (ee.ImageCollection(CAMS).filter(ee.Filter.eq("model_initialization_datetime", run))
            .filterDate(start, start.advance(24, "hour")))
    empty = ee.Image.constant([0.0] * len(BANDS)).rename(list(BANDS.values())).updateMask(0)   # run missing entirely
    mean = ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.select(list(BANDS.values())).mean(), empty))
    bands = [mean.select(v).multiply(SCALE[k_]).rename(k_) for k_, v in BANDS.items()]
    n = ee.Image.constant(coll.size()).toFloat().rename("n_steps")
    tag = f"i{init_date:%Y%m%d}_k{k}"
    return ee.Image.cat(bands + [n]).rename([f"{tag}__{s}" for s in STATS])


def fetch_month(period, last_init):
    inits = pd.date_range(period.start_time, period.end_time.normalize())
    inits = inits[inits <= last_init]
    if len(inits) == 0:
        return inits, None
    shape = (len(LEADS) * len(STATS), config.CAMS_NODES["n_lat"], config.CAMS_NODES["n_lon"])
    path = config.GEE_CACHE_DIR / "cams" / f"cams_{period}.npz"
    complete_month = len(inits) == period.days_in_month
    cached = load_month_cache(path, inits, shape) if complete_month else None
    if cached is not None:
        return cached
    pieces = []
    for i in range(0, len(inits), 10):                       # numpy limits the band list of one answer
        chunk = inits[i:i + 10]
        img = ee.Image.cat([lead_day_image(d, k) for d in chunk for k in LEADS])
        part, _ = fetch_array(img, sources.ee_grid(config.CAMS_NODES))
        pieces.append(part.reshape(len(chunk), len(LEADS) * len(STATS), *part.shape[1:]))
    cube = np.concatenate(pieces)
    if complete_month and (dt.date.today() - period.end_time.date()).days >= config.CACHE_AFTER_DAYS:
        save_month_cache(path, inits, cube)
    return inits, cube


def build_table(first, last_init, verbose=True):
    """All CAMS 12Z runs from `first` to `last_init` -> long DataFrame (see module docstring)."""
    first, last_init = pd.Timestamp(first), pd.Timestamp(last_init)
    months = pd.period_range(first, last_init, freq="M")

    def job(period):
        inits, cube = fetch_month(period, last_init)
        if verbose:
            print(f"  {period}: {len(inits)} runs")
        return inits, cube

    with ThreadPoolExecutor(max_workers=config.GEE_WORKERS) as pool:
        parts = [p for p in pool.map(job, months) if p[1] is not None]
    inits = pd.DatetimeIndex(np.concatenate([p[0] for p in parts]))
    cube = np.concatenate([p[1] for p in parts])
    if inits.has_duplicates or not inits.equals(pd.date_range(inits[0], inits[-1])):
        raise RuntimeError("CAMS months do not join into a continuous daily series")
    inits, cube = inits[inits >= first], cube[inits >= first]

    # (runs, leads*stats, lat, lon) -> (runs, leads, stats, lat, lon) -> centroids
    cube = cube.reshape(len(inits), len(LEADS), len(STATS), *cube.shape[-2:])
    n_idx = STATS.index("n_steps")
    complete = cube[:, :, n_idx] == STEPS_PER_DAY                            # (runs, leads, lat, lon)
    cube[:, :, :n_idx] = np.where(complete[:, :, None], cube[:, :, :n_idx], np.nan)   # partial run -> missing
    at = sources.bilinear(cube, config.CAMS_NODES)                            # (runs, leads, stats, 6)
    ids, names, _, _ = sources.centroids()
    rows = []
    for a, init in enumerate(inits):
        for j, k in enumerate(LEADS):
            for d, (did, name) in enumerate(zip(ids, names)):
                rows.append({"init_date": init, "lead_day": k, "target_date": init + pd.Timedelta(days=1 + k),
                             "district_id": did, "district": name,
                             **{s: at[a, j, si, d] for si, s in enumerate(STATS)}})
    return pd.DataFrame(rows)


def main():
    args = parse_dates(__doc__.splitlines()[1])
    init_ee()
    first = pd.Timestamp(args.start) - pd.Timedelta(days=config.FEATURE_WARMUP_DAYS + 5)
    print(f"CAMS 12Z runs {first.date()} -> {args.end}")
    table = build_table(first, args.end)
    config.MET_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, config.MET_DIR / "cams_daily.csv", float_format="%.6g")
    miss = table["aod550"].isna().mean()
    print(f"Saved {config.MET_DIR / 'cams_daily.csv'} ({len(table)} rows); runs with missing/incomplete steps: {miss:.2%}")
    print(table[list(BANDS)].describe().loc[["min", "mean", "max"]].round(3).T.to_string())


if __name__ == "__main__":
    with pipeline_lock():
        main()
