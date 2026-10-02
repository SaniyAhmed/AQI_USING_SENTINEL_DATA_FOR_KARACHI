"""
PHASE 3 - STEP 2: ERA5 meteorology -> daily values at the six district centroids.

ERA5 (ECMWF reanalysis, hourly, 0.25 deg ~ 28 km) is read on Earth Engine. For every
LOCAL day (Karachi = UTC+5, so UTC [d-1 19:00, d 19:00)) the server computes from the
24 hourly fields:

    t2m_mean/max/min  degC      2 m temperature
    td_mean           degC      2 m dew point
    rh_mean/min       %         relative humidity (Magnus formula from t2m and dew point)
    u10_mean,v10_mean m/s       10 m wind components (daily vector mean -> wind direction)
    ws_mean/max       m/s       10 m wind speed (mean / max of the hourly speeds)
    blh_mean/max/min  m         planetary boundary layer height (PBLH)
    vent_mean/min     m2/s      ventilation coefficient = PBLH x wind speed, computed HOURLY,
                                then averaged (better than mean(PBLH) x mean(speed))
    sp_mean           hPa       surface pressure
    precip_sum        mm        total precipitation
    n_hours           -         number of hourly fields used (must be 24)

The small native lattice (5 x 6 nodes) comes back and is interpolated bilinearly to the
six centroids (sources.bilinear). Months are cached (data/cache/gee/era5/); the newest
months are re-read every run because ERA5T (preliminary) values can be revised.

Output: data/processed/met/era5_daily.csv   (date, district_id, district + the columns above)

Run:  python phase3_features/step2_era5.py [--start 2022-01-01] [--end 2026-09-25]
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

STATS = ["t2m_mean", "td_mean", "rh_mean", "u10_mean", "v10_mean", "ws_mean", "blh_mean", "vent_mean",
         "sp_mean", "t2m_max", "blh_max", "ws_max", "t2m_min", "rh_min", "blh_min", "vent_min",
         "precip_sum", "n_hours"]
HOURLY = "ECMWF/ERA5/HOURLY"
ERA5_CACHE_AFTER_DAYS = 100        # ERA5T -> final ERA5 replacement happens within ~3 months


def hourly_fields(img):
    """One ERA5 hour -> the derived hourly bands (all in the units listed above)."""
    t = img.select("temperature_2m").subtract(273.15).rename("t2m")
    td = img.select("dewpoint_temperature_2m").subtract(273.15).rename("td")

    def magnus(x):
        return x.multiply(17.625).divide(x.add(243.04)).exp()

    rh = magnus(td).divide(magnus(t)).multiply(100).clamp(0, 100).rename("rh")
    u = img.select("u_component_of_wind_10m").rename("u10")
    v = img.select("v_component_of_wind_10m").rename("v10")
    ws = u.hypot(v).rename("ws")
    blh = img.select("boundary_layer_height").rename("blh")
    return (ee.Image.cat([t, td, rh, u, v, ws, blh, blh.multiply(ws).rename("vent"),
                          img.select("surface_pressure").divide(100).rename("sp"),
                          img.select("total_precipitation").multiply(1000).rename("precip"),
                          ee.Image.constant(1).rename("one")])
            .copyProperties(img, ["system:time_start"]))


def day_image(hourly, day):
    """All statistics of one local day as one image with bands named  d<yyyymmdd>__<stat>."""
    start = ee.Date(day.strftime("%Y-%m-%d")).advance(-config.LOCAL_UTC_OFFSET_H, "hour")
    coll = hourly.filterDate(start, start.advance(24, "hour"))
    mean, mx, mn, sm = coll.mean(), coll.max(), coll.min(), coll.sum()
    bands = ([mean.select(b).rename(f"{b}_mean") for b in ["t2m", "td", "rh", "u10", "v10", "ws", "blh", "vent", "sp"]]
             + [mx.select(b).rename(f"{b}_max") for b in ["t2m", "blh", "ws"]]
             + [mn.select(b).rename(f"{b}_min") for b in ["t2m", "rh", "blh", "vent"]]
             + [sm.select("precip").rename("precip_sum"), sm.select("one").rename("n_hours")])
    tag = day.strftime("d%Y%m%d")
    return ee.Image.cat([b.rename(f"{tag}__{n}") for b, n in zip(bands, STATS)])


def last_complete_local_day():
    coll = ee.ImageCollection(HOURLY)
    last = pd.Timestamp(coll.aggregate_max("system:time_start").getInfo(), unit="ms")   # last hour (UTC, hour start)
    return ((last + pd.Timedelta(hours=1)) - pd.Timedelta(hours=24 - config.LOCAL_UTC_OFFSET_H)).normalize()


def fetch_month(period, last_day):
    days = pd.date_range(period.start_time, period.end_time.normalize())
    days = days[days <= last_day]
    if len(days) == 0:
        return days, None
    shape = (len(STATS), config.ERA5_NODES["n_lat"], config.ERA5_NODES["n_lon"])
    path = config.GEE_CACHE_DIR / "era5" / f"era5_{period}.npz"
    complete_month = len(days) == period.days_in_month
    cached = load_month_cache(path, days, shape) if complete_month else None
    if cached is not None:
        return cached
    hourly = ee.ImageCollection(HOURLY).map(hourly_fields).filterDate(
        ee.Date(days[0].strftime("%Y-%m-%d")).advance(-config.LOCAL_UTC_OFFSET_H, "hour"),
        ee.Date(days[-1].strftime("%Y-%m-%d")).advance(24 - config.LOCAL_UTC_OFFSET_H, "hour"))
    # 10 days per request: numpy refuses answers whose band list (header) is too long.
    pieces = []
    for i in range(0, len(days), 10):
        chunk = days[i:i + 10]
        part, names = fetch_array(ee.Image.cat([day_image(hourly, d) for d in chunk]),
                                  sources.ee_grid(config.ERA5_NODES))
        assert [n.split("__")[1] for n in names[:len(STATS)]] == STATS
        pieces.append(part.reshape(len(chunk), len(STATS), *part.shape[1:]))
    cube = np.concatenate(pieces)
    if complete_month and (dt.date.today() - period.end_time.date()).days >= ERA5_CACHE_AFTER_DAYS:
        save_month_cache(path, days, cube)
    return days, cube


def main():
    args = parse_dates(__doc__.splitlines()[1])
    init_ee()
    last_day = min(last_complete_local_day(), pd.Timestamp(args.end))
    first = pd.Timestamp(args.start) - pd.Timedelta(days=config.FEATURE_WARMUP_DAYS + 3)
    months = pd.period_range(first, last_day, freq="M")
    print(f"ERA5: {first.date()} -> {last_day.date()} (last complete local day on Earth Engine), "
          f"{len(months)} months")

    def job(period):
        days, cube = fetch_month(period, last_day)
        bad = 0 if cube is None else int((cube[:, STATS.index("n_hours")].reshape(len(days), -1) != 24).any(axis=1).sum())
        print(f"  {period}: {len(days)} days" + (f", {bad} incomplete" if bad else ""))
        return days, cube

    with ThreadPoolExecutor(max_workers=config.GEE_WORKERS) as pool:
        parts = [p for p in pool.map(job, months) if p[1] is not None]
    dates = pd.DatetimeIndex(np.concatenate([p[0] for p in parts]))
    cube = np.concatenate([p[1] for p in parts])
    if dates.has_duplicates or not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise RuntimeError("ERA5 months do not join into a continuous daily series")
    incomplete = (cube[:, STATS.index("n_hours")] != 24).any(axis=(1, 2))
    cube[incomplete] = np.nan                                               # a day without 24 hours is missing

    at_centroids = sources.bilinear(cube, config.ERA5_NODES)                # (days, stats, 6)
    ids, names, _, _ = sources.centroids()
    rows = []
    for k, (did, name) in enumerate(zip(ids, names)):
        df = pd.DataFrame(at_centroids[:, :, k], columns=STATS)
        df.insert(0, "date", dates); df.insert(1, "district_id", did); df.insert(2, "district", name)
        rows.append(df)
    table = pd.concat(rows).sort_values(["date", "district_id"]).reset_index(drop=True)
    config.MET_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, config.MET_DIR / "era5_daily.csv", float_format="%.6g")
    print(f"Saved {config.MET_DIR / 'era5_daily.csv'}  ({len(table)} rows)")
    print(table.drop(columns=["date", "district_id", "district"]).describe().loc[["min", "mean", "max"]].round(2).T.to_string())


if __name__ == "__main__":
    with pipeline_lock():
        main()
