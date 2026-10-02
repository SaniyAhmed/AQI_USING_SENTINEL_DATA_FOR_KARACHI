"""
PHASE 3 - STEP 3: 850 hPa temperature history (for the temperature-inversion proxy).

The plan defines the inversion proxy as  T(2 m) - T(850 hPa).  Earth Engine's ERA5 has no
pressure-level data, so the 850 hPa temperature comes from the ARCHIVED NOAA GFS runs
(Open-Meteo "historical forecast" API: the first hours of every GFS run = analysis-like).
This is also exactly what the operational forecast uses (GFS 850 hPa), so training and
operation see the same quantity. (MERRA-2 T850 on Earth Engine was checked as an independent
cross-check: mean difference 0.1 K in January; in July the monsoon 850 hPa temperature barely
varies, which lowers the correlation.)

Local-day mean of the hourly 850 hPa temperature at 12 GFS nodes, interpolated bilinearly to the
six centroids.  Completed calendar years are cached; the current year is re-read every run.

Output: data/processed/met/gfs_t850_daily.csv   (date, district_id, district, t850_mean)

Run:  python phase3_features/step3_gfs_t850.py
"""
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase2_ingestion"))
import numpy as np
import pandas as pd

import config
import openmeteo
import sources
from cache_io import atomic_csv, atomic_write, pipeline_lock
from cli import parse_dates

CACHE = config.DATA_DIR / "cache" / "openmeteo"


def hourly_t850(year, end):
    """Hourly T850 at the 12 nodes for one calendar year (+1 day each side for local-day edges)."""
    start_d = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=1)
    end_d = min(pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=1), end)
    path = CACHE / f"t850_{year}.parquet"
    complete = end_d >= pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=1) and year < dt.date.today().year
    if complete and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink()                                   # unreadable -> download again
    df = openmeteo.fetch_hourly(config.OPEN_METEO_ARCHIVE_URL, ["temperature_850hPa"],
                                start_date=start_d.strftime("%Y-%m-%d"), end_date=end_d.strftime("%Y-%m-%d"),
                                models="gfs_seamless")
    if complete:
        atomic_write(path, lambda p: df.to_parquet(p, index=False))
    return df


def main():
    args = parse_dates(__doc__.splitlines()[1])
    start = pd.Timestamp(args.start) - pd.Timedelta(days=config.FEATURE_WARMUP_DAYS + 3)
    end = pd.Timestamp(args.end)
    frames = []
    for year in range(start.year, end.year + 1):
        df = hourly_t850(year, end)
        print(f"  {year}: {len(df) // 12} hours x 12 nodes, missing {df['temperature_850hPa'].isna().mean():.2%}")
        frames.append(df)
    h = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "lat", "lon"])
    h["day"] = (h["time"] + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)).dt.normalize()
    g = h.groupby(["day", "lat", "lon"])["temperature_850hPa"]
    daily = pd.DataFrame({"t850_mean": g.mean(), "n": g.count()}).reset_index()
    daily = daily[daily["n"] == 24]                          # complete local days only
    days = pd.date_range(start, end)
    cube = np.full((len(days), openmeteo.NODES["n_lat"], openmeteo.NODES["n_lon"]), np.nan)
    di = days.get_indexer(daily["day"])
    keep = di >= 0
    li = np.searchsorted(-openmeteo.NODE_LATS, -daily["lat"].to_numpy())
    lj = np.searchsorted(openmeteo.NODE_LONS, daily["lon"].to_numpy())
    cube[di[keep], li[keep], lj[keep]] = daily["t850_mean"].to_numpy()[keep]
    table = openmeteo.to_centroid_table(days, {"t850_mean": cube})
    table = table.dropna(subset=["t850_mean"])
    config.MET_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, config.MET_DIR / "gfs_t850_daily.csv", float_format="%.5g")
    print(f"Saved {config.MET_DIR / 'gfs_t850_daily.csv'} ({len(table)} rows, "
          f"{table['date'].min().date()} -> {table['date'].max().date()}); "
          f"T850 min/mean/max {table['t850_mean'].min():.1f}/{table['t850_mean'].mean():.1f}/{table['t850_mean'].max():.1f} degC")


if __name__ == "__main__":
    with pipeline_lock():
        main()
