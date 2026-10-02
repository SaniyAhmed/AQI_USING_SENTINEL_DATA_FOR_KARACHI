"""
PHASE 3 - STEP 5: OPERATIONAL forecast features - NOAA GFS pull for T+0 -> T+1 (automated).

Run every morning (about 10:00 Karachi time, after the 00Z GFS run is out):

    python phase3_features/step5_gfs_forecast.py                    (issue date = today, Karachi time)
    python phase3_features/step5_gfs_forecast.py --issue 2026-09-26

What it does
  1. Pulls the latest NOAA GFS run (Open-Meteo, model gfs_seamless) at the 12 grid nodes around the
     centroids: 5 past days (the model analysis-like early hours) + today + 3 forecast days.
     The RAW hourly answer is archived with the pull time in data/forecast/gfs/ - so, unlike
     the historical data, real forecasts are kept from now on (retrain on them later).
  2. Aggregates local days with the SAME definitions as ERA5 (openmeteo.local_day_stats) and
     interpolates bilinearly to the six centroids.
  3. Refreshes the recent CAMS 12Z runs (Earth Engine) - the run of day D-1 is the newest allowed.
  4. Builds the identical feature rows the training table has (feature_lib.build_features), with
     met_source = "gfs", for the issue date, restricted to config.LIVE_HORIZONS (today + tomorrow
     only): 6 districts x 2 horizons = 12 rows. The historical training table still uses the full
     config.HORIZONS (incl. T+2) - only this live/operational output is narrower.

Output: data/forecast/features_<issue_date>.parquet / .csv   and   data/forecast/gfs_daily_<issue_date>.csv

The satellite table must contain day D-1 (run phase2_ingestion/run_phase2.py first).
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase2_ingestion"))
import pandas as pd

import config
import feature_lib
import openmeteo
import step4_cams
from cache_io import atomic_csv, atomic_write, pipeline_lock
from gee_utils import init_ee

GFS_VARS = openmeteo.VARS_SURFACE + ["temperature_850hPa"]


def local_today():
    return pd.Timestamp((dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=config.LOCAL_UTC_OFFSET_H)).date())


def pull_gfs(issue_date):
    """Latest GFS -> (raw hourly DataFrame, daily-at-centroids DataFrame incl. t850_mean)."""
    hourly = openmeteo.fetch_hourly(config.OPEN_METEO_FORECAST_URL, GFS_VARS, models="gfs_seamless",
                                    past_days=5, forecast_days=5)
    pulled = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
    raw_path = config.FORECAST_DIR / "gfs" / f"gfs_raw_{pulled}.parquet"
    atomic_write(raw_path, lambda p: hourly.assign(pulled_at_utc=pulled).to_parquet(p, index=False))
    days, cube = openmeteo.local_day_stats(hourly)
    table = openmeteo.to_centroid_table(days, cube)
    lo, hi = issue_date - pd.Timedelta(days=3), issue_date + pd.Timedelta(days=max(config.LIVE_HORIZONS))
    table = table[(table["date"] >= lo) & (table["date"] <= hi)].reset_index(drop=True)
    need = pd.date_range(lo, hi)
    missing = need.difference(pd.DatetimeIndex(table["date"].unique()))
    if len(missing):
        raise RuntimeError(f"GFS answer has no complete local day for {list(missing.date)} (pulled {pulled})")
    print(f"GFS pulled {pulled}: {len(hourly) // 12} hours x 12 nodes, {raw_path.name}")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--issue", default=None, help="issue date YYYY-MM-DD (default: today, Karachi time)")
    args = ap.parse_args()
    issue = pd.Timestamp(args.issue) if args.issue else local_today()
    print(f"Issue date {issue.date()} (forecasts for {issue.date()}, +1 day)")

    sat = pd.read_parquet(config.PHASE2_TABLE.with_suffix(".parquet"))
    if sat["date"].max() < issue - pd.Timedelta(days=1):
        raise SystemExit(f"Satellite table ends {sat['date'].max().date()}, but day {(issue - pd.Timedelta(days=1)).date()} "
                         "is needed. Run phase2_ingestion/run_phase2.py first.")
    met = pull_gfs(issue)
    init_ee()
    cams = step4_cams.build_table(issue - pd.Timedelta(days=config.FEATURE_WARMUP_DAYS + 6),
                                  issue - pd.Timedelta(days=1), verbose=False)
    feats = feature_lib.build_features(sat, met, cams, pd.DatetimeIndex([issue]), horizons=config.LIVE_HORIZONS, met_source="gfs")
    config.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{issue:%Y-%m-%d}"
    atomic_csv(met, config.FORECAST_DIR / f"gfs_daily_{tag}.csv", float_format="%.6g")
    atomic_write(config.FORECAST_DIR / f"features_{tag}.parquet", lambda p: feats.to_parquet(p, index=False))
    atomic_csv(feats, config.FORECAST_DIR / f"features_{tag}.csv", float_format="%.6g")
    empty = feats.select_dtypes("number").isna().mean()
    print(f"Saved {config.FORECAST_DIR / f'features_{tag}.parquet'}: {len(feats)} rows x {feats.shape[1]} columns "
          f"(6 districts x {len(config.LIVE_HORIZONS)} horizons)")
    print("empty feature columns:", empty[empty > 0].round(3).to_dict() or "none")


if __name__ == "__main__":
    with pipeline_lock():
        main()
