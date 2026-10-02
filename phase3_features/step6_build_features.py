"""
PHASE 3 - STEP 6: Build the training feature table.

Inputs
    data/processed/phase2_daily_features.parquet    satellite series (Phase 2)
    data/processed/met/era5_daily.csv               ERA5 meteorology at the centroids (step 2)
    data/processed/met/gfs_t850_daily.csv           850 hPa temperature (step 3)
    data/processed/met/cams_daily.csv               CAMS 12Z-run aerosol (step 4)

Output
    data/processed/features/phase3_features.parquet / .csv
    one row per (issue_date, district, horizon 0/1/2); see feature_lib.py for every column.

Rows are kept only when the target-day meteorology exists (ERA5 ends about a week before today);
the newest days are produced by the operational path (step5_gfs_forecast.py) instead.

Run:  python phase3_features/step6_build_features.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import pandas as pd

import config
import feature_lib
from cache_io import atomic_csv, atomic_write, pipeline_lock


def load_inputs():
    sat = pd.read_parquet(config.PHASE2_TABLE.with_suffix(".parquet"))
    met = pd.read_csv(config.MET_DIR / "era5_daily.csv", parse_dates=["date"])
    t850 = pd.read_csv(config.MET_DIR / "gfs_t850_daily.csv", parse_dates=["date"])[["date", "district_id", "t850_mean"]]
    met = met.merge(t850, on=["date", "district_id"], how="left", validate="one_to_one")
    cams = pd.read_csv(config.MET_DIR / "cams_daily.csv", parse_dates=["init_date", "target_date"])
    return sat, met, cams


def main():
    sat, met, cams = load_inputs()
    first = pd.Timestamp(config.START_DATE) + pd.Timedelta(days=config.FEATURE_WARMUP_DAYS)
    last = min(sat["date"].max() + pd.Timedelta(days=1), met["date"].max())      # D-1 must be in the satellite table
    issue_dates = pd.date_range(first, last)
    feats = feature_lib.build_features(sat, met, cams, issue_dates, met_source="era5")
    n_all = len(feats)
    feats = feats[feats["met_available"]].reset_index(drop=True)
    print(f"{len(feats)} rows kept of {n_all} (target-day meteorology missing for the rest, "
          f"i.e. issue dates after {met['date'].max().date()} minus the horizon)")
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(config.FEATURE_TABLE.with_suffix(".parquet"), lambda p: feats.to_parquet(p, index=False))
    atomic_csv(feats, config.FEATURE_TABLE.with_suffix(".csv"), float_format="%.6g")
    num = feats.select_dtypes("number")
    print(f"Saved {config.FEATURE_TABLE.with_suffix('.parquet')}: {feats.shape[0]} rows x {feats.shape[1]} columns, "
          f"issue dates {feats['issue_date'].min().date()} -> {feats['issue_date'].max().date()}")
    empty = num.isna().mean().sort_values(ascending=False)
    print("Columns with the most empty values:\n" + (empty[empty > 0].head(12) * 100).round(2).astype(str).add(" %").to_string())


if __name__ == "__main__":
    with pipeline_lock():
        main()
