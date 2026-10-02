"""
PHASE 4 - STEP 4: Ground-truth targets and the unified dataset.

Targets (for the TARGET day D+h of each feature row; one value per district and day)
    pm25_target, pm10_target     ug/m3   mean of the district's stations (step 3)
    n_stations_target            how many stations stand behind the value (use as a sample weight / filter)
    aqi_pm25                     EPA sub-index of the PM2.5 daily mean          <- primary target
    aqi_pm10                     EPA sub-index of the PM10 daily mean (few stations measure PM10)
    aqi_target                   max of the available sub-indices (the plan's AQI = max(I_PM2.5, I_PM10, I_NO2, I_O3))
    aqi_pollutants_used          which sub-indices were available that day, e.g. "pm25" or "pm25+pm10"
NO2 and O3 have NO ground measurements in Karachi (every one of the 56 stations reports PM only), so their
sub-indices cannot be built from real data and are not invented; aqi.py has the tables for when data exists.

Ground-history features (anchored to the ISSUE day: only days <= D-1, exactly like the satellite lags)
    g_<pollutant>_lag1..3, _roll3/_roll7_mean/_std   of the district's own daily mean   (pm25, pm10, aqi_pm25)
    g_pm25_city_lag1                                 mean over ALL stations of Karachi on day D-1
    Rolling windows accept gaps (>= half the days); a lag is empty when that day has no valid value.

Output: data/processed/features/phase4_dataset.parquet / .csv  - every Phase 3 feature row plus the targets;
        rows without a target keep `target_available` = False (Phase 5 trains on the True rows).

Run:  python phase4_ground/step4_dataset.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase3_features"))
import numpy as np
import pandas as pd

import aqi
import config
import feature_lib
from cache_io import atomic_csv, atomic_write, pipeline_lock

GROUND = config.PROCESSED_DIR / "ground"
DATASET = config.FEATURE_DIR / "phase4_dataset"


def district_ground(d=None):
    """Wide tables (date x district) of the district daily means, plus AQI sub-indices, on a continuous daily index.
    `d` = the district_daily table (read from disk by default; the validator passes a modified copy)."""
    if d is None:
        d = pd.read_csv(GROUND / "district_daily.csv", parse_dates=["date"])
    days = pd.date_range(pd.Timestamp(config.START_DATE) - pd.Timedelta(days=20), d["date"].max())
    ids = sorted(d["district_id"].unique())
    wide = {}
    for par in ("pm25", "pm10"):
        p = d[d["pollutant"] == par]
        wide[par] = p.pivot(index="date", columns="district_id", values="mean").reindex(index=days, columns=ids)
        wide[f"n_{par}"] = p.pivot(index="date", columns="district_id", values="n_stations").reindex(index=days, columns=ids)
    wide["aqi_pm25"] = pd.DataFrame(aqi.sub_index("pm25", wide["pm25"].to_numpy()), index=days, columns=ids)
    wide["aqi_pm10"] = pd.DataFrame(aqi.sub_index("pm10", wide["pm10"].to_numpy()), index=days, columns=ids)
    return wide, days, ids


def city_mean(days, s=None):
    if s is None:
        s = pd.read_csv(GROUND / "station_daily.csv", parse_dates=["date"])
    s = s[s["parameter"] == "pm25"]
    return s.groupby("date")["value"].mean().reindex(days)


def targets_table(wide):
    """Long table (target_date, district_id) with every target column."""
    frames = {"pm25_target": wide["pm25"], "pm10_target": wide["pm10"], "n_stations_target": wide["n_pm25"],
              "aqi_pm25": wide["aqi_pm25"], "aqi_pm10": wide["aqi_pm10"]}
    t = feature_lib._stack(frames, lambda k: k).reset_index().rename(columns={"date": "target_date"})
    both = t[["aqi_pm25", "aqi_pm10"]]
    t["aqi_target"] = both.max(axis=1, skipna=True)
    used = np.where(t["aqi_pm25"].notna() & t["aqi_pm10"].notna(), "pm25+pm10",
                    np.where(t["aqi_pm25"].notna(), "pm25", np.where(t["aqi_pm10"].notna(), "pm10", "")))
    t["aqi_pollutants_used"] = used
    dom = np.where(t["aqi_pm10"].fillna(-1) > t["aqi_pm25"].fillna(-1), "pm10", "pm25")
    t["aqi_dominant"] = np.where(t["aqi_target"].isna(), "", dom)
    return t


def ground_history_features(wide, days, ids, station_daily=None):
    """Lags / rolling statistics anchored to the issue day D (days <= D-1 only)."""
    frames = {}
    for name, w in (("pm25", wide["pm25"]), ("pm10", wide["pm10"]), ("aqi_pm25", wide["aqi_pm25"])):
        for kind, df in feature_lib.causal_lags_rolls(w, min_frac=0.5).items():
            frames[(name, kind)] = df
    out = feature_lib._stack(frames, lambda k: f"g_{k[0]}_{k[1]}")
    city = city_mean(days, station_daily)
    city_lag1 = pd.DataFrame({i: city.shift(1) for i in ids})
    out["g_pm25_city_lag1"] = city_lag1.stack(future_stack=True).reindex(out.index)
    return out


def main():
    feats = pd.read_parquet(config.FEATURE_TABLE.with_suffix(".parquet"))
    wide, days, ids = district_ground()
    hist = ground_history_features(wide, days, ids).reset_index().rename(columns={"date": "issue_date"})
    tgt = targets_table(wide)
    ds = feats.merge(hist, on=["issue_date", "district_id"], how="left", validate="many_to_one")
    ds = ds.merge(tgt, on=["target_date", "district_id"], how="left", validate="many_to_one")
    ds["target_available"] = ds["pm25_target"].notna()
    ds = ds.sort_values(["issue_date", "district_id", "horizon"]).reset_index(drop=True)
    if ds.duplicated(["issue_date", "district_id", "horizon"]).any() or len(ds) != len(feats):
        raise RuntimeError("the join changed the number of rows")
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(DATASET.with_suffix(".parquet"), lambda p: ds.to_parquet(p, index=False))
    atomic_csv(ds, DATASET.with_suffix(".csv"), float_format="%.6g")

    ok = ds[ds["target_available"]]
    print(f"Dataset: {len(ds)} rows x {ds.shape[1]} columns; rows with a PM2.5 target: {len(ok)} ({len(ok) / len(ds):.1%})")
    print("\nRows with a target, by district and year of the target day (median stations behind the value):")
    t = ok.assign(year=ok["target_date"].dt.year).groupby(["district", "year"]).agg(rows=("pm25_target", "size"), stations=("n_stations_target", "median"))
    print(t.reset_index().pivot(index="district", columns="year", values="rows").fillna(0).astype(int).to_string())
    print("\nAQI target category share (rows with a target):")
    print(aqi.category(ok["aqi_target"]).value_counts(normalize=True).round(3).to_string())
    print("\nPM2.5 target:", ok["pm25_target"].describe().round(1).to_dict())
    print(f"\nSaved {DATASET.with_suffix('.parquet')}")


if __name__ == "__main__":
    with pipeline_lock():
        main()
