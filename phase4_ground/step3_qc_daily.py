"""
PHASE 4 - STEP 3: Quality control, local-day means, and district daily series of ground PM.

Hourly values -> (1) hour-level QC -> (2) sensor-level QC -> (3) Karachi-local-day means -> (4) district means.

(1) Hour QC - an hourly value is dropped when it is
      flagged by the provider, negative, above the physical limit (PM2.5 > 1000, PM10 > 2000 ug/m3),
      or part of a flat-line (the identical value for >= 12 hours in a row = stuck sensor).
(2) Sensor QC - low-cost sensors are sometimes indoors or faulty. A sensor is excluded ENTIRELY when, over
      the days it shares with other stations, its daily mean is persistently far from the network median:
      median ratio outside [0.4, 2.5] (needs >= 20 shared days). The excluded sensors are listed in the report.
(3) Local day = Asia/Karachi (UTC+5). A station-day needs >= 18 valid hours (75 %, the EPA completeness
      rule for 24-hour averages); otherwise it is missing, never estimated.
(4) District day = mean over the district's stations that have a valid station-day, plus how many stations
      contributed. The plan's targets are district values, so `n_stations` matters: one sensor is a noisy
      stand-in for a whole district.

Outputs (data/processed/ground/)
    station_daily.csv    date, sensor_id, location_id, parameter, provider, district, value, n_hours
    district_daily.csv   date, district_id, district, pollutant, mean, median, min, max, std, n_stations
    qc_report.json       counts of what each rule removed, excluded sensors, coverage by district and year

Run:  python phase4_ground/step3_qc_daily.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

import config
from cache_io import atomic_csv, atomic_write, pipeline_lock

GROUND_DIR = config.RAW_DIR / "ground"
OUT_DIR = config.PROCESSED_DIR / "ground"
LIMIT = {"pm25": 1000.0, "pm10": 2000.0, "no2": 1000.0, "o3": 600.0, "so2": 3000.0, "co": 100000.0}
FLATLINE_HOURS = 12
MIN_HOURS_PER_DAY = 18
RATIO_OK = (0.4, 2.5)
MIN_SHARED_DAYS = 20


def hour_qc(df, parameter):
    """Returns the hourly frame with a `valid` column and the count removed by each rule."""
    df = df.copy().sort_values("hour_utc").reset_index(drop=True)
    v = df["value"].astype(float)
    flagged = df["has_flags"].fillna(False).astype(bool)
    negative = v < 0
    too_high = v > LIMIT[parameter]
    run_id = (v != v.shift()).cumsum()                                # constant stretches
    run_len = v.groupby(run_id).transform("size")
    contiguous = df["hour_utc"].groupby(run_id).transform(lambda s: (s.max() - s.min()) / pd.Timedelta(hours=1) + 1) == run_len
    flat = (run_len >= FLATLINE_HOURS) & contiguous
    df["valid"] = v.notna() & ~flagged & ~negative & ~too_high & ~flat
    removed = {"flagged": int(flagged.sum()), "negative": int(negative.sum()), "too_high": int(too_high.sum()),
               "flatline": int(flat.sum()), "hours_total": int(len(df))}
    return df, removed


def main():
    sensors = pd.read_csv(GROUND_DIR / "sensors.csv")
    stations = pd.read_csv(GROUND_DIR / "stations.csv")
    meta = sensors.merge(stations[["location_id", "name", "provider", "district_id", "district", "is_reference_grade"]], on="location_id")
    meta = meta[meta["district_id"].notna()]
    report = {"hour_qc": {}, "sensors_excluded": [], "sensor_days_dropped_incomplete": 0}
    daily_parts = []
    for r in meta.itertuples():
        path = GROUND_DIR / "hourly" / f"sensor_{r.sensor_id}.parquet"
        dpath = GROUND_DIR / "hourly" / f"daily_sensor_{r.sensor_id}.parquet"
        if path.exists():
            h = pd.read_parquet(path)
            if h.empty:
                continue
            h, removed = hour_qc(h, r.parameter)
            report["hour_qc"][str(r.sensor_id)] = removed
            h = h[h["valid"]]
            h["date"] = (h["hour_utc"] + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)).dt.normalize()
            g = h.groupby("date")["value"].agg(["mean", "count"]).reset_index()
            source = "hourly"
        elif dpath.exists():
            # daily-only sensor (its hourly endpoint is broken on OpenAQ's side): range and completeness rules only
            dd = pd.read_parquet(dpath)
            ok = dd["value"].notna() & (dd["value"] >= 0) & (dd["value"] <= LIMIT[r.parameter]) & ~dd["has_flags"].fillna(False).astype(bool)
            report["hour_qc"][str(r.sensor_id)] = {"daily_only": True, "days_total": int(len(dd)), "days_rejected": int((~ok).sum())}
            g = dd[ok][["date", "value", "observed_count"]].rename(columns={"value": "mean", "observed_count": "count"})
            source = "daily_endpoint"
        else:
            continue
        report["sensor_days_dropped_incomplete"] += int((g["count"] < MIN_HOURS_PER_DAY).sum())
        g = g[g["count"] >= MIN_HOURS_PER_DAY]
        g["source"] = source
        g["sensor_id"], g["location_id"], g["parameter"] = r.sensor_id, r.location_id, r.parameter
        g["provider"], g["district_id"], g["district"] = r.provider, int(r.district_id), r.district
        g["is_reference_grade"] = bool(r.is_reference_grade)
        daily_parts.append(g.rename(columns={"mean": "value", "count": "n_hours"}))
    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily[daily["date"] >= pd.Timestamp(config.START_DATE) - pd.Timedelta(days=20)]

    # ---- sensor QC against the network median (per pollutant)
    excluded = []
    for par, d in daily.groupby("parameter"):
        med = d.groupby("date")["value"].median()
        cnt = d.groupby("date")["sensor_id"].nunique()
        for sid, s in d.groupby("sensor_id"):
            others = d[(d["date"].isin(s["date"])) & (d["sensor_id"] != sid)].groupby("date")["value"].median()
            j = s.set_index("date")["value"].reindex(others.index).dropna()
            shared = len(j)
            if shared < MIN_SHARED_DAYS:
                continue
            ratio = float((j / others.reindex(j.index).replace(0, np.nan)).median())
            if not (RATIO_OK[0] <= ratio <= RATIO_OK[1]):
                info = meta[meta["sensor_id"] == sid].iloc[0]
                excluded.append({"sensor_id": int(sid), "station": info["name"], "provider": info["provider"], "parameter": par,
                                 "median_ratio_to_network": round(ratio, 2), "shared_days": shared})
    report["sensors_excluded"] = excluded
    drop_ids = {e["sensor_id"] for e in excluded}
    daily = daily[~daily["sensor_id"].isin(drop_ids)].sort_values(["date", "parameter", "sensor_id"]).reset_index(drop=True)
    if daily.duplicated(["date", "sensor_id"]).any():
        raise RuntimeError("duplicate sensor-days")

    # ---- district daily series
    grp = daily.groupby(["date", "district_id", "district", "parameter"])["value"]
    dist = grp.agg(mean="mean", median="median", min="min", max="max", std="std", n_stations="count").reset_index()
    dist = dist.rename(columns={"parameter": "pollutant"}).sort_values(["date", "district_id", "pollutant"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(daily, OUT_DIR / "station_daily.csv", float_format="%.5g")
    atomic_csv(dist, OUT_DIR / "district_daily.csv", float_format="%.5g")

    pm25 = dist[dist["pollutant"] == "pm25"]
    cov = (pm25.assign(year=pm25["date"].dt.year).groupby(["district", "year"]).agg(days=("date", "nunique"), median_stations=("n_stations", "median")))
    report["pm25_days_by_district_year"] = {f"{k[0]}|{k[1]}": {"days": int(v.days), "median_stations": float(v.median_stations)} for k, v in cov.iterrows()}
    atomic_write(OUT_DIR / "qc_report.json", lambda p: Path(p).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8"))

    print(f"{len(daily)} valid station-days from {daily['sensor_id'].nunique()} sensors; {len(excluded)} sensors excluded as unrepresentative")
    for e in excluded:
        print("   excluded:", e)
    print("\nPM2.5 district-days with data (columns = year); median stations behind each district-day in brackets")
    tab = cov.reset_index().pivot(index="district", columns="year", values="days").fillna(0).astype(int)
    st = cov.reset_index().pivot(index="district", columns="year", values="median_stations")
    print(tab.astype(str).add(" (").add(st.round(0).fillna(0).astype(int).astype(str)).add(")").to_string())
    hourly_qc = [v for v in report["hour_qc"].values() if "hours_total" in v]
    print(f"\nhours removed by the hour rules ({len(hourly_qc)} hourly sensors):",
          {k: sum(v[k] for v in hourly_qc) for k in ["flagged", "negative", "too_high", "flatline", "hours_total"]})
    print(f"{len(report['hour_qc']) - len(hourly_qc)} sensors are daily-only (their hourly endpoint is broken on OpenAQ's side)")


if __name__ == "__main__":
    with pipeline_lock():
        main()
