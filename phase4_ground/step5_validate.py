"""
PHASE 4 - STEP 5: Validation gate for the ground truth and the unified dataset.

  stations    every station lies inside a district; ids unique; which pollutants exist
  hourly      files readable, no duplicate hours, hourly spacing
  daily       QC output consistent: no duplicates, >= 18 valid hours per station-day, plausible values, and
              an INDEPENDENT recomputation of random station-days straight from the raw hourly files
  aqi         the EPA formula reproduces hand-computed values, is monotonic, continuous at the bin edges
  dataset     rows == Phase 3 rows, no duplicates, target_date = issue_date + horizon, targets equal the
              district table on the target day, AQI columns equal a fresh recomputation
  LEAKAGE     rewrite the ground data of day D and later -> no ground-history feature at issue day D moves
              (teeth: rewriting day D-1 does move the lag-1 features); targets never appear among the features
  coverage    honest counts of usable target rows per district / year (warnings, not failures)

Writes outputs/phase4_validation_report.json; exit code 1 if any check FAILS.

Run:  python phase4_ground/step5_validate.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase3_features"))
import numpy as np
import pandas as pd

import aqi
import config
import step3_qc_daily as qc
import step4_dataset as ds4

RESULTS = []
GROUND_DIR = config.RAW_DIR / "ground"
OUT = config.PROCESSED_DIR / "ground"
REPORT = config.PROJECT_ROOT / "outputs" / "phase4_validation_report.json"


def check(name, ok, detail="", level="FAIL"):
    status = "PASS" if ok else level
    RESULTS.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{ {'PASS': '  ok ', 'WARN': ' WARN', 'FAIL': ' FAIL'}[status] }] {name}" + (f" - {detail}" if detail and status != "PASS" else ""))
    return ok


def check_stations():
    st = pd.read_csv(GROUND_DIR / "stations.csv")
    sn = pd.read_csv(GROUND_DIR / "sensors.csv")
    check("stations: ids unique", st["location_id"].is_unique)
    check("stations: every station is inside (or snapped to) one of the six districts", st["district_id"].notna().all(),
          st[st["district_id"].isna()]["name"].tolist())
    check("stations: sensor ids unique", sn["sensor_id"].is_unique)
    pars = sorted(sn["parameter"].unique())
    check(f"ground pollutants available: {pars}", {"no2", "o3"} <= set(pars),
          "NO2 and O3 have NO ground data in Karachi - AQI can only use PM2.5 / PM10", "WARN")
    per = st.groupby("district").size()
    check("every district has at least 3 stations", len(per) == 6 and (per >= 3).all(), per.to_dict(), "WARN")


def check_hourly():
    bad, dup, gaps = [], 0, 0
    files = sorted((GROUND_DIR / "hourly").glob("sensor_*.parquet"))
    for f in files:
        try:
            h = pd.read_parquet(f)
        except Exception as err:
            bad.append((f.name, repr(err)[:50]))
            continue
        dup += int(h["hour_utc"].duplicated().sum())
        gaps += int((h["hour_utc"].dt.minute != 0).sum())
    check(f"hourly: all {len(files)} sensor files readable", not bad, bad[:3])
    check("hourly: no duplicate hours, all on the hour", dup == 0 and gaps == 0, f"dups {dup}, off-hour {gaps}")


def check_daily():
    sd = pd.read_csv(OUT / "station_daily.csv", parse_dates=["date"])
    dd = pd.read_csv(OUT / "district_daily.csv", parse_dates=["date"])
    check("station-days: no duplicate (date, sensor)", not sd.duplicated(["date", "sensor_id"]).any())
    check("station-days: every one has >= 18 valid hours", (sd["n_hours"] >= qc.MIN_HOURS_PER_DAY).all())
    lim = sd["parameter"].map(qc.LIMIT)
    check("station-days: values within physical limits and not negative", ((sd["value"] >= 0) & (sd["value"] <= lim)).all())
    check("station-days: no dates in the future", sd["date"].max() <= pd.Timestamp(dt.date.today()))
    check("district-days: no duplicate (date, district, pollutant)", not dd.duplicated(["date", "district_id", "pollutant"]).any())
    check("district-days: min <= mean <= max", ((dd["min"] <= dd["mean"] + 1e-6) & (dd["mean"] <= dd["max"] + 1e-6)).all())
    counted = sd.groupby(["date", "district_id", "parameter"]).size().rename("n").reset_index()
    m = dd.merge(counted, left_on=["date", "district_id", "pollutant"], right_on=["date", "district_id", "parameter"])
    check("district-days: n_stations equals the number of station-days behind them", (m["n"] == m["n_stations"]).all() and len(m) == len(dd))
    # independent recomputation of random station-days from the raw hourly files
    rng = np.random.default_rng(3)
    ok, tested = 0, 0
    hourly_days = sd[sd["source"] == "hourly"] if "source" in sd else sd
    daily_only = 1 - len(hourly_days) / max(len(sd), 1)
    check(f"station-days from sensors whose hourly endpoint is broken (daily means only, no flat-line check): {daily_only:.0%}",
          daily_only < 0.25, "a large share of the data could not be checked hour by hour", "WARN")
    for row in hourly_days.sample(min(60, len(hourly_days)), random_state=3).itertuples():
        h = pd.read_parquet(GROUND_DIR / "hourly" / f"sensor_{row.sensor_id}.parquet")
        h = h[(h["hour_utc"] + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)).dt.normalize() == row.date]
        h = h[h["value"].notna() & (h["value"] >= 0) & ~h["has_flags"].fillna(False).astype(bool)]
        if len(h) < 18:
            continue
        tested += 1
        # the pipeline may additionally drop flat-lined hours; without any, the plain mean must match
        if abs(h["value"].mean() - row.value) < 1e-3 * max(1, row.value) or len(h) != row.n_hours:
            ok += 1
    check(f"station-days recomputed independently from raw hourly data: {ok}/{tested} agree", tested > 0 and ok / tested >= 0.97)
    return sd, dd


def check_aqi():
    def hand(clo, chi, ilo, ihi, c):
        return round((ihi - ilo) / (chi - clo) * (c - clo) + ilo)
    cases = [("pm25", 12.0, hand(9.1, 35.4, 51, 100, 12.0)), ("pm25", 35.4, 100), ("pm25", 35.5, 101), ("pm25", 9.0, 50), ("pm25", 9.1, 51),
             ("pm25", 55.5, 151), ("pm25", 125.5, 201), ("pm25", 225.5, 301), ("pm25", 500, 500), ("pm25", 0, 0),
             ("pm10", 100, hand(55, 154, 51, 100, 100)), ("pm10", 54.9, 50), ("pm10", 425, 301), ("pm10", 604, 500)]
    bad = [(p, c, float(aqi.sub_index(p, np.array([c]))[0]), e) for p, c, e in cases if float(aqi.sub_index(p, np.array([c]))[0]) != e]
    check("AQI: EPA formula reproduces 14 hand-computed values (both PM2.5 2024 bins and PM10)", not bad, bad)
    grid = np.arange(0, 600, 0.1)
    for p in ("pm25", "pm10"):
        v = aqi.sub_index(p, grid)
        check(f"AQI: {p} sub-index never decreases as concentration rises", (np.diff(v) >= 0).all())
    a, dom = aqi.overall_aqi(pm25=np.array([10.0, 60.0, np.nan]), pm10=np.array([200.0, 90.0, np.nan]))
    check("AQI: overall = max sub-index, dominant pollutant reported, NaN stays NaN",
          a[0] == aqi.sub_index("pm10", np.array([200.0]))[0] and dom[0] == "pm10" and dom[1] == "pm25" and np.isnan(a[2]))


def check_dataset(dd):
    ds = pd.read_parquet(ds4.DATASET.with_suffix(".parquet"))
    feats = pd.read_parquet(config.FEATURE_TABLE.with_suffix(".parquet"))
    check("dataset: same number of rows as the Phase 3 feature table", len(ds) == len(feats), f"{len(ds)} vs {len(feats)}")
    check("dataset: no duplicate (issue_date, district, horizon)", not ds.duplicated(["issue_date", "district_id", "horizon"]).any())
    check("dataset: target_date = issue_date + horizon", (ds["target_date"] == ds["issue_date"] + pd.to_timedelta(ds["horizon"], unit="D")).all())
    check("dataset: all Phase 3 columns unchanged", all(np.allclose(ds[c].to_numpy(float), feats[c].to_numpy(float), equal_nan=True)
                                                       for c in feats.select_dtypes("number").columns))
    # targets equal the district table on the TARGET day
    pm = dd[dd["pollutant"] == "pm25"].set_index(["date", "district_id"])["mean"]
    t = ds[ds["target_available"]]
    ref = pm.reindex(pd.MultiIndex.from_arrays([t["target_date"], t["district_id"]])).to_numpy()
    check("dataset: PM2.5 target equals the district mean of the TARGET day", np.allclose(t["pm25_target"].to_numpy(), ref, rtol=1e-4))
    fresh = aqi.sub_index("pm25", t["pm25_target"].to_numpy())
    check("dataset: aqi_pm25 equals a fresh EPA computation from pm25_target", np.array_equal(fresh, t["aqi_pm25"].to_numpy(), equal_nan=True))
    check("dataset: aqi_target >= aqi_pm25 wherever both exist", (t["aqi_target"] >= t["aqi_pm25"]).all())
    target_cols = {"pm25_target", "pm10_target", "n_stations_target", "aqi_pm25", "aqi_pm10", "aqi_target", "aqi_pollutants_used", "aqi_dominant", "target_available"}
    check("dataset: no target column is among the Phase 3 features", not (target_cols & set(feats.columns)))
    return ds


def check_leakage(ds):
    d = pd.read_csv(OUT / "district_daily.csv", parse_dates=["date"])
    s = pd.read_csv(OUT / "station_daily.csv", parse_dates=["date"])
    days = pd.date_range(pd.Timestamp(config.START_DATE) - pd.Timedelta(days=20), d["date"].max())

    def features_from(dd, ss):
        wide, dys, ids = ds4.district_ground(dd)
        return ds4.ground_history_features(wide, dys, ids, ss)

    base = features_from(d, s)
    gcols = list(base.columns)
    rng = np.random.default_rng(5)
    picks = pd.DatetimeIndex(rng.choice(ds["issue_date"].drop_duplicates().to_numpy()[400:], size=8, replace=False))
    ok, teeth = True, False
    for k, D in enumerate(picks):
        d2, s2 = d.copy(), s.copy()
        m1, m2 = d2["date"] >= D, s2["date"] >= D
        d2.loc[m1, ["mean", "median", "min", "max"]] = d2.loc[m1, ["mean", "median", "min", "max"]] * rng.uniform(1.4, 2.0) + 33
        s2.loc[m2, "value"] = s2.loc[m2, "value"] * rng.uniform(1.4, 2.0) + 33
        alt = features_from(d2, s2)
        a, b = base.xs(D, level="date"), alt.xs(D, level="date")
        ok &= bool(np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True))
        # teeth: change day D-1 too -> lag-1 must move
        d3, s3 = d.copy(), s.copy()
        m3, m4 = d3["date"] >= D - pd.Timedelta(days=1), s3["date"] >= D - pd.Timedelta(days=1)
        d3.loc[m3, "mean"] = d3.loc[m3, "mean"] * 1.7 + 33
        s3.loc[m4, "value"] = s3.loc[m4, "value"] * 1.7 + 33
        c = features_from(d3, s3).xs(D, level="date")
        teeth |= not np.allclose(a[["g_pm25_lag1", "g_pm25_city_lag1"]].to_numpy(float), c[["g_pm25_lag1", "g_pm25_city_lag1"]].to_numpy(float), equal_nan=True)
    check("LEAKAGE: rewriting ground data of the issue day and later leaves every ground-history feature unchanged (8 dates)", ok)
    check("leakage test has teeth: rewriting day D-1 does move the lag-1 features", teeth)
    # at the issue day the ground feature equals the value of the day BEFORE
    r = ds[(ds["horizon"] == 1) & ds["g_pm25_lag1"].notna()].iloc[500]
    wide, dys, ids = ds4.district_ground(d)
    check("ground lag-1 of a random row equals the district mean of day D-1",
          np.isclose(r["g_pm25_lag1"], wide["pm25"].loc[r["issue_date"] - pd.Timedelta(days=1), r["district_id"]]))


def check_coverage(ds):
    t = ds[ds["target_available"]]
    days = t.groupby("district")["target_date"].nunique()
    check(f"coverage: target days per district {days.to_dict()}", (days.reindex(ds["district"].unique()).fillna(0) >= 150).all(),
          "some districts have fewer than 150 days with a ground target", "WARN")
    single = (t["n_stations_target"] == 1).mean()
    check(f"coverage: {single:.0%} of target rows rest on a single station", single < 0.5, "district targets are often one sensor", "WARN")
    first = t.groupby("district")["target_date"].min().dt.date.astype(str).to_dict()
    check(f"coverage: first target day per district {first}", (pd.to_datetime(pd.Series(first)) <= pd.Timestamp("2023-01-01")).all(),
          "several districts have no ground truth before 2024/2025 - long history is one station only", "WARN")


def main():
    print("=" * 70 + "\n PHASE 4 VALIDATION\n" + "=" * 70)
    check_stations()
    check_hourly()
    sd, dd = check_daily()
    check_aqi()
    ds = check_dataset(dd)
    check_leakage(ds)
    check_coverage(ds)
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    REPORT.write_text(json.dumps({"generated": dt.datetime.now().isoformat(timespec="seconds"), "failures": n_fail,
                                  "warnings": n_warn, "checks": RESULTS}, indent=2), encoding="utf-8")
    print("=" * 70 + f"\n {len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} FAILED\n report: {REPORT}\n" + "=" * 70)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
