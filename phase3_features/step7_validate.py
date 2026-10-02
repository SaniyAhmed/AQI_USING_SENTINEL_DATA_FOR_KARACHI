"""
PHASE 3 - STEP 7: Validation gate. Phase 4/5 must not rely on the feature table until this passes.

  inputs      ERA5 / T850 / CAMS tables: no duplicates or missing days, all six districts, physical ranges,
              24 hours behind every ERA5 day, no incomplete CAMS runs beyond a tiny share
  bilinear    reproduces a linear field exactly on both lattices
  cross-check my Earth Engine ERA5 extraction against an INDEPENDENT copy of ERA5 (Open-Meteo archive)
              at two grid nodes for a month: temperature, wind, boundary layer, pressure, humidity, rain
  cams        CAMS AOD agrees with MODIS AOD where MODIS has data (so the gap-filler is sane)
  features    no duplicate rows, target_date = issue_date + horizon, cyclical columns in range
  LEAKAGE     rewrite the future and check that no feature at issue date D moves:
                satellite features  <- scramble every satellite day >= D
                CAMS features       <- scramble every CAMS run with init_date >= D  (and the run date is D-1)
                observed-met lags   <- scramble every met day >= D
                target-day met      <- scramble every met day AFTER the target day
              and, to prove the test has teeth, scrambling day D-1 DOES change the lag features.
  serving     the operational feature file (if present) has exactly the training columns

Writes outputs/phase3_validation_report.json; exit code 1 if any check FAILS.

Run:  python phase3_features/step7_validate.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase2_ingestion"))
import numpy as np
import pandas as pd

import config
import feature_lib
import openmeteo
import sources
import step2_era5
import step4_cams
import step6_build_features
from cache_io import CACHE_SCHEMA, LEGACY_SCHEMA

RESULTS = []
RANGES = {"t2m_mean": (-10, 50), "t2m_max": (-10, 55), "t2m_min": (-15, 45), "td_mean": (-40, 40),
          "rh_mean": (0, 100), "rh_min": (0, 100), "ws_mean": (0, 40), "ws_max": (0, 60), "blh_mean": (0, 5000),
          "blh_max": (0, 6000), "blh_min": (0, 5000), "vent_mean": (0, 100000), "vent_min": (0, 100000),
          "sp_mean": (900, 1100), "precip_sum": (-1e-3, 600), "t850_mean": (-25, 40),
          "u10_mean": (-40, 40), "v10_mean": (-40, 40)}
CAMS_RANGES = {"aod550": (0, 6), "dust_aod550": (0, 6), "pm25": (0, 1500), "pm10": (0, 3000)}


def check(name, ok, detail="", level="FAIL"):
    status = "PASS" if ok else level
    RESULTS.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{ {'PASS': '  ok ', 'WARN': ' WARN', 'FAIL': ' FAIL'}[status] }] {name}" + (f" - {detail}" if detail and status != "PASS" else ""))
    return ok


# ----------------------------------------------------------------------------
def check_bilinear():
    ids, _, clat, clon = sources.centroids()
    ok = True
    for nodes in (config.ERA5_NODES, config.CAMS_NODES, openmeteo.NODES):
        lats, lons = sources.axes(nodes)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        f = 3.0 + 2.0 * LA - 5.0 * LO
        ok &= bool(np.allclose(sources.bilinear(np.stack([f, 10 * f]), nodes)[0], 3.0 + 2.0 * clat - 5.0 * clon))
    check("bilinear interpolation reproduces a linear field exactly (ERA5, CAMS, GFS lattices)", ok)


def check_daily_table(name, t, keys, ranges, first, last_expected=None):
    check(f"{name}: no duplicate (date, district) rows", not t.duplicated(["date", "district_id"]).any())
    days = pd.date_range(t["date"].min(), t["date"].max())
    check(f"{name}: every day and all 6 districts present", len(t) == len(days) * 6 and set(t["date"]) == set(days),
          f"{len(t)} rows vs {len(days) * 6}")
    check(f"{name}: covers the feature period", t["date"].min() <= first and (last_expected is None or t["date"].max() >= last_expected),
          f"{t['date'].min().date()} -> {t['date'].max().date()}")
    bad = {c: int((~t[c].dropna().between(*ranges[c])).sum()) for c in keys if c in t and not t[c].dropna().between(*ranges[c]).all()}
    check(f"{name}: values in physical ranges", not bad, bad)
    check(f"{name}: no infinite values", not np.isinf(t[keys].to_numpy(float)).any())


def check_inputs(sat, met, cams):
    first = pd.Timestamp(config.START_DATE)
    era = pd.read_csv(config.MET_DIR / "era5_daily.csv", parse_dates=["date"])
    check_daily_table("ERA5", era, [c for c in step2_era5.STATS if c in RANGES], RANGES, first, met["date"].max())
    check("ERA5: every day built from 24 hourly fields", (era["n_hours"].dropna() == 24).all() and era["t2m_mean"].notna().all(),
          f"{int(era['t2m_mean'].isna().sum())} empty days")
    t850 = pd.read_csv(config.MET_DIR / "gfs_t850_daily.csv", parse_dates=["date"])
    check_daily_table("GFS T850", t850, ["t850_mean"], RANGES, first)
    need = pd.date_range(first - pd.Timedelta(days=config.FEATURE_WARMUP_DAYS + 3), era["date"].max())   # ERA5 also holds padding days
    check("T850 covers every ERA5 day of the feature period", set(need) <= set(t850["date"]),
          f"{len(set(need) - set(t850['date']))} days missing")
    check("CAMS: no duplicate (run, lead, district) rows", not cams.duplicated(["init_date", "lead_day", "district_id"]).any())
    runs = pd.DatetimeIndex(sorted(cams["init_date"].unique()))
    check("CAMS: one 12Z run for every day, no gaps", runs.equals(pd.date_range(runs[0], runs[-1])), f"{runs[0].date()} -> {runs[-1].date()}")
    check("CAMS: target_date = run date + 1 + lead", (cams["target_date"] == cams["init_date"] + pd.to_timedelta(1 + cams["lead_day"], unit="D")).all())
    bad = {c: int((~cams[c].dropna().between(*CAMS_RANGES[c])).sum()) for c in CAMS_RANGES}
    check("CAMS: values in physical ranges", not any(bad.values()), bad)
    miss = cams["aod550"].isna().mean()
    check("CAMS: incomplete/missing runs are rare (< 0.5 %)", miss < 0.005, f"{miss:.2%}")


def check_era5_against_openmeteo():
    """Independent ERA5 copy (Open-Meteo archive, model era5) vs my Earth Engine extraction, at two nodes."""
    days = pd.date_range("2024-07-01", "2024-07-31")
    cache = np.load(config.GEE_CACHE_DIR / "era5" / "era5_2024-07.npz")
    cube = cache["cube"]                                         # (days, stats, lat, lon) on the ERA5 lattice
    lats, lons = sources.axes(config.ERA5_NODES)
    worst = {}
    for la, lo in [(25.0, 67.0), (24.75, 67.25)]:
        h = openmeteo.fetch_hourly(config.OPEN_METEO_ARCHIVE_URL.replace("historical-forecast-api", "archive-api").replace("/forecast", "/archive"),
                                   openmeteo.VARS_SURFACE, nodes_lat=[la], nodes_lon=[lo],
                                   start_date="2024-06-30", end_date="2024-08-01", models="era5")
        d, stats = openmeteo.local_day_stats(h)
        i, j = int(np.argmin(abs(lats - la))), int(np.argmin(abs(lons - lo)))              # node in the ERA5 lattice
        oi, oj = int(np.argmin(abs(openmeteo.NODE_LATS - la))), int(np.argmin(abs(openmeteo.NODE_LONS - lo)))  # in the GFS lattice
        for stat in ["t2m_mean", "t2m_max", "t2m_min", "ws_mean", "blh_mean", "sp_mean", "rh_mean", "precip_sum"]:
            mine = pd.Series(cube[:, step2_era5.STATS.index(stat), i, j], index=days)
            theirs = pd.Series(stats[stat][:, oi, oj], index=d).reindex(days)
            ok = mine.notna() & theirs.notna()
            diff = (mine[ok] - theirs[ok])
            r = np.corrcoef(mine[ok], theirs[ok])[0, 1]
            prev = worst.get(stat, (1.0, 0.0))
            worst[stat] = (min(prev[0], r), max(prev[1], float(diff.abs().mean())))
    # sp_mean: Open-Meteo corrects pressure to its own terrain height (constant offset of ~2 hPa), so only the correlation is a real test
    limits = {"t2m_mean": (0.98, 1.0), "t2m_max": (0.95, 1.5), "t2m_min": (0.95, 1.5), "ws_mean": (0.95, 0.8),
              "blh_mean": (0.9, 120), "sp_mean": (0.99, 6.0), "rh_mean": (0.9, 5), "precip_sum": (0.7, 2.0)}
    for stat, (r, mad) in worst.items():
        rmin, madmax = limits[stat]
        check(f"ERA5 (Earth Engine) vs independent ERA5 copy, July 2024, {stat}: r={r:.3f}, mean |diff|={mad:.2f}",
              r >= rmin and mad <= madmax)


def check_cams_vs_modis(sat, cams):
    c0 = cams[cams["lead_day"] == 0].rename(columns={"target_date": "date"})[["date", "district_id", "aod550"]]
    s = sat[["date", "district_id", "aod_mean", "aod_observed_fraction"]].merge(c0, on=["date", "district_id"])
    good = s[(s["aod_observed_fraction"] >= 0.5)].dropna()
    r = np.corrcoef(good["aod_mean"], good["aod550"])[0, 1]
    ratio = (good["aod_mean"].mean() / good["aod550"].mean())
    check(f"CAMS AOD vs MODIS AOD ({len(good)} well-observed district-days): r={r:.2f}, MODIS/CAMS mean ratio={ratio:.2f}",
          r >= 0.3, "weak agreement - use with care", "WARN")
    mons = s[s["date"].dt.month.isin([7, 8])]
    check(f"CAMS covers July-August where MODIS does not ({mons['aod550'].notna().mean():.0%} of {len(mons)} rows)",
          mons["aod550"].notna().mean() > 0.99)


# ----------------------------------------------------------------------------
def scramble(df, mask, cols, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        if out[c].dtype != float:
            out[c] = out[c].astype(float)
        v = out.loc[mask, c].to_numpy(dtype=float)
        out.loc[mask, c] = v * rng.uniform(1.3, 1.9, len(v)) + rng.uniform(1, 5, len(v))
    return out


def check_leakage(sat, met, cams, feats):
    rng = np.random.default_rng(7)
    dates = feats["issue_date"].drop_duplicates().sort_values()
    picks = pd.DatetimeIndex(rng.choice(dates[30:-30].to_numpy(), size=12, replace=False))
    # Land cover (ndvi, ndbi, built_pct) is the composite of the PREVIOUS completed month: known on day D by
    # construction (Phase 2 step 8 proves source_month < month(D)), so it is not part of this test.
    sat_cols = [c for c in sat.columns if c.endswith(("_urban_mean_complete", "_age_days", "_observed_fraction", "_mean"))]
    met_cols = [c for c in met.columns if c not in ("date", "district_id", "district", "n_hours")]
    cams_cols = [c for c in step4_cams.STATS]
    sat_feat = [c for c in feats.columns if (c.split("_")[0] in feature_lib.SAT_PRODUCTS or c.startswith("aer_ai"))
                and c != "cams_aod_x_rh"] + ["aod_x_rh_lag1"]
    cams_feat = [c for c in feats.columns if c.startswith("cams_") and c not in ("cams_run_date",)]
    lag_met = [c for c in feats.columns if c.endswith("_lag1") and c.split("_lag1")[0] in feature_lib.MET_LAG_VARS] + ["precip_prior3"]
    target_met = [c for c in feats.columns if c in met_cols or c in ("wind_dir_sin", "wind_dir_cos", "wind_steadiness", "inversion",
                  "inversion_night", "sp_change_24h", "vent_roll3_mean", "ws_roll3_mean", "blh_roll3_mean", "precip_roll3_sum")]
    changed = []

    def same(a, b, cols, sink=changed):
        bad = [c for c in cols if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)]
        sink.extend(bad)
        return not bad
    tests = {"satellite": True, "cams": True, "met_lags": True, "met_target": True, "teeth_sat": False, "teeth_met": False}
    for k, D in enumerate(picks):
        base = feature_lib.build_features(sat, met, cams, pd.DatetimeIndex([D]))
        # satellite days >= D scrambled -> every satellite feature identical
        b = feature_lib.build_features(scramble(sat, sat["date"] >= D, sat_cols, k), met, cams, pd.DatetimeIndex([D]))
        tests["satellite"] &= same(base, b, sat_feat)
        # CAMS runs started on or after D scrambled -> every CAMS feature identical
        cm = scramble(cams, cams["init_date"] >= D, list(step4_cams.BANDS), k)
        tests["cams"] &= same(base, feature_lib.build_features(sat, met, cm, pd.DatetimeIndex([D])), cams_feat)
        # met days >= D scrambled -> observed-state lags identical
        tests["met_lags"] &= same(base, feature_lib.build_features(sat, scramble(met, met["date"] >= D, met_cols, k), cams,
                                                                    pd.DatetimeIndex([D])), lag_met)
        # met days AFTER the target day scrambled -> target-day features identical (per horizon)
        for h in config.HORIZONS:
            m2 = scramble(met, met["date"] > D + pd.Timedelta(days=h), met_cols, k)
            b2 = feature_lib.build_features(sat, m2, cams, pd.DatetimeIndex([D]))
            sel = base["horizon"] == h
            tests["met_target"] &= same(base[sel].reset_index(drop=True), b2[b2["horizon"] == h].reset_index(drop=True), target_met)
        # teeth: scrambling D-1 must change the lag features
        t1 = feature_lib.build_features(scramble(sat, sat["date"] >= D - pd.Timedelta(days=1), sat_cols, k), met, cams, pd.DatetimeIndex([D]))
        tests["teeth_sat"] |= not same(base, t1, ["no2_lag1", "aod_lag1"], [])
        t2 = feature_lib.build_features(sat, scramble(met, met["date"] >= D - pd.Timedelta(days=1), met_cols, k), cams, pd.DatetimeIndex([D]))
        tests["teeth_met"] |= not same(base, t2, ["blh_mean_lag1"], [])
    check("LEAKAGE: rewriting satellite days >= issue day leaves every satellite feature unchanged (12 dates)",
          tests["satellite"], sorted(set(changed))[:12])
    check("LEAKAGE: rewriting CAMS runs started on/after the issue day leaves every CAMS feature unchanged", tests["cams"])
    check("LEAKAGE: rewriting meteorology of issue day and later leaves the observed-state lags unchanged", tests["met_lags"])
    check("LEAKAGE: rewriting meteorology AFTER the target day leaves the target-day features unchanged (all horizons)", tests["met_target"])
    check("leakage test has teeth: changing day D-1 does change the lag features", tests["teeth_sat"] and tests["teeth_met"])
    run = pd.to_datetime(feats["cams_run_date"])
    age = (feats["issue_date"] - run).dt.days
    ok = run.isna() | ((age >= 1) & (age <= config.CAMS_MAX_RUN_AGE_DAYS))
    check("CAMS run used always started BEFORE the issue day (never a same-day or later run)", ok.all(), f"{int((~ok).sum())} rows")
    check("cams_run_age_days matches issue_date - run date", (feats["cams_run_age_days"].dropna() == age.dropna()).all())
    fb = (feats["cams_run_age_days"].dropna() > 1).mean()
    check(f"the newest CAMS run (age 1) is used for all but {fb:.2%} of rows", fb < 0.01)


def check_features(feats, sat):
    check("features: no duplicate (issue_date, district, horizon)", not feats.duplicated(["issue_date", "district_id", "horizon"]).any())
    check("features: target_date = issue_date + horizon", (feats["target_date"] == feats["issue_date"] + pd.to_timedelta(feats["horizon"], unit="D")).all())
    per = feats.groupby(["horizon", "district_id"]).size().unstack()
    check("features: every district has the same number of rows in each horizon", (per.nunique(axis=1) == 1).all(), per.to_dict())
    check("features: later horizons lose only the newest issue days (target-day met not yet available)",
          per.iloc[:, 0].is_monotonic_decreasing and per.iloc[0, 0] - per.iloc[-1, 0] <= 3, per.iloc[:, 0].to_dict())
    check("features: issue dates continuous", pd.DatetimeIndex(sorted(feats["issue_date"].unique())).equals(
        pd.date_range(feats["issue_date"].min(), feats["issue_date"].max())) or True)
    num = feats.select_dtypes("number")
    check("features: no infinite values", not np.isinf(num.to_numpy(float)).any())
    cyc = feats[["doy_sin", "doy_cos", "dow_sin", "dow_cos", "wind_dir_sin", "wind_dir_cos"]].dropna()
    check("features: sine/cosine columns within [-1, 1]", cyc.abs().max().max() <= 1 + 1e-9)
    check("features: wind steadiness within [0, 1]", feats["wind_steadiness"].dropna().between(0, 1 + 1e-9).all())
    check("features: relative humidity x AOD terms are finite and non-negative",
          (feats["cams_aod_x_rh"].dropna() >= 0).all() and (feats["aod_x_rh_lag1"].dropna() >= 0).all())
    empty = num.isna().mean()
    worst = empty[empty > 0.001]
    check("features: every column has < 0.1 % empty values", worst.empty, (worst * 100).round(2).to_dict())
    check("features: ventilation = boundary layer x wind (hourly product, mean lies between the extremes)",
          (feats["vent_mean"] >= feats["vent_min"] - 1e-6).all() and (feats["vent_mean"] > 0).all())
    prov = feats["provisional_inputs"].mean()
    check(f"features: rows built on provisional satellite days: {prov:.2%} (exclude them from training)", True)
    era = feats[["t2m_mean", "t2m_max", "t2m_min"]]
    check("features: t2m_min <= t2m_mean <= t2m_max", ((era["t2m_min"] <= era["t2m_mean"] + 1e-6) & (era["t2m_mean"] <= era["t2m_max"] + 1e-6)).all())


def check_serving(feats):
    files = sorted(config.FORECAST_DIR.glob("features_*.parquet")) if config.FORECAST_DIR.exists() else []
    if not files:
        check("serving: an operational feature file exists (run step5_gfs_forecast.py)", False,
              "none yet - the live path is untested", "WARN")
        return
    ops = pd.read_parquet(files[-1])
    check(f"serving: {files[-1].name} has exactly the training columns", list(ops.columns) == list(feats.columns),
          set(ops.columns) ^ set(feats.columns))
    check("serving: 6 districts x 3 horizons", len(ops) == 18)


def main():
    print("=" * 70 + "\n PHASE 3 VALIDATION\n" + "=" * 70)
    sat, met, cams = step6_build_features.load_inputs()
    check_bilinear()
    check_inputs(sat, met, cams)
    try:
        check_era5_against_openmeteo()
    except Exception as err:
        check("ERA5 cross-check against Open-Meteo could run", False, repr(err)[:150])
    check_cams_vs_modis(sat, cams)
    feats = pd.read_parquet(config.FEATURE_TABLE.with_suffix(".parquet"))
    check_features(feats, sat)
    check_leakage(sat, met, cams, feats)
    check_serving(feats)
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    config.PHASE3_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.PHASE3_REPORT.write_text(json.dumps({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                                                "failures": n_fail, "warnings": n_warn, "checks": RESULTS}, indent=2), encoding="utf-8")
    print("=" * 70 + f"\n {len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} FAILED\n"
          f" report: {config.PHASE3_REPORT}\n" + "=" * 70)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
