"""
PHASE 2 - STEP 8: Validation gate. Phase 3 must not start until this passes.

It re-checks EVERYTHING Phase 2 produced, without downloading anything:

  hygiene      no leftover temporary files, no stray empty files in the project folder
  cache        every cached month is readable, complete, right shape, no infinities
  csv          per product: no duplicate rows, no missing days or districts, no future
               dates, fractions in [0, 1], values in physical ranges, NaN <=> valid_fraction 0
  netcdf       every monthly grid readable, days complete and unique, flags legal, and the
               district means re-computed from the grids equal the CSV (proves the files match)
  leakage      (a) backward gap filling: changing FUTURE days never changes any earlier day
                   (tested on synthetic data and on real cached months)
               (b) Sentinel-2 daily rows only use a strictly earlier, completed month
               (c) built-up weights come from a period before START_DATE
               (d) the *_complete columns only use past days (truncation test)
  final table  rows == days x 6 districts, no duplicates, parquet == csv, completeness stats

Writes outputs/phase2_validation_report.json and prints a summary. Exit code 1 if
any check FAILS (warnings are physical facts such as the monsoon gap, not errors).

Run:
    python phase2_ingestion/step8_validate.py
"""
import calendar
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import config
from cache_io import CACHE_SCHEMA, LEGACY_SCHEMA, TMP_MARK
from gapfill import FLAG_IDW, FLAG_OBSERVED, FLAG_TEMPORAL, gap_fill_cube
from products import PRODUCTS
from spatial_utils import load_grid, zonal_timeseries

# Physical plausibility of DISTRICT MEANS (generous: only impossible values fail).
# SO2: 1 DU = 44600 umol/m2; the Nov 2025 Hayli Gubbi plume (all pixels ~4500-33000) is real.
RANGES = {"no2": (-100, 3000), "o3": (100, 600), "so2": (-1000, 45000), "co": (0, 150),
          "aer_ai": (-20, 20), "aod": (-0.1, 5.0)}
RESULTS = []


def check(name, ok, detail="", level="FAIL"):
    """Record a check. level is what a failure means: FAIL (blocks Phase 3) or WARN (fact)."""
    status = "PASS" if ok else level
    RESULTS.append({"check": name, "status": status, "detail": str(detail)})
    mark = {"PASS": "  ok ", "WARN": " WARN", "FAIL": " FAIL"}[status]
    print(f"[{mark}] {name}" + (f" - {detail}" if detail and status != "PASS" else ""))
    return ok


def yesterday():
    return pd.Timestamp(dt.date.today() - dt.timedelta(days=1))


# ----------------------------------------------------------------------------
def check_hygiene():
    tmp = [p for root in (config.DATA_DIR, config.PROJECT_ROOT / "outputs")
           for p in root.rglob(f"*{TMP_MARK}[0-9]*") if p.is_file()]
    check("no leftover temporary files", not tmp, f"{len(tmp)} found, e.g. {tmp[:2]}")
    stray = [p.name for p in config.PROJECT_ROOT.iterdir()
             if p.is_file() and p.stat().st_size == 0]
    check("no stray empty files in the project folder", not stray, stray)
    quarantined = list((config.GEE_CACHE_DIR / "_quarantine").glob("*")) if (config.GEE_CACHE_DIR / "_quarantine").exists() else []
    check("quarantined cache files (already re-downloaded)", not quarantined,
          f"{len(quarantined)} file(s) in data/cache/gee/_quarantine - safe to delete", "WARN")


def check_cache(shape):
    bad, n = [], 0
    for path in sorted(config.GEE_CACHE_DIR.glob("*/*.npz")):
        # cams/ and era5/ are Phase 3 caches (different cube layout, validated by phase3 step7); once
        # Phase 3 has run they sit next to the Phase 2 product caches and must not be judged by them.
        if path.parent.name in ("_quarantine", "cams", "era5"):
            continue
        n += 1
        try:
            with np.load(path, allow_pickle=False) as z:
                if path.parent.name == "sentinel2":
                    ok = int(z["schema"]) == 2 and z["grids"].shape == (4, *shape) \
                        and z["grids"].dtype == np.float32 and not np.isinf(z["grids"]).any()
                    if not ok:
                        bad.append((path.name, "layout"))
                    continue
                schema = int(z["schema"]) if "schema" in z.files else LEGACY_SCHEMA
                dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
                cube = z["cube"]
            y, m = map(int, path.stem.split("_")[-1].split("-"))
            expected = pd.date_range(dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1]))
            if schema not in (CACHE_SCHEMA, LEGACY_SCHEMA) or not dates.equals(expected) \
                    or cube.shape != (len(dates), *shape) or cube.dtype != np.float32 \
                    or np.isinf(cube).any():
                bad.append((path.name, "content"))
        except Exception as err:
            bad.append((path.name, repr(err)[:60]))
    check(f"all {n} cached months readable and complete", not bad, bad[:5])


def check_product_csv(key, grid):
    path = config.DISTRICT_DAILY_DIR / f"{key}.csv"
    if not check(f"{key}: csv exists", path.exists(), path):
        return None
    t = pd.read_csv(path, parse_dates=["date"])
    need = ["date", "district_id", "district", "mean", "std", "urban_mean", "observed_fraction",
            "idw_fraction", "temporal_fraction", "valid_fraction"]
    if not check(f"{key}: columns", all(c in t.columns for c in need), set(need) - set(t.columns)):
        return None
    check(f"{key}: no duplicate (date, district) rows", not t.duplicated(["date", "district_id"]).any(),
          int(t.duplicated(["date", "district_id"]).sum()))
    days = pd.date_range(config.START_DATE, t["date"].max())
    check(f"{key}: starts at START_DATE", t["date"].min() == pd.Timestamp(config.START_DATE), t["date"].min())
    check(f"{key}: every day and district present", len(t) == len(days) * len(grid["ids"])
          and set(t["date"]) == set(days), f"{len(t)} rows vs {len(days) * len(grid['ids'])}")
    check(f"{key}: no dates in the future", t["date"].max() <= yesterday(), t["date"].max())
    idmap = dict(zip(grid["ids"], grid["names"]))
    check(f"{key}: district ids/names match Phase 1",
          all(idmap.get(i) == n for i, n in zip(t["district_id"], t["district"])))
    num = t[["mean", "std", "urban_mean", "observed_fraction", "idw_fraction",
             "temporal_fraction", "valid_fraction"]]
    check(f"{key}: no infinite values", not np.isinf(num.to_numpy(dtype=float)).any())
    fr = t[["observed_fraction", "idw_fraction", "temporal_fraction", "valid_fraction"]]
    check(f"{key}: fractions within [0, 1]", ((fr >= -1e-6) & (fr <= 1 + 1e-6)).all().all())
    gap = (t["observed_fraction"] + t["idw_fraction"] + t["temporal_fraction"] - t["valid_fraction"]).abs()
    check(f"{key}: observed + IDW + temporal = valid", (gap < 1e-3).all(), f"max diff {gap.max():.4f}")
    check(f"{key}: mean is empty exactly when valid_fraction is 0",
          (t["mean"].isna() == (t["valid_fraction"] <= 1e-9)).all(),
          int((t["mean"].isna() != (t["valid_fraction"] <= 1e-9)).sum()))
    check(f"{key}: urban_mean empty exactly when mean is empty",
          (t["urban_mean"].isna() == t["mean"].isna()).mean() > 0.999,
          f"{int((t['urban_mean'].isna() != t['mean'].isna()).sum())} rows differ", "WARN")
    lo, hi = RANGES[key]
    out = t["mean"].dropna()
    check(f"{key}: district means in physical range [{lo}, {hi}]", out.between(lo, hi).all(),
          f"{int((~out.between(lo, hi)).sum())} outside; min {out.min():.3g} max {out.max():.3g}")
    check(f"{key}: std is not negative", (t["std"].dropna() >= 0).all())
    return t


def check_netcdf(key, table, grid):
    files = sorted((config.SATELLITE_DIR / key).glob(f"{key}_*.nc"))
    check(f"{key}: one netcdf per month, no extras",
          [f.stem.split("_")[-1] for f in files] == [str(p) for p in pd.period_range(
              table["date"].min(), table["date"].max(), freq="M")], f"{len(files)} files")
    built = None
    with rasterio.open(config.BUILT_WEIGHT_FILE) as src:
        built = src.read(1)
    problems, times = [], []
    max_err = 0.0
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                ds = ds.load()
        except Exception as err:
            problems.append((f.name, repr(err)[:50]))
            continue
        time = pd.DatetimeIndex(ds.time.values)
        period = pd.Period(f.stem.split("_")[-1], "M")
        exp = pd.date_range(period.start_time, min(period.end_time.normalize(), table["date"].max()))
        if not time.equals(exp):
            problems.append((f.name, "days differ from calendar"))
        times.append(time)
        v, fl = ds["value"].values, ds["flag"].values
        if v.shape[1:] != grid["shape"] or not np.isin(fl, [0, 1, 2, 3, 9]).all():
            problems.append((f.name, "shape/flags"))
            continue
        # flags 0/1/2 must carry a value, flag 3 must be empty; flag 9 (outside all districts,
        # never used in any district mean) may keep the raw satellite value.
        if not (np.isfinite(v[np.isin(fl, [0, 1, 2])]).all() and not np.isfinite(v[fl == 3]).any()):
            problems.append((f.name, "value/flag mismatch"))
        lo, hi = PRODUCTS[key]["valid_range"]
        obs = v[fl == 0]                                    # real observations must be in range
        if ((obs < lo) | (obs > hi)).any():
            problems.append((f.name, "observed values outside valid range"))
        # recompute district means and compare with the CSV
        mean, _, _ = zonal_timeseries(v, grid)
        umean = zonal_timeseries(v, grid, extra_weight=built)[0]
        sub = table[table["date"].isin(time)].sort_values(["date", "district_id"])
        csv_mean = sub["mean"].to_numpy().reshape(len(time), -1)
        csv_um = sub["urban_mean"].to_numpy().reshape(len(time), -1)
        for a, b in ((mean, csv_mean), (umean, csv_um)):
            both = np.isfinite(a) & np.isfinite(b)
            if (np.isfinite(a) != np.isfinite(b)).any():
                problems.append((f.name, "NaN pattern differs from csv"))
            if both.any():
                err = np.abs(a[both] - b[both]) / (np.abs(a[both]) + 1e-3)
                max_err = max(max_err, float(err.max()))
                if err.max() > 1e-4:
                    problems.append((f.name, f"csv mismatch {err.max():.1e}"))
    all_t = pd.DatetimeIndex(np.concatenate(times)) if times else pd.DatetimeIndex([])
    check(f"{key}: netcdf days unique and complete", not all_t.has_duplicates and len(all_t) == table["date"].nunique(),
          f"{len(all_t)} days vs {table['date'].nunique()}")
    check(f"{key}: netcdf readable, flags legal, district means equal csv", not problems,
          f"{problems[:3]} (max rel. error {max_err:.1e})")


# ----------------------------------------------------------------------------
# Leakage tests
# ----------------------------------------------------------------------------
def causal_test(cube, targets, name, cuts):
    """Perturbing days >= cut must not change gap-filled days < cut."""
    kw = dict(res_m=1000, radius_m=12_000, mode=config.TEMPORAL_FILL_MODE, window=config.TEMPORAL_FILL_WINDOW)
    full, flags = gap_fill_cube(cube, targets, **kw)
    rng = np.random.default_rng(0)
    ok = True
    for cut in cuts:
        changed = cube.copy()
        changed[cut:] = np.where(rng.random(changed[cut:].shape) < 0.5,
                                 rng.normal(50, 20, changed[cut:].shape), np.nan)
        alt, _ = gap_fill_cube(changed, targets, **kw)
        ok &= bool(np.array_equal(full[:cut], alt[:cut], equal_nan=True))
    check(f"leakage test (gap filling uses only past days): {name}", ok)


def check_leakage(grid, tables):
    check("TEMPORAL_FILL_MODE is 'backward'", config.TEMPORAL_FILL_MODE == "backward",
          config.TEMPORAL_FILL_MODE)
    targets = grid["weights"].sum(axis=0) > 0
    rng = np.random.default_rng(1)
    syn = rng.normal(50, 10, (14, *grid["shape"])).astype("float32")
    syn[rng.random(syn.shape) < 0.55] = np.nan
    causal_test(syn, targets, "synthetic cube", cuts=[3, 7, 11])
    for key in ("no2", "aod"):                            # real cached months incl. a monsoon one
        month = "2022-07" if key == "aod" else "2022-03"
        f = config.GEE_CACHE_DIR / key / f"{key}_{month}.npz"
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                causal_test(z["cube"][:16], targets, f"real {key} {month}", cuts=[5, 10])

    # (b) Sentinel-2 daily rows: strictly earlier completed month
    lc = pd.read_csv(config.DISTRICT_DAILY_DIR / "landcover.csv", parse_dates=["date"])
    src_end = pd.PeriodIndex(lc["source_month"], freq="M").end_time.normalize()
    check("Sentinel-2 daily rows use a strictly earlier, completed month",
          (src_end < lc["date"]).all() and (lc["lc_age_days"] >= 1).all())
    check("Sentinel-2 daily table has no duplicates", not lc.duplicated(["date", "district_id"]).any())
    check("Sentinel-2 has values for every day", lc[["ndvi", "ndbi", "built_pct"]].notna().all().all(),
          lc[["ndvi", "ndbi", "built_pct"]].isna().sum().to_dict())
    # (c) built-up weights period
    check("built-up weight period ends before START_DATE",
          pd.Period(config.BUILT_WEIGHT_PERIOD[1], "M").end_time < pd.Timestamp(config.START_DATE),
          config.BUILT_WEIGHT_PERIOD)
    # (d) *_complete columns are causal
    from step7_finalize import complete_district_series
    ok = True
    for key in ("aod", "no2"):
        g = tables[key][tables[key]["district_id"] == grid["ids"][0]].sort_values("date").reset_index(drop=True)
        full = complete_district_series(g)
        for cut in (400, 800, 1200):
            trunc = complete_district_series(g.iloc[:cut])
            ok &= full.iloc[:cut][["mean_complete", "urban_mean_complete", "age_days"]].reset_index(drop=True) \
                .equals(trunc[["mean_complete", "urban_mean_complete", "age_days"]].reset_index(drop=True))
    check("*_complete columns use only past days (truncation test)", ok)


def check_final(tables, grid):
    csv_path, pq_path = config.PHASE2_TABLE.with_suffix(".csv"), config.PHASE2_TABLE.with_suffix(".parquet")
    if not check("final table exists (csv + parquet)", csv_path.exists() and pq_path.exists()):
        return
    t = pd.read_csv(csv_path, parse_dates=["date"])
    p = pd.read_parquet(pq_path)
    days = pd.date_range(config.START_DATE, t["date"].max())
    check("final: rows == days x districts", len(t) == len(days) * len(grid["ids"]), len(t))
    check("final: no duplicate (date, district) rows", not t.duplicated(["date", "district_id"]).any())
    check("final: parquet matches csv", len(p) == len(t) and (p["date"].to_numpy() == t["date"].to_numpy()).all()
          and np.allclose(p["no2_mean"].to_numpy(float), t["no2_mean"].to_numpy(float), rtol=1e-5, equal_nan=True))
    check("final: table ends at the last day of the daily files", t["date"].max() == max(
        x["date"].max() for x in tables.values()))
    for key in PRODUCTS:
        empty = (t[f"{key}_complete_level"] == len(config.COMPLETE_WINDOWS_DAYS) + 1).mean()
        check(f"final: {key}_mean_complete empty exactly at level 3",
              (t[f"{key}_mean_complete"].isna() == (t[f"{key}_complete_level"] == len(config.COMPLETE_WINDOWS_DAYS) + 1)).all())
        check(f"final: {key} rows still empty after 90-day fallback", empty < 0.005,
              f"{empty:.2%}", "WARN")


def monsoon_summary(tables):
    out = {}
    for key, t in tables.items():
        m = t.assign(month=t["date"].dt.month)
        out[key] = {"real_observation_share_by_month": (m.groupby("month")["observed_fraction"].mean().round(3)).to_dict(),
                    "days_with_no_district_value_pct_by_month": (m.assign(e=m["mean"].isna()).groupby("month")["e"].mean() * 100).round(1).to_dict()}
    return out


def main():
    grid = load_grid(1000)
    print("=" * 70 + "\n PHASE 2 VALIDATION\n" + "=" * 70)
    check_hygiene()
    check_cache(grid["shape"])
    tables = {}
    for key in PRODUCTS:
        t = check_product_csv(key, grid)
        if t is not None:
            tables[key] = t
            check_netcdf(key, t, grid)
    if len(tables) == len(PRODUCTS):
        check_leakage(grid, tables)
        check_final(tables, grid)
        summary = monsoon_summary(tables)
    else:
        summary = {}
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    config.VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.VALIDATION_REPORT.write_text(json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "failures": n_fail, "warnings": n_warn, "checks": RESULTS, "coverage_summary": summary},
        indent=2, default=str), encoding="utf-8")
    print("=" * 70)
    print(f" {len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} FAILED")
    print(f" report: {config.VALIDATION_REPORT}")
    print("=" * 70)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
