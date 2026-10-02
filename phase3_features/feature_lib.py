"""
feature_lib.py - turns the Phase 2 satellite table + daily meteorology + CAMS runs into one
row per (issue_date D, district, horizon h) for direct multi-horizon forecasting.

THE TIMING RULE (see config.py). A forecast is issued on the morning of day D. It may use
    satellite data of days <= D-1          (the overpasses of day D are not yet available)
    meteorology of D-1 and earlier as observed/analysed, and for the TARGET day D+h a forecast
        (training: ERA5 stands in for the forecast = "perfect prognosis"; operation: GFS)
    CAMS model aerosol from the 12Z run of day D-1 (for D+h) or of older runs (for lags)
Every "lag" / "rolling" feature below is anchored to D-1, never to D, and the CAMS run date is
stored. step7_validate.py proves this by rewriting the future and checking nothing moves.

Row layout (long format): issue_date, target_date, horizon, district_id, district, met_source,
then features. Train one model per horizon (filter on `horizon`), as the plan prescribes.

Column groups
    <p>_lag1..3, <p>_roll3_mean/std, <p>_roll7_mean/std, <p>_age_d1, <p>_obsfrac_d1
                                  satellite series as of D-1  (p = no2, o3, so2, co, aer_ai, aod)
    ndvi, ndbi, built_pct         Sentinel-2 land cover (composite of the previous completed month)
    t2m_*, rh_*, ws_*, wind_*, blh_*, vent_*, sp_*, precip_*, inversion*   meteorology of the TARGET day
    *_lag1 (met), precip_prior3   meteorology as observed on D-1 .. D-3
    vent_roll3_mean, ws_roll3_mean, precip_roll3_sum, blh_roll3_mean       3 days ending on the target day
    cams_*                        model aerosol for the TARGET day (run D-1, else D-2: cams_run_age_days)
                                  + cams_*_lag1 / roll3 / roll7
    aod_x_rh_lag1, cams_aod_x_rh  AOD x relative humidity (hygroscopic growth)
    doy_sin/cos, dow_sin/cos, is_sunday       cyclical calendar encodings of the TARGET day
"""
import numpy as np
import pandas as pd

import config

SAT_PRODUCTS = ["no2", "o3", "so2", "co", "aer_ai", "aod"]
MET_LAG_VARS = ["t2m_mean", "rh_mean", "ws_mean", "blh_mean", "vent_mean", "sp_mean"]
CAMS_VARS = ["aod550", "dust_aod550", "pm25", "pm10"]


# ----------------------------------------------------------------------------
# Generic causal helpers (reused in Phase 4 for the ground AQI history)
# ----------------------------------------------------------------------------
def causal_lags_rolls(wide, lags=(1, 2, 3), windows=(3, 7), min_frac=1.0):
    """
    wide: DataFrame indexed by a CONTINUOUS daily DatetimeIndex (one column per series).
    Returns a dict of DataFrames anchored to the ISSUE date D, using days <= D-1 only:
        lag{k}      = x[D-k]
        roll{w}_mean, roll{w}_std = mean / std of x[D-w .. D-1]
    A window needs at least ceil(w * min_frac) real values (default: all of them; ground data with gaps
    uses 0.5). Standard deviations always need at least 2.
    """
    if not wide.index.is_monotonic_increasing or not wide.index.equals(pd.date_range(wide.index[0], wide.index[-1])):
        raise ValueError("series must be a continuous, sorted daily index")
    out = {f"lag{k}": wide.shift(k) for k in lags}
    past = wide.shift(1)
    for w in windows:
        need = max(1, int(np.ceil(w * min_frac)))
        out[f"roll{w}_mean"] = past.rolling(w, min_periods=need).mean()
        out[f"roll{w}_std"] = past.rolling(w, min_periods=max(2, need)).std()
    return out


def _wide(df, col, until=None):
    """(date, district_id) table -> DataFrame date x district for one column, continuous daily index.
    `until` extends the index (with empty rows) so lags can be anchored on a day the table does not have yet."""
    w = df.pivot(index="date", columns="district_id", values=col).sort_index()
    end = w.index.max() if until is None else max(w.index.max(), pd.Timestamp(until))
    return w.reindex(pd.date_range(w.index.min(), end))


def _stack(frames, name_of):
    """{name: wide DataFrame (date x district)} -> long DataFrame indexed (date, district_id)."""
    parts = {name_of(k): v.stack(future_stack=True) for k, v in frames.items()}
    out = pd.DataFrame(parts)
    out.index.names = ["date", "district_id"]
    return out


# ----------------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------------
def satellite_block(sat, until):
    """Satellite state as of D-1 (index = issue date D, up to `until`)."""
    frames = {}
    for p in SAT_PRODUCTS:
        w = _wide(sat, f"{p}_urban_mean_complete", until)
        for name, df in causal_lags_rolls(w).items():
            frames[(p, name)] = df
        frames[(p, "age_d1")] = _wide(sat, f"{p}_age_days", until).shift(1)
        frames[(p, "obsfrac_d1")] = _wide(sat, f"{p}_observed_fraction", until).shift(1)
    out = _stack(frames, lambda k: f"{k[0]}_{k[1]}")
    out["provisional_inputs"] = _wide(sat, "provisional", until).shift(1).astype(float).stack(future_stack=True).reindex(out.index)
    return out


def landcover_block(monthly, issue_dates, district_ids):
    """
    Sentinel-2 land cover known on issue day D = the composite of the PREVIOUS calendar month
    (a composite is complete only when its month is over). Read straight from the monthly table so it also
    works on a live issue day that the daily satellite table does not contain yet.
    """
    m = monthly.set_index(["month", "district_id"])[["ndvi", "ndbi", "built_pct"]]
    rows = []
    for d in issue_dates:
        src = (d.to_period("M") - 1).start_time
        for did in district_ids:
            key = (src, did)
            rows.append({"date": d, "district_id": did, **(m.loc[key].to_dict() if key in m.index else {})})
    out = pd.DataFrame(rows).set_index(["date", "district_id"])
    return out.reindex(columns=["ndvi", "ndbi", "built_pct"])


def derive_met(met):
    """Derived meteorological columns (per date and district) from the daily statistics."""
    m = met.copy()
    m["precip_sum"] = m["precip_sum"].clip(lower=0)                         # rounding can leave -1e-5
    speed = np.hypot(m["u10_mean"], m["v10_mean"])
    m["wind_dir_sin"] = -m["u10_mean"] / speed.where(speed > 1e-6)          # meteorological "from" direction
    m["wind_dir_cos"] = -m["v10_mean"] / speed.where(speed > 1e-6)
    m["wind_steadiness"] = speed / m["ws_mean"].where(m["ws_mean"] > 1e-6)   # 1 = constant direction all day
    if "t850_mean" in m:
        m["inversion"] = m["t2m_mean"] - m["t850_mean"]                     # T(2 m) - T(850 hPa): small = stable
        m["inversion_night"] = m["t2m_min"] - m["t850_mean"]
    return m


def met_block(met, issue_dates, horizon):
    """Meteorology of the target day D+h, of D-1.. as observed, and 3-day windows ending on the target day."""
    m = derive_met(met)
    cols = [c for c in m.columns if c not in ("date", "district_id", "district", "n_hours")]
    frames = {c: _wide(m, c) for c in cols}
    idx = pd.date_range(min(frames[cols[0]].index.min(), issue_dates.min() - pd.Timedelta(days=10)),
                        max(frames[cols[0]].index.max(), issue_dates.max() + pd.Timedelta(days=horizon)))
    frames = {c: f.reindex(idx) for c, f in frames.items()}
    tgt = {}
    for c in cols:                                              # value on the target day D+h
        tgt[c] = frames[c].shift(-horizon)
    tgt["sp_change_24h"] = tgt["sp_mean"] - frames["sp_mean"].shift(-horizon + 1)
    for c, name in [("vent_mean", "vent_roll3_mean"), ("ws_mean", "ws_roll3_mean"), ("blh_mean", "blh_roll3_mean")]:
        tgt[name] = frames[c].rolling(3, min_periods=3).mean().shift(-horizon)          # days D+h-2 .. D+h
    tgt["precip_roll3_sum"] = frames["precip_sum"].rolling(3, min_periods=3).sum().shift(-horizon)
    for c in MET_LAG_VARS:                                      # observed state of D-1
        tgt[f"{c}_lag1"] = frames[c].shift(1)
    tgt["precip_prior3"] = frames["precip_sum"].rolling(3, min_periods=3).sum().shift(1)   # D-3 .. D-1
    out = _stack(tgt, lambda k: k)
    out["met_available"] = out["t2m_mean"].notna()
    return out


def cams_block(cams, issue_dates, horizon):
    """
    CAMS aerosol for the target day D+h, from the 12Z run of D-1 (lead day h). If that run is missing
    or incomplete (Earth Engine can lag by days) the newest OLDER run that has the day is used instead
    (D-2; lead day h+1). `cams_run_age_days` says which (1 = normal). Lags come from
    the lead-0 series of older runs.
    """
    c = cams.copy()
    c["init_date"] = pd.to_datetime(c["init_date"]); c["target_date"] = pd.to_datetime(c["target_date"])
    cur = None
    for age in range(1, config.CAMS_MAX_RUN_AGE_DAYS + 1):          # age = issue date - run date, in days
        part = c[c["lead_day"] == horizon + age - 1].copy()
        part["date"] = part["init_date"] + pd.Timedelta(days=age)
        part = part.set_index(["date", "district_id"])[CAMS_VARS + ["init_date"]]
        part = part.rename(columns={**{v: f"cams_{v}" for v in CAMS_VARS}, "init_date": "cams_run_date"})
        part["cams_run_age_days"] = float(age)
        if cur is None:
            cur = part
        else:
            part = part.reindex(cur.index.union(part.index))
            cur = cur.reindex(part.index)
            missing = cur["cams_aod550"].isna()
            cur.loc[missing] = part.loc[missing]
    # lags: series by TARGET date from lead 0 (run started the day before the target day)
    lag0 = c[c["lead_day"] == 0]
    frames = {}
    for v in ("aod550", "pm25"):
        w = _wide(lag0.rename(columns={"target_date": "date"}), v)
        for name, df in causal_lags_rolls(w, lags=(1,), windows=(3, 7)).items():
            if name.endswith("_std"):
                continue
            frames[(v, name)] = df
    frames[("dust_aod550", "lag1")] = _wide(lag0.rename(columns={"target_date": "date"}), "dust_aod550").shift(1)
    lagged = _stack(frames, lambda k: f"cams_{k[0]}_{k[1]}")
    return cur.join(lagged, how="outer")


def calendar_block(target_dates):
    doy = target_dates.dayofyear.to_numpy()
    dow = target_dates.dayofweek.to_numpy()                     # Monday = 0
    return pd.DataFrame({"doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                         "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
                         "is_sunday": (dow == 6).astype("int8")}, index=target_dates)


# ----------------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------------
def build_features(sat, met, cams, issue_dates, horizons=config.HORIZONS, met_source="era5", landcover=None):
    """
    sat, met, cams: the three input tables (see module docstring).
    issue_dates:    DatetimeIndex of forecast issue days D.
    landcover:      monthly Sentinel-2 table (district_monthly/sentinel2.csv); default: read it from config.
    Returns the long feature table (one row per issue_date, district, horizon).
    """
    issue_dates = pd.DatetimeIndex(issue_dates)
    sat_f = satellite_block(sat, until=issue_dates.max())
    if landcover is None:
        landcover = pd.read_csv(config.DISTRICT_MONTHLY_DIR / "sentinel2.csv", parse_dates=["month"])
    sat_f = sat_f.join(landcover_block(landcover, issue_dates, sorted(sat["district_id"].unique())))
    blocks = []
    for h in horizons:
        m = met_block(met, issue_dates, h)
        c = cams_block(cams, issue_dates, h)
        df = sat_f.join(m, how="inner").join(c, how="left")
        df = df[df.index.get_level_values("date").isin(issue_dates)].reset_index()
        df = df.rename(columns={"date": "issue_date"})
        df.insert(1, "horizon", h)
        df.insert(2, "target_date", df["issue_date"] + pd.Timedelta(days=h))
        df["met_source"] = met_source
        df = pd.concat([df, calendar_block(pd.DatetimeIndex(df["target_date"])).reset_index(drop=True)], axis=1)
        df["cams_aod_x_rh"] = df["cams_aod550"] * df["rh_mean"] / 100
        df["aod_x_rh_lag1"] = df["aod_lag1"] * df["rh_mean_lag1"] / 100
        blocks.append(df)
    out = pd.concat(blocks, ignore_index=True)
    names = dict(zip(sat["district_id"], sat["district"]))
    out.insert(3, "district", out["district_id"].map(names))
    lead = ["issue_date", "target_date", "horizon", "district_id", "district", "met_source"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    return out.sort_values(["issue_date", "district_id", "horizon"]).reset_index(drop=True)
