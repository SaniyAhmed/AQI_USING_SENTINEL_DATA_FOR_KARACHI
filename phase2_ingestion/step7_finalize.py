"""
PHASE 2 - STEP 7: Build the final daily table (one row per date and district).

Reads every data/processed/district_daily/<product>.csv and landcover.csv and
writes
    data/processed/phase2_daily_features.csv
    data/processed/phase2_daily_features.parquet     (same content, faster to load)

For every satellite product p the table has
    p_mean, p_urban_mean, p_std          district values after gap filling
    p_observed_fraction                  share of the district with REAL retrievals that day
    p_valid_fraction                     share with a value after gap filling
    p_mean_complete, p_urban_mean_complete
                                         never-empty version (see below)
    p_complete_level                     0 = direct value (valid_fraction >= MIN_VALID_FRACTION)
                                         1 = median of the last 30 days' direct values
                                         2 = median of the last 90 days' direct values
                                         3 = still empty (no direct value in the last 90 days)
    p_age_days                           days since the last direct value (0 = today)
plus ndvi, ndbi, built_pct (Sentinel-2, previous month's composite), and
`provisional` = the day is recent enough that final satellite files may still
change (do not train on provisional rows).

Why "complete" columns?  In the monsoon MODIS AOD has almost no retrievals
(clouds; verified: even the raw, unfiltered MAIAC output has 0-17 % coverage
on July days), so `p_mean` is often empty. Deep-learning models (LSTM/TFT)
cannot take empty inputs, so `p_mean_complete` carries the recent past
forward. Only PAST days are used (trailing windows that exclude the day
itself), so nothing leaks from the future, and `p_complete_level` /
`p_age_days` tell the model how stale the number is. Gradient-boosted trees
can use the plain `p_mean` (they accept empty values) - use whichever you like.

Run:
    python phase2_ingestion/step7_finalize.py
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

import config
from cache_io import atomic_csv, atomic_write, pipeline_lock

PRODUCT_COLUMNS = ["mean", "urban_mean", "std", "observed_fraction", "valid_fraction"]


def complete_district_series(g):
    """
    g: one district's rows sorted by date (continuous days).
    Returns a DataFrame with mean_complete, urban_mean_complete, complete_level, age_days.
    Only values from strictly earlier days are used to fill (shift(1) + trailing window).
    """
    g = g.set_index("date")
    direct_ok = g["valid_fraction"] >= config.MIN_VALID_FRACTION
    out = pd.DataFrame(index=g.index)
    level = pd.Series(np.where(direct_ok & g["mean"].notna(), 0, np.nan), index=g.index)
    for col in ["mean", "urban_mean"]:
        direct = g[col].where(direct_ok)
        filled = direct.copy()
        for lvl, window in enumerate(config.COMPLETE_WINDOWS_DAYS, start=1):
            past_median = direct.shift(1).rolling(f"{window}D", min_periods=1).median()
            use = filled.isna() & past_median.notna()
            filled[use] = past_median[use]
            if col == "mean":
                level[use & level.isna()] = lvl
        out[f"{col}_complete"] = filled
    out["complete_level"] = level.fillna(len(config.COMPLETE_WINDOWS_DAYS) + 1).astype("int8")
    pos = np.arange(len(g), dtype=float)
    last_direct = pd.Series(np.where((level == 0).to_numpy(), pos, np.nan)).ffill().to_numpy()
    out["age_days"] = pd.Series(pos - last_direct, index=g.index)      # NaN until the first direct value
    return out.reset_index()


def build_product_frame(key, path):
    t = pd.read_csv(path, parse_dates=["date"]).sort_values(["district_id", "date"])
    parts = []
    for did, g in t.groupby("district_id", sort=False):
        parts.append(complete_district_series(g).assign(district_id=did))
    comp = pd.concat(parts)
    t = t.merge(comp, on=["date", "district_id"], how="left", validate="one_to_one")
    cols = PRODUCT_COLUMNS + ["mean_complete", "urban_mean_complete", "complete_level", "age_days"]
    t = t[["date", "district_id", "district"] + cols]
    return t.rename(columns={c: f"{key}_{c}" for c in cols})


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from products import PRODUCTS
    frames = []
    for key in PRODUCTS:
        path = config.DISTRICT_DAILY_DIR / f"{key}.csv"
        if not path.exists():
            raise SystemExit(f"{path} is missing - run steps 3 and 4 first.")
        frames.append(build_product_frame(key, path))
    table = frames[0]
    for f in frames[1:]:
        table = table.merge(f.drop(columns="district"), on=["date", "district_id"],
                            how="outer", validate="one_to_one")
    lc = pd.read_csv(config.DISTRICT_DAILY_DIR / "landcover.csv", parse_dates=["date"])
    lc = lc[["date", "district_id", "ndvi", "ndbi", "built_pct", "source_month", "lc_age_days"]]
    table = table.merge(lc, on=["date", "district_id"], how="left", validate="one_to_one")

    provisional_from = pd.Timestamp(dt.date.today()) - pd.Timedelta(days=config.PROVISIONAL_DAYS)
    table["provisional"] = table["date"] >= provisional_from
    table = table.sort_values(["date", "district_id"]).reset_index(drop=True)
    if table.duplicated(["date", "district_id"]).any():
        raise RuntimeError("duplicate (date, district) rows in the final table")

    atomic_csv(table, config.PHASE2_TABLE.with_suffix(".csv"), float_format="%.6g")
    atomic_write(config.PHASE2_TABLE.with_suffix(".parquet"),
                 lambda p: table.to_parquet(p, index=False))
    print(f"Final table: {len(table)} rows x {table.shape[1]} columns, "
          f"{table['date'].min().date()} -> {table['date'].max().date()}, "
          f"{table['district_id'].nunique()} districts")
    print("\nShare of rows by how the *_mean_complete value was obtained "
          "(0 direct, 1 last 30 d, 2 last 90 d, 3 empty):")
    lev = pd.DataFrame({k: table[f"{k}_complete_level"].value_counts(normalize=True)
                        for k in PRODUCTS}).fillna(0).sort_index()
    print((lev * 100).round(1).to_string())
    print(f"\nSaved {config.PHASE2_TABLE.with_suffix('.csv')} and .parquet")


if __name__ == "__main__":
    with pipeline_lock():
        main()
