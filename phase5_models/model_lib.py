"""
model_lib.py - shared helpers for Phase 5 (used by every step1..step4 script).

Design matrix
    feature_columns(df)      every column of the Phase 4 dataset except config.NON_FEATURE_COLS
    build_design_matrix(df)  the feature columns + a FIXED 6-column one-hot encoding of district
                              (fixed category order = config.DISTRICT_NAMES, so a fold that happens
                              to contain only some districts still produces the same columns)
    Trees (LightGBM / XGBoost) are used with their native missing-value handling: the design matrix
    is never imputed, NaN is passed straight through.

Cross-validation
    assign_folds(df)   groups the TARGET rows by issue_date (so every horizon and district of one
    issue day stays on the same side of the split - zero temporal leakage) and cuts the chronologically
    sorted issue dates into config.N_CV_FOLDS blocks of approximately equal ROW count (not equal
    calendar time: ground-station coverage is heavily back-loaded to 2025+, see phase4 decisions).
    Each of the N folds is used once as the validation block (GroupKFold-by-time-block): trees are
    only ever trained on issue dates outside the validation block, but a block may sit before OR
    after the validation block in time (this is the "block holdout" the project plan calls for, not
    a strict walk-forward split). This still gives full out-of-fold coverage for evaluation while
    keeping every (issue_date, district, horizon) triple of a validation day out of its own training set.

Baseline
    persistence_prediction(df)   naive forecast: tomorrow (and the day after) will look like
    yesterday. Uses g_pm25_lag1 (the district's own PM2.5 mean of day D-1), the same value for every
    horizon of a given issue day - the benchmark every real model must beat.

AQI conversion & metrics
    to_aqi(pm25)              wraps phase4_ground/aqi.py so Phase 5 reuses the exact EPA 2024 formula
    concentration_metrics, aqi_metrics, evaluate   RMSE/MAE/R2 and AQI-MAE/accuracy/macro-F1
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "phase4_ground"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

import aqi
import config

CATEGORY_LABELS = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------
DATASET_FILE = config.FEATURE_DIR / "phase4_dataset.parquet"


def load_dataset():
    return pd.read_parquet(DATASET_FILE)


def target_rows(df=None):
    """Rows with a real PM2.5 ground target, the only rows Phase 5 trains or validates on."""
    if df is None:
        df = load_dataset()
    return df[df["pm25_target"].notna()].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Cross-validation blocks
# ----------------------------------------------------------------------------
def assign_folds(df, n_folds=config.N_CV_FOLDS):
    """Return a Series (issue_date -> fold id 0..n_folds-1), one fold per unique issue date."""
    rows_per_date = df.groupby("issue_date").size().sort_index()
    cum = rows_per_date.cumsum()
    total = float(cum.iloc[-1])
    edges = np.array([total * k / n_folds for k in range(1, n_folds)])
    fold = np.searchsorted(edges, cum.to_numpy(), side="right")
    return pd.Series(fold, index=cum.index, name="cv_fold")


def fold_summary(df, fold_of_date):
    """One row per fold: date range, row count, districts present - so sparsity is visible up front."""
    d = df.assign(cv_fold=df["issue_date"].map(fold_of_date))
    rows = d.groupby("cv_fold").agg(issue_dates=("issue_date", "nunique"), rows=("issue_date", "size"),
                                     first=("issue_date", "min"), last=("issue_date", "max"))
    districts = d.groupby("cv_fold")["district"].agg(lambda s: sorted(s.unique()))
    rows["districts_present"] = districts
    return rows


# ----------------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------------
def feature_columns(df):
    non_feat = set(config.NON_FEATURE_COLS)
    return [c for c in df.columns if c not in non_feat]


def build_design_matrix(df):
    """(X, feature_names) - base feature columns + a fixed 6-column district one-hot."""
    cols = feature_columns(df)
    x = df[cols].reset_index(drop=True).copy()
    dist = pd.Categorical(df["district"].to_numpy(), categories=config.DISTRICT_NAMES)
    dummies = pd.get_dummies(dist, prefix="district").astype("int8")
    dummies.index = x.index
    x = pd.concat([x, dummies], axis=1)
    # object columns would be a stray non-numeric leak; everything else (float/int/bool) is cast to a
    # single plain float64 block. A mixed-dtype DataFrame (float64 + int8 + bool columns, as produced
    # above) crashes LightGBM's C Dataset construction on Windows (access violation deep in
    # LGBM_DatasetSetField) - one uniform float64 array avoids it and is exactly as informative for trees.
    bad = [c for c in x.columns if x[c].dtype == object]
    if bad:
        raise TypeError(f"non-numeric feature columns leaked into the design matrix: {bad}")
    x = x.astype("float64")
    return x, list(x.columns)


# ----------------------------------------------------------------------------
# Persistence baseline
# ----------------------------------------------------------------------------
def persistence_prediction(df):
    """Naive forecast for every horizon: yesterday's district PM2.5 mean (g_pm25_lag1), clamped >= 0."""
    return df["g_pm25_lag1"].clip(lower=0)


# ----------------------------------------------------------------------------
# AQI conversion (reuses phase4_ground/aqi.py - the exact EPA 2024 piecewise formula)
# ----------------------------------------------------------------------------
def to_aqi(pm25):
    pm25 = np.clip(np.asarray(pm25, dtype=float), 0, None)
    return aqi.sub_index("pm25", pm25)


def to_category(aqi_values):
    return aqi.category(aqi_values).astype(str)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def concentration_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")}


def aqi_metrics(y_true_conc, y_pred_conc):
    a_true, a_pred = to_aqi(y_true_conc), to_aqi(y_pred_conc)
    c_true, c_pred = to_category(a_true), to_category(a_pred)
    return {"aqi_mae": float(mean_absolute_error(a_true, a_pred)),
            "aqi_category_accuracy": float((c_true == c_pred).mean()),
            "aqi_category_f1_macro": float(f1_score(c_true, c_pred, labels=CATEGORY_LABELS, average="macro", zero_division=0))}


def evaluate(y_true_conc, y_pred_conc):
    out = concentration_metrics(y_true_conc, y_pred_conc)
    out.update(aqi_metrics(y_true_conc, y_pred_conc))
    out["n"] = int(len(y_true_conc))
    return out
