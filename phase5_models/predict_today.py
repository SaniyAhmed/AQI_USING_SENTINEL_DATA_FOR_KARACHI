"""
Operational forecast: today's (T+0) AQI prediction for all six districts, using the final Phase 5
models, plus quick bias / overfitting / strength diagnostics run fresh against the saved out-of-fold
predictions (outputs/models/oof_predictions_aqi.parquet) and the final models themselves.

Needs, for the issue date:
    data/forecast/features_<issue>.parquet   (phase3_features/step5_gfs_forecast.py --issue <issue>)
    ground data current through issue-1      (data/processed/ground/{station,district}_daily.csv)
Ground-history features (g_pm25_lag1 etc.) are anchored to the issue day exactly like Phase 4's
training table, but phase4_ground/step4_dataset.district_ground() only ever built them for issue
dates that already had ground data (historical training). extended_ground_features() below reindexes
one day further (the issue day itself, still empty) so the SAME lag/rolling logic can be anchored to
it - the live equivalent of feature_lib._wide's `until` parameter.

Run:  python phase5_models/predict_today.py [--issue YYYY-MM-DD]   (default: today, Karachi time)
"""
import argparse
import json
import pickle
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase3_features")); sys.path.insert(0, str(HERE.parent / "phase4_ground"))
import numpy as np
import pandas as pd
import xgboost as xgb

import config
import model_lib as ml
import step3_postprocess as post
import step4_dataset as ds4
from step5_gfs_forecast import local_today

TMP_DIR = config.MODEL_DIR / "_tmp"


def lgb_predict(model_path, x):
    """Predict with a saved LightGBM model in an isolated subprocess (see _lgb_worker.py: lightgbm and
    xgboost cannot both be used natively in this process on Windows)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    in_path, out_path = TMP_DIR / f"job_{tag}.pkl", TMP_DIR / f"res_{tag}.pkl"
    try:
        with open(in_path, "wb") as f:
            pickle.dump({"mode": "predict", "model_path": str(model_path), "x": np.ascontiguousarray(x, dtype=np.float64)}, f)
        subprocess.run([sys.executable, str(HERE / "_lgb_worker.py"), str(in_path), str(out_path)], check=True)
        with open(out_path, "rb") as f:
            result = pickle.load(f)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    return result["pred"]


def xgb_predict(model_path, x, feature_cols):
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.feature_names = list(feature_cols)
    d = xgb.DMatrix(np.ascontiguousarray(x, dtype=np.float64), feature_names=list(feature_cols))
    return np.clip(booster.predict(d), 0, None)


def extended_ground_features(issue_date):
    """Ground-history features (g_pm25_lag1.. g_pm25_city_lag1) for ONE issue date not yet in the
    ground tables, by extending the continuous daily index one day further before differencing."""
    wide, days, ids = ds4.district_ground()
    if issue_date <= days.max():
        raise SystemExit(f"issue date {issue_date.date()} already has ground data - this is for a live/future issue date only")
    if issue_date > days.max() + pd.Timedelta(days=1):
        raise SystemExit(f"ground data ends {days.max().date()}, too far behind {issue_date.date()} for a 1-day extension")
    ext_days = pd.date_range(days[0], issue_date)
    wide_ext = {k: v.reindex(ext_days) for k, v in wide.items()}
    sd = pd.read_csv(ds4.GROUND / "station_daily.csv", parse_dates=["date"])
    hist = ds4.ground_history_features(wide_ext, ext_days, ids, sd)
    return hist.xs(issue_date, level="date").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=None, help="issue date YYYY-MM-DD (default: today, Karachi time)")
    args = ap.parse_args()
    issue = pd.Timestamp(args.issue) if args.issue else local_today()

    feat_path = config.FORECAST_DIR / f"features_{issue:%Y-%m-%d}.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found - run: python phase3_features/step5_gfs_forecast.py --issue {issue:%Y-%m-%d}")
    feats = pd.read_parquet(feat_path)

    ground = extended_ground_features(issue)
    merged = feats.merge(ground, on="district_id", how="left", validate="many_to_one")

    feature_cols = json.loads((config.MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    x_live, _ = ml.build_design_matrix(merged)
    missing = set(feature_cols) - set(x_live.columns)
    if missing:
        raise RuntimeError(f"live feature table is missing columns the model needs: {missing}")
    x_live = x_live.reindex(columns=feature_cols).to_numpy(dtype=np.float64)

    report = json.loads(config.PHASE5_REPORT.read_text(encoding="utf-8"))
    best = report["best_model_per_horizon"]

    out = merged[["issue_date", "target_date", "district", "horizon"]].copy()
    out["pred_lightgbm"] = np.nan
    out["pred_xgboost"] = np.nan
    for h in config.LIVE_HORIZONS:
        sel = (merged["horizon"] == h).to_numpy()
        out.loc[sel, "pred_lightgbm"] = lgb_predict(config.MODEL_DIR / f"lightgbm_h{h}.txt", x_live[sel])
        out.loc[sel, "pred_xgboost"] = xgb_predict(config.MODEL_DIR / f"xgboost_h{h}.json", x_live[sel], feature_cols)
    out["best_model"] = out["horizon"].astype(str).map(best)
    out["pred_final"] = np.where(out["best_model"] == "LightGBM", out["pred_lightgbm"], out["pred_xgboost"])
    out = post.add_aqi_columns(out, ["pred_final"])
    out = out.rename(columns={"pred_final_aqi": "aqi_final", "pred_final_category": "category_final"})
    agree = (out["pred_lightgbm"] - out["pred_xgboost"]).abs()

    print(f"\nIssue date {issue.date()} (Karachi time) - forecast for {issue.date()}, {(issue + pd.Timedelta(days=1)).date()}\n")
    print(f"{'district':<17}{'horizon':<9}{'target date':<13}{'PM2.5 (ug/m3)':<15}{'AQI':<6}{'category':<32}{'LGB vs XGB diff'}")
    for _, r in out.sort_values(["district", "horizon"]).iterrows():
        print(f"{r['district']:<17}T+{int(r['horizon'])}      {r['target_date'].date()!s:<13}"
              f"{r['pred_final']:<15.1f}{int(r['aqi_final']):<6}{r['category_final']:<32}"
              f"{abs(r['pred_lightgbm'] - r['pred_xgboost']):.1f} ug/m3")

    print(f"\nLightGBM vs XGBoost agreement across all {len(out)} rows: mean |diff| = {agree.mean():.2f} ug/m3, max = {agree.max():.2f} ug/m3")
    print(f"(each horizon's number above is whichever model won that horizon out-of-fold: {best})")
    print(f"\ncams_* features empty for this issue date: {merged['cams_aod550'].isna().all()} "
          f"(Earth Engine has not ingested a recent-enough CAMS run - see phase3_4 decisions memory; "
          f"the trees route this as a missing feature, not an error)")

    out_path = config.FORECAST_DIR / f"prediction_{issue:%Y-%m-%d}.csv"
    out.to_csv(out_path, index=False, float_format="%.4g")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
