"""
PHASE 5 - STEP 2: Model training architecture (LightGBM + XGBoost, direct multi-horizon).

For each horizon h in (0, 1, 2), and for each of the 5 time-block folds from step1:
    train on the other 4 folds, predict the held-out fold (LightGBM and XGBoost), so every
    prediction is out-of-fold (no row is ever scored by a model that saw it during training).
Sample weight = n_stations_target directly (denser ground networks pull the loss harder - see
phase3_4 decisions: 52% of target rows rest on a single sensor). Missing values (e.g. live T+2 CAMS
features can be NaN, see phase3_4 decisions) are passed straight to the trees; LightGBM/XGBoost route
them natively, no imputation anywhere in this pipeline.

Overfitting control (added after the first pass showed in-sample RMSE ~5 ug/m3 vs out-of-fold
RMSE ~13-15 ug/m3 - a real, if unsurprising, gap for ~2,000-2,500 rows against 142 features):
EVERY fit (per fold and the final model) uses early stopping against an INNER, chronologically-later
slice of its own training portion (temporal_inner_split, last 15% of issue dates) - never the outer
CV fold being scored, so this adds no leakage, it just stops adding trees once they stop helping on
genuinely unseen-in-training days. The final, all-data model per horizon is then refit on every target
row at a FIXED round count (the median of the 5 folds' early-stopped counts) - this uses 100% of the
data for the deployed model while keeping the round count an honest, held-out-earned number rather
than a guess.

LightGBM fits run in a separate subprocess (_lgb_worker.py): importing lightgbm and xgboost in the
SAME process makes LightGBM's native Dataset construction intermittently crash on Windows
("OSError: access violation" inside LGBM_DatasetSetField - reproduced directly, a native
OpenMP/runtime DLL conflict between the two wheels, not a data problem). XGBoost fits stay in-process.

Writes:
    outputs/models/oof_predictions.parquet    issue_date, district, horizon, cv_fold, pm25_target,
                                               n_stations_target, pred_persistence, pred_lightgbm, pred_xgboost
    outputs/models/{lightgbm,xgboost}_h{h}.*  final (all-data) model per horizon per framework

Run:  python phase5_models/step2_train_models.py   (needs step1_cross_validation.py to have run first)
"""
import pickle
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import xgboost as xgb

import config
import model_lib as ml
from cache_io import atomic_write

# Conservative, regularised settings: ~2,000-2,500 training rows per horizon against 142 features
# means a high feature-to-sample ratio, so shallow trees + row/column subsampling matter more here
# than squeezing extra leaves out of num_leaves/max_depth. n_estimators is a CEILING - early stopping
# (below) picks the actual count. Both frameworks use the same effective capacity/regularisation so
# neither is handicapped in the comparison.
INNER_VAL_FRAC = 0.15     # share of each training portion's LATEST issue dates held out for early stopping
ES_ROUNDS = 50
MAX_ESTIMATORS = 2000

LGB_PARAMS = dict(n_estimators=MAX_ESTIMATORS, num_leaves=15, max_depth=4, learning_rate=0.03,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.6, min_child_samples=15,
                   reg_alpha=0.1, reg_lambda=1.0, random_state=config.RANDOM_SEED, verbosity=-1, n_jobs=1)
XGB_PARAMS = dict(n_estimators=MAX_ESTIMATORS, max_depth=4, learning_rate=0.03, subsample=0.7, colsample_bytree=0.6,
                   min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", missing=np.nan,
                   random_state=config.RANDOM_SEED, n_jobs=1)

TMP_DIR = config.MODEL_DIR / "_tmp"


def temporal_inner_split(dates, frac=INNER_VAL_FRAC):
    """Positions (into `dates`) of the earlier (1-frac) and the chronologically LATEST `frac` share -
    an early-stopping validation slice that is always "in the future" relative to its own inner-train,
    exactly like the real forecasting task, and entirely inside the caller's outer-training rows."""
    order = np.argsort(dates, kind="stable")
    cut = int(len(order) * (1 - frac))
    return order[:cut], order[cut:]


def fit_lightgbm(x_train, y_train, w_train, feature_cols, x_val=None, x_es=None, y_es=None, n_estimators=None, model_out=None):
    """Delegates to _lgb_worker.py in a fresh process (see module docstring). Returns (pred_val,
    best_iteration): pred_val is predictions on x_val (clipped >= 0) or None; best_iteration is the
    early-stopped round count, or None if x_es was not given."""
    params = dict(LGB_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    in_path, out_path = TMP_DIR / f"job_{tag}.pkl", TMP_DIR / f"res_{tag}.pkl"
    job = {"mode": "fit", "x_train": np.ascontiguousarray(x_train, dtype=np.float64), "y_train": y_train, "w_train": w_train,
           "x_val": np.ascontiguousarray(x_val, dtype=np.float64) if x_val is not None else None,
           "x_es": np.ascontiguousarray(x_es, dtype=np.float64) if x_es is not None else None,
           "y_es": y_es, "es_rounds": ES_ROUNDS,
           "feature_cols": list(feature_cols), "params": params, "model_out": str(model_out) if model_out else None}
    try:
        with open(in_path, "wb") as f:
            pickle.dump(job, f)
        subprocess.run([sys.executable, str(HERE / "_lgb_worker.py"), str(in_path), str(out_path)], check=True)
        with open(out_path, "rb") as f:
            result = pickle.load(f)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    return result["pred_val"], result["best_iteration"]


def fit_xgboost(x_train, y_train, w_train, feature_cols, x_es=None, y_es=None, n_estimators=None):
    """Returns (fitted_model, best_iteration) - best_iteration is None unless x_es/y_es were given."""
    params = dict(XGB_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    fit_kwargs = {}
    if x_es is not None:
        params["early_stopping_rounds"] = ES_ROUNDS
        params["eval_metric"] = "rmse"
        fit_kwargs["eval_set"] = [(np.ascontiguousarray(x_es, dtype=np.float64), y_es)]
        fit_kwargs["verbose"] = False
    m = xgb.XGBRegressor(**params)
    m.fit(np.ascontiguousarray(x_train, dtype=np.float64), y_train, sample_weight=w_train, **fit_kwargs)
    m.get_booster().feature_names = list(feature_cols)
    best_iteration = getattr(m, "best_iteration", None)
    return m, (int(best_iteration) if best_iteration is not None else None)


def run_horizon(t, h):
    sub = t[t["horizon"] == h].reset_index(drop=True)
    x_all, feature_cols = ml.build_design_matrix(sub)
    x_all = x_all.to_numpy(dtype=np.float64)
    y_all = sub["pm25_target"].to_numpy()
    w_all = sub[config.WEIGHT_COL].to_numpy()
    dates_all = sub["issue_date"].to_numpy()

    oof = sub[["issue_date", "district", "horizon", "cv_fold", "pm25_target", config.WEIGHT_COL]].copy()
    oof["pred_persistence"] = ml.persistence_prediction(sub).to_numpy()
    oof["pred_lightgbm"] = np.nan
    oof["pred_xgboost"] = np.nan
    lgb_rounds, xgb_rounds = [], []

    for f in sorted(sub["cv_fold"].unique()):
        train_mask = (sub["cv_fold"] != f).to_numpy()
        val_mask = ~train_mask
        train_idx = np.where(train_mask)[0]
        inner_train_pos, inner_val_pos = temporal_inner_split(dates_all[train_idx])
        it_idx, iv_idx = train_idx[inner_train_pos], train_idx[inner_val_pos]

        pred_l, best_l = fit_lightgbm(x_all[it_idx], y_all[it_idx], w_all[it_idx], feature_cols,
                                       x_val=x_all[val_mask], x_es=x_all[iv_idx], y_es=y_all[iv_idx])
        oof.loc[val_mask, "pred_lightgbm"] = pred_l
        lgb_rounds.append(best_l)

        xgb_model, best_x = fit_xgboost(x_all[it_idx], y_all[it_idx], w_all[it_idx], feature_cols,
                                         x_es=x_all[iv_idx], y_es=y_all[iv_idx])
        oof.loc[val_mask, "pred_xgboost"] = np.clip(xgb_model.predict(x_all[val_mask]), 0, None)
        xgb_rounds.append(best_x)
        print(f"  horizon T+{h} fold {f}: trained on {len(it_idx)} rows (+{len(iv_idx)} held out for early "
              f"stopping), validated on {int(val_mask.sum())} rows - best rounds: LightGBM {best_l}, XGBoost {best_x}")

    # Final (deployed) model: refit on ALL target rows at a FIXED round count - the median of the 5
    # folds' early-stopped counts, an honest held-out-earned number - so the production model uses
    # every row without needing its own held-out slice.
    final_lgb_rounds = int(np.median(lgb_rounds))
    final_xgb_rounds = int(np.median(xgb_rounds))
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fit_lightgbm(x_all, y_all, w_all, feature_cols, n_estimators=final_lgb_rounds,
                 model_out=config.MODEL_DIR / f"lightgbm_h{h}.txt")
    final_xgb, _ = fit_xgboost(x_all, y_all, w_all, feature_cols, n_estimators=final_xgb_rounds)
    atomic_write(config.MODEL_DIR / f"xgboost_h{h}.json", lambda p: final_xgb.save_model(str(p)))
    print(f"  horizon T+{h} final models (all {len(sub)} target rows): LightGBM {final_lgb_rounds} rounds, "
          f"XGBoost {final_xgb_rounds} rounds (median of the 5 folds' early-stopped counts: "
          f"LightGBM {lgb_rounds}, XGBoost {xgb_rounds})")
    return oof


def main():
    t = ml.target_rows()
    assign = pd.read_parquet(config.MODEL_DIR / "cv_assignment.parquet")
    t = t.merge(assign, on=["issue_date", "district", "horizon"], how="inner", validate="one_to_one")
    if len(t) != len(assign):
        raise RuntimeError("cross-validation assignment does not cover every target row - re-run step1_cross_validation.py")

    frames = []
    for h in config.HORIZONS:
        print(f"\n=== horizon T+{h} ({len(t[t['horizon'] == h])} target rows) ===")
        frames.append(run_horizon(t, h))
    oof = pd.concat(frames, ignore_index=True)
    if oof[["pred_lightgbm", "pred_xgboost"]].isna().any().any():
        raise RuntimeError("some rows never got an out-of-fold prediction")

    atomic_write(config.MODEL_DIR / "oof_predictions.parquet", lambda p: oof.to_parquet(p, index=False))
    print(f"\nSaved {config.MODEL_DIR / 'oof_predictions.parquet'} ({len(oof)} rows)")


if __name__ == "__main__":
    main()
