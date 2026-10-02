"""
_lgb_worker.py - runs ONE LightGBM operation (fit, or read back a saved model's importance) in its
own process.

Why a subprocess: importing lightgbm and xgboost in the SAME Windows process makes LightGBM's native
calls intermittently crash with "OSError: access violation reading 0x0" (reproduced directly for both
Dataset construction during fit AND simply reading a saved model's feature names back - a native
OpenMP/runtime DLL conflict between the two wheels, not a data problem). step2/step4 need both
frameworks, so every LightGBM operation is delegated here, in a fresh interpreter that never imports
xgboost.

Protocol: argv = [input_pickle_path, output_pickle_path].
    input pickle (fit):        {"mode": "fit", "x_train", "y_train", "w_train", "x_val" (or None),
                                 "feature_cols", "params", "model_out" (or None),
                                 "x_es"/"y_es" (or None) - inner early-stopping validation set,
                                 "es_rounds" (default 50)}
    input pickle (importance): {"mode": "importance", "model_path"}
    input pickle (predict):    {"mode": "predict", "model_path", "x"}
    output pickle (fit):        {"pred_val": array or None, "best_iteration": int or None}
    output pickle (importance): {"names": [...], "gain": [...]}
    output pickle (predict):    {"pred": array}
"""
import pickle
import sys

import numpy as np
import lightgbm as lgb


def do_fit(job):
    m = lgb.LGBMRegressor(**job["params"])
    fit_kwargs = {}
    if job.get("x_es") is not None:
        # Early stopping on an inner, chronologically-later slice of the TRAINING portion only (never
        # the outer CV validation fold) - caps how many trees are grown so the model does not memorise
        # the training set (in-sample RMSE was ~5 ug/m3 vs ~13-15 ug/m3 out-of-fold before this).
        fit_kwargs["eval_X"] = job["x_es"]
        fit_kwargs["eval_y"] = job["y_es"]
        fit_kwargs["eval_metric"] = "rmse"
        fit_kwargs["callbacks"] = [lgb.early_stopping(job.get("es_rounds", 50), verbose=False)]
    m.fit(job["x_train"], job["y_train"], sample_weight=job["w_train"], feature_name=job["feature_cols"], **fit_kwargs)
    pred_val = None
    if job.get("x_val") is not None:
        pred_val = np.clip(m.predict(job["x_val"]), 0, None)
    if job.get("model_out"):
        m.booster_.save_model(job["model_out"])
    best_iteration = int(m.best_iteration_) if getattr(m, "best_iteration_", None) else None
    return {"pred_val": pred_val, "best_iteration": best_iteration}


def do_importance(job):
    booster = lgb.Booster(model_file=job["model_path"])
    return {"names": booster.feature_name(), "gain": booster.feature_importance(importance_type="gain").tolist()}


def do_predict(job):
    booster = lgb.Booster(model_file=job["model_path"])
    pred = np.clip(booster.predict(np.ascontiguousarray(job["x"], dtype=np.float64)), 0, None)
    return {"pred": pred}


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        job = pickle.load(f)
    mode = job.get("mode")
    if mode == "importance":
        result = do_importance(job)
    elif mode == "predict":
        result = do_predict(job)
    else:
        result = do_fit(job)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
