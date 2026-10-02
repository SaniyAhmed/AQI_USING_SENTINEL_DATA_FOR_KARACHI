"""
PHASE 5 - STEP 4: Evaluation metrics, feature importance, and the Phase 5 validation gate.

Reads step2's out-of-fold predictions (every target row scored by a model that never saw it during
training) and step3's AQI conversion, then:
  1. reports concentration metrics (RMSE, MAE, R2) and AQI metrics (AQI MAE, category accuracy,
     macro F1) per horizon and overall, for LightGBM, XGBoost and the persistence baseline;
  2. reports gain-based feature importance (from the final, all-data models) per horizon and
     framework, plus one combined ranking, to eyeball physical plausibility (AOD, PBLH, humidity,
     the district's own recent PM2.5 should rank high - not an arbitrary satellite band);
  3. runs the Phase 5 validation gate (model coverage, zero target leakage, beats the baseline)
     and, if it passes, writes the final predictions table and confirms the final models on disk.

Writes:
    outputs/phase5_validation_report.json
    data/processed/predictions.parquet   issue_date, target_date, district, horizon, pm25_target,
        aqi_pm25 (true), n_stations_target, cv_fold, every model's prediction + AQI + category,
        and best_model / pred_final / aqi_final / category_final (the better of LightGBM/XGBoost per horizon)

Run:  python phase5_models/step4_evaluate_and_gate.py   (needs step1 and step2 to have run first)
"""
import datetime as dt
import json
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
import step3_postprocess as post
from cache_io import atomic_write

RESULTS = []
MODELS = ["pred_persistence", "pred_lightgbm", "pred_xgboost"]
MODEL_LABEL = {"pred_persistence": "Persistence baseline", "pred_lightgbm": "LightGBM", "pred_xgboost": "XGBoost"}


def check(name, ok, detail="", level="FAIL"):
    status = "PASS" if ok else level
    RESULTS.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{ {'PASS': '  ok ', 'WARN': ' WARN', 'FAIL': ' FAIL'}[status] }] {name}" + (f" - {detail}" if detail and status != "PASS" else ""))
    return ok


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_table(df):
    """{ (horizon or 'overall', model_col): metrics dict }. pred_persistence is NaN on the handful of
    rows where a district has no ground value for D-1 yet (its very first monitored days - g_pm25_lag1
    cannot exist before there is a day before it); those rows are skipped for the persistence column
    only (n in the result says how many), never for LightGBM/XGBoost, which always produce a number."""
    rows = {}
    for h in list(config.HORIZONS) + ["overall"]:
        sub = df if h == "overall" else df[df["horizon"] == h]
        for col in MODELS:
            valid = sub[sub[col].notna()]
            rows[(h, col)] = ml.evaluate(valid["pm25_target"], valid[col])
    return rows


def print_metrics_table(rows):
    header = f"{'horizon':>9} | {'model':>21} | {'RMSE':>7} {'MAE':>7} {'R2':>7} | {'AQI MAE':>8} {'AQI acc':>8} {'AQI F1':>7} | n"
    print(header); print("-" * len(header))
    for h in list(config.HORIZONS) + ["overall"]:
        for col in MODELS:
            m = rows[(h, col)]
            print(f"{('T+' + str(h)) if h != 'overall' else 'overall':>9} | {MODEL_LABEL[col]:>21} | "
                  f"{m['rmse']:7.2f} {m['mae']:7.2f} {m['r2']:7.3f} | {m['aqi_mae']:8.2f} "
                  f"{m['aqi_category_accuracy']:8.1%} {m['aqi_category_f1_macro']:7.3f} | {m['n']}")


def per_district_metrics(df):
    out = {}
    for d in config.DISTRICT_NAMES:
        sub = df[df["district"] == d]
        if len(sub) < 10:
            continue
        out[d] = {MODEL_LABEL[col]: ml.evaluate(sub.loc[sub[col].notna(), "pm25_target"], sub.loc[sub[col].notna(), col]) for col in MODELS}
    return out


# ----------------------------------------------------------------------------
# Feature importance (gain-based, from the final all-data models saved by step2)
# ----------------------------------------------------------------------------
def lightgbm_importance_via_subprocess(model_path):
    """Reads a saved LightGBM model's gain importance in an isolated subprocess (see _lgb_worker.py
    docstring: lightgbm and xgboost cannot both be imported in this process on Windows)."""
    tmp_dir = config.MODEL_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    in_path, out_path = tmp_dir / f"job_{tag}.pkl", tmp_dir / f"res_{tag}.pkl"
    try:
        with open(in_path, "wb") as f:
            pickle.dump({"mode": "importance", "model_path": str(model_path)}, f)
        subprocess.run([sys.executable, str(HERE / "_lgb_worker.py"), str(in_path), str(out_path)], check=True)
        with open(out_path, "rb") as f:
            result = pickle.load(f)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
    return result["names"], np.array(result["gain"], dtype=float)


def load_importance(framework, h, feature_cols):
    if framework == "lightgbm":
        names, gain = lightgbm_importance_via_subprocess(config.MODEL_DIR / f"lightgbm_h{h}.txt")
    else:
        booster = xgb.Booster()
        booster.load_model(str(config.MODEL_DIR / f"xgboost_h{h}.json"))
        score = booster.get_score(importance_type="gain")
        names_in_order = booster.feature_names or feature_cols
        gain = np.array([score.get(n, 0.0) for n in names_in_order])
        names = names_in_order
    s = pd.Series(gain, index=names, dtype=float)
    return s / s.sum() if s.sum() > 0 else s


def feature_importance_report(feature_cols):
    combined = pd.Series(0.0, index=feature_cols)
    per_model = {}
    n_models = 0
    for framework in ("lightgbm", "xgboost"):
        for h in config.HORIZONS:
            imp = load_importance(framework, h, feature_cols).reindex(feature_cols).fillna(0.0)
            per_model[(framework, h)] = imp
            combined = combined.add(imp, fill_value=0.0)
            n_models += 1
    combined /= n_models
    return combined.sort_values(ascending=False), per_model


PHYSICAL_KEYWORDS = ("aod", "blh", "vent", "rh_", "humid", "g_pm25", "g_aqi", "cams_pm25", "no2", "inversion")


# ----------------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------------
def run_gate(df, rows, feature_cols):
    n_expected = len(ml.target_rows())
    # pred_persistence is legitimately NaN on the first day(s) of a district's ground history (no D-1
    # value exists yet, see metrics_table); only the two trained models must be complete everywhere.
    check(f"coverage: out-of-fold predictions exist for all {n_expected} target rows, every horizon",
          len(df) == n_expected and not df[["pred_lightgbm", "pred_xgboost"]].isna().any().any(), f"got {len(df)} rows")

    leak_cols = set(feature_cols) & set(config.NON_FEATURE_COLS)
    check("zero leakage: no target or identifier column is among the model features", not leak_cols, leak_cols)
    assign = pd.read_parquet(config.MODEL_DIR / "cv_assignment.parquet")
    check("zero leakage: every target row has exactly one fold assignment (no duplicate issue_date/district/horizon)",
          not assign.duplicated(["issue_date", "district", "horizon"]).any() and len(assign) == n_expected)

    overall = rows[("overall", "pred_persistence")]
    for col in ("pred_lightgbm", "pred_xgboost"):
        m = rows[("overall", col)]
        check(f"{MODEL_LABEL[col]} beats the persistence baseline overall (RMSE {m['rmse']:.2f} vs {overall['rmse']:.2f}, "
              f"MAE {m['mae']:.2f} vs {overall['mae']:.2f})", m["rmse"] < overall["rmse"] and m["mae"] < overall["mae"])
        per_h_wins = sum(rows[(h, col)]["rmse"] < rows[(h, "pred_persistence")]["rmse"] for h in config.HORIZONS)
        check(f"{MODEL_LABEL[col]} beats persistence on RMSE in every individual horizon ({per_h_wins}/3)",
              per_h_wins == len(config.HORIZONS), f"{per_h_wins}/3 horizons", "WARN")

    combined, _ = feature_importance_report(feature_cols)
    top15 = combined.head(15)
    physical_hits = [f for f in top15.index if any(k in f.lower() for k in PHYSICAL_KEYWORDS)]
    check(f"feature importance: physically-expected drivers (AOD/PBLH/ventilation/humidity/recent ground PM2.5/CAMS) "
          f"appear in the top 15: {physical_hits}", len(physical_hits) >= 3, list(top15.index), "WARN")
    return combined


def main():
    print("=" * 78 + "\n PHASE 5 EVALUATION AND VALIDATION GATE\n" + "=" * 78)
    oof = pd.read_parquet(config.MODEL_DIR / "oof_predictions_aqi.parquet") if (config.MODEL_DIR / "oof_predictions_aqi.parquet").exists() \
        else post.add_aqi_columns(pd.read_parquet(config.MODEL_DIR / "oof_predictions.parquet"),
                                   ["pm25_target", "pred_persistence", "pred_lightgbm", "pred_xgboost"])
    feature_cols = json.loads((config.MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))

    rows = metrics_table(oof)
    print("\nOut-of-fold performance (concentration + AQI metrics):\n")
    print_metrics_table(rows)

    print("\nPer-district overall metrics (fold 0, South-only history means early years are South-only - see step1):")
    for d, models in per_district_metrics(oof).items():
        best = min(models.items(), key=lambda kv: kv[1]["rmse"])
        print(f"  {d} (n={best[1]['n']}): " + ", ".join(f"{k} RMSE={v['rmse']:.1f}" for k, v in models.items()))

    print("\nFeature importance (gain-based, final all-data models, averaged over 2 frameworks x 3 horizons):")
    combined = run_gate(oof, rows, feature_cols)
    print(combined.head(20).round(4).to_string())

    # ------------------------------------------------------------------
    # final predictions table: for each horizon, "final" = whichever of LightGBM/XGBoost has the
    # lower out-of-fold RMSE for that horizon (a per-horizon model choice, made honestly from OOF data)
    # ------------------------------------------------------------------
    best_by_horizon = {h: ("pred_lightgbm" if rows[(h, "pred_lightgbm")]["rmse"] <= rows[(h, "pred_xgboost")]["rmse"] else "pred_xgboost")
                        for h in config.HORIZONS}
    print(f"\nBest model per horizon (lower OOF RMSE): {{{', '.join(f'T+{h}: {MODEL_LABEL[c]}' for h, c in best_by_horizon.items())}}}")
    final = oof.copy()
    final["target_date"] = final["issue_date"] + pd.to_timedelta(final["horizon"], unit="D")
    final["best_model"] = final["horizon"].map(best_by_horizon).map(MODEL_LABEL)
    final["pred_final"] = np.select([final["horizon"] == h for h in best_by_horizon], [final[c] for c in best_by_horizon.values()])
    final = post.add_aqi_columns(final, ["pred_final"])
    final = final.rename(columns={"pred_final_aqi": "aqi_final", "pred_final_category": "category_final"})
    lead = ["issue_date", "target_date", "district", "horizon", "cv_fold", "pm25_target", "pm25_target_aqi", "n_stations_target"]
    final = final[lead + [c for c in final.columns if c not in lead]]

    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    if n_fail == 0:
        atomic_write(config.PREDICTIONS_FILE.with_suffix(".parquet"), lambda p: final.to_parquet(p, index=False))
        print(f"\nSaved {config.PREDICTIONS_FILE.with_suffix('.parquet')} ({len(final)} rows)")
        print(f"Final models already on disk in {config.MODEL_DIR} (lightgbm_h*.txt, xgboost_h*.json).")
    else:
        print(f"\n{n_fail} gate check(s) FAILED - predictions.parquet NOT written. Fix the failure(s) above and re-run.")

    config.PHASE5_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "failures": n_fail, "warnings": n_warn,
              "checks": RESULTS, "metrics": {f"{h}|{col}": rows[(h, col)] for h in list(config.HORIZONS) + ["overall"] for col in MODELS},
              "top_features": combined.head(20).round(4).to_dict(), "best_model_per_horizon": {str(h): MODEL_LABEL[c] for h, c in best_by_horizon.items()}}
    config.PHASE5_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=" * 78 + f"\n {len(RESULTS)} checks: {len(RESULTS) - n_fail - n_warn} passed, {n_warn} warnings, {n_fail} FAILED\n"
          f" report: {config.PHASE5_REPORT}\n" + "=" * 78)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
