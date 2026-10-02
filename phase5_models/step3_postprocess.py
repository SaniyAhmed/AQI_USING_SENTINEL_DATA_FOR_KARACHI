"""
PHASE 5 - STEP 3: Post-processing - PM2.5 concentration -> official EPA 2024 AQI sub-index.

    I_p = (I_high - I_low) / (C_high - C_low) * (C_p - C_low) + I_low

The exact breakpoint table and truncation rule already live in phase4_ground/aqi.py (built and
validated in Phase 4 - see its 14 hand-computed check cases and monotonicity test in
phase4_ground/step5_validate.py). Step 5 reuses that table rather than duplicating it, so the
target AQI (aqi_pm25) and every predicted AQI go through the exact same formula. This module just
adds the two things Phase 5 needs on top: clamping predictions at 0 before conversion (a regression
model can output a small negative number; concentration cannot be negative) and the 6 official bands.

    pm25_to_aqi(pm25)   clamp at 0, truncate to 0.1 ug/m3, EPA 2024 piecewise sub-index (0-500)
    categorize(aqi)     Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy / Very Unhealthy / Hazardous
    add_aqi_columns(df, pred_cols)   for every column name in pred_cols, adds "<col>_aqi" and "<col>_category"

Run standalone to sanity-check the conversion and attach AQI columns to step2's out-of-fold
predictions:  python phase5_models/step3_postprocess.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

import config
import model_lib as ml
from cache_io import atomic_write

BANDS = [(0, 50, "Good"), (51, 100, "Moderate"), (101, 150, "Unhealthy for Sensitive Groups"),
         (151, 200, "Unhealthy"), (201, 300, "Very Unhealthy"), (301, 500, "Hazardous")]


def pm25_to_aqi(pm25):
    """Vectorised EPA 2024 PM2.5 sub-index. Negative concentrations (a regression artefact) are
    clamped to 0 ug/m3 before conversion; aqi.sub_index truncates to 0.1 ug/m3 and rounds the index."""
    return ml.to_aqi(pm25)


def categorize(aqi_values):
    """Official 6-band category of an AQI value (NaN stays NaN)."""
    return ml.to_category(aqi_values)


def add_aqi_columns(df, pred_cols):
    out = df.copy()
    for col in pred_cols:
        out[f"{col}_aqi"] = pm25_to_aqi(out[col])
        out[f"{col}_category"] = categorize(out[f"{col}_aqi"])
    return out


def _self_check():
    """The 6-band boundaries must match EPA exactly; used as a quick smoke test, not the full gate
    (the exhaustive checks - monotonicity, 14 hand cases, continuity at bin edges - live in
    phase4_ground/step5_validate.py and already ran in Phase 4)."""
    probe = np.array([-5.0, 0.0, 9.0, 9.1, 35.4, 35.5, 55.4, 55.5, 125.4, 125.5, 225.4, 225.5, 400.0])
    expect = [0, 0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500]
    got = pm25_to_aqi(probe)
    ok = np.array_equal(np.asarray(expect), got.astype(int)) if len(expect) == len(got) else False
    print(f"self-check (clamp + EPA 2024 bins): {'ok' if ok else 'MISMATCH'} -> {got.tolist()}")
    return ok


def main():
    if not _self_check():
        raise SystemExit("PM2.5 -> AQI conversion failed its self-check - do not trust downstream metrics")

    oof_path = config.MODEL_DIR / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"\n{oof_path} not found yet - run step2_train_models.py first. Self-check only, nothing else to do.")
        return
    oof = pd.read_parquet(oof_path)
    pred_cols = ["pm25_target", "pred_persistence", "pred_lightgbm", "pred_xgboost"]
    withaqi = add_aqi_columns(oof, pred_cols)
    atomic_write(config.MODEL_DIR / "oof_predictions_aqi.parquet", lambda p: withaqi.to_parquet(p, index=False))
    print(f"\nAQI category share of the true target (all out-of-fold target rows):")
    print(withaqi["pm25_target_category"].value_counts(normalize=True).round(3).to_string())
    print(f"\nSaved {config.MODEL_DIR / 'oof_predictions_aqi.parquet'}")


if __name__ == "__main__":
    main()
