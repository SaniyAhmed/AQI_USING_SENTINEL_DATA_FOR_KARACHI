"""
Runs all Phase 5 steps and ends with the validation gate.

    python phase5_models/run_phase5.py

  1. step1_cross_validation   time-block CV folds + the feature set
  2. step2_train_models       LightGBM + XGBoost, direct multi-horizon, out-of-fold predictions
  3. step3_postprocess        PM2.5 -> EPA 2024 AQI (self-check + attaches AQI columns to the OOF table)
  4. step4_evaluate_and_gate  metrics, feature importance, the gate: exit code 1 if ANY check FAILS

Needs data/processed/features/phase4_dataset.parquet (Phase 4 must have run and passed its gate first).
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from cache_io import clean_stale_tmp, pipeline_lock
import config

STEPS = ["step1_cross_validation.py", "step2_train_models.py", "step3_postprocess.py", "step4_evaluate_and_gate.py"]


def main():
    clean_stale_tmp(config.MODEL_DIR)
    for step in STEPS:
        print("\n" + "=" * 70 + f"\n  {step}\n" + "=" * 70)
        sys.argv = [step]
        runpy.run_path(str(HERE / step), run_name="__main__")
    print("\nPhase 5 complete and validated.")


if __name__ == "__main__":
    with pipeline_lock():
        main()
