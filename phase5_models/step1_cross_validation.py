"""
PHASE 5 - STEP 1: Cross-validation blocks and the feature set.

Loads data/processed/features/phase4_dataset.parquet, keeps only rows with a real PM2.5 ground
target (pm25_target notna - 7,401 of 30,870 rows, see phase4 decisions), and cuts them into
config.N_CV_FOLDS (5) time blocks grouped by issue_date, so every horizon (0/1/2) and district of a
given issue day stays on the same side of a split (zero temporal leakage, see model_lib.assign_folds
for why these are approximately-equal-ROW blocks rather than equal-calendar-time blocks: ground
station coverage outside Karachi South only starts in 2025, see phase3_4 decisions).

Ground truth is heavily back-loaded (South: 2022-01 on; East: 2024-10; Central/West: 2025-06;
Korangi/Malir: 2025-11), so the EARLY folds are mostly/only Karachi South and the LATE folds are the
only ones with all six districts. That is a real property of the data, not a bug - printed below so
it is visible before any model is trained, and referenced again by step4's per-district metrics.

Writes:
    outputs/models/cv_assignment.parquet   (issue_date, district, horizon, cv_fold) for every target row
    outputs/models/feature_columns.json    the exact feature list (incl. one-hot district) steps 2-4 must reuse

Run:  python phase5_models/step1_cross_validation.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import pandas as pd

import config
import model_lib as ml
from cache_io import atomic_write


def main():
    df = ml.load_dataset()
    t = ml.target_rows(df)
    print(f"Phase 4 dataset: {len(df)} rows. Rows with a PM2.5 ground target: {len(t)} ({len(t) / len(df):.1%}).")

    fold_of_date = ml.assign_folds(t)
    summary = ml.fold_summary(t, fold_of_date)
    print(f"\n{config.N_CV_FOLDS} time-block folds (grouped by issue_date, ~equal row count):")
    for k, row in summary.iterrows():
        print(f"  fold {k}: {row['first'].date()} .. {row['last'].date()}  "
              f"{row['issue_dates']} issue dates, {row['rows']} rows  -  districts: {', '.join(row['districts_present'])}")
    only_south = [k for k, row in summary.iterrows() if row["districts_present"] == ["Karachi South"]]
    if only_south:
        print(f"\n  NOTE: fold(s) {only_south} contain Karachi South only - no other district has ground "
              f"truth that far back. Their out-of-fold metrics (step 4) describe South alone.")

    t = t.assign(cv_fold=t["issue_date"].map(fold_of_date))
    x, feature_cols = ml.build_design_matrix(t)
    print(f"\nDesign matrix: {len(feature_cols)} feature columns (incl. {len(config.DISTRICT_NAMES)} one-hot district columns).")
    empty = x.isna().all()
    if empty.any():
        print(f"  WARNING: columns entirely NaN in the target rows (still passed to the trees as missing): {list(x.columns[empty])}")

    assignment = t[["issue_date", "district", "horizon", "cv_fold"]].copy()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(config.MODEL_DIR / "cv_assignment.parquet", lambda p: assignment.to_parquet(p, index=False))
    atomic_write(config.MODEL_DIR / "feature_columns.json",
                 lambda p: p.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8"))
    print(f"\nSaved {config.MODEL_DIR / 'cv_assignment.parquet'}")
    print(f"Saved {config.MODEL_DIR / 'feature_columns.json'}")


if __name__ == "__main__":
    main()
