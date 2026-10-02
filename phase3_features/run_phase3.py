"""
Runs all Phase 3 steps and ends with the validation gate.

    python phase3_features/run_phase3.py              (START_DATE in config.py -> newest ERA5 day)

  1. step1_check_sources  ERA5 / CAMS lattices line up with ours (exact)
  2. step2_era5           ERA5 local-day meteorology at the six centroids
  3. step3_gfs_t850       850 hPa temperature (archived GFS)
  4. step4_cams           CAMS 12Z-run aerosol (leak-free timestamps)
  5. step6_build_features the training table: one row per (issue date, district, horizon)
  6. step7_validate       the gate: exit code 1 if ANY check fails

Every morning, for a live forecast, run step5_gfs_forecast.py (after run_phase2.py has brought the
satellite table up to yesterday).  Phase 2's run_phase2.py must have been run first.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase2_ingestion"))
from cache_io import clean_stale_tmp, pipeline_lock
import config

STEPS = ["step1_check_sources.py", "step2_era5.py", "step3_gfs_t850.py", "step4_cams.py",
         "step6_build_features.py", "step7_validate.py"]


def main():
    clean_stale_tmp(config.DATA_DIR)
    for step in STEPS:
        print("\n" + "=" * 70 + f"\n  {step}\n" + "=" * 70)
        sys.argv = [step]
        runpy.run_path(str(HERE / step), run_name="__main__")
    print("\nPhase 3 complete and validated.")


if __name__ == "__main__":
    with pipeline_lock():
        main()
