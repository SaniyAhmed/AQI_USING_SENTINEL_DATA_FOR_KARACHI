"""
Runs all Phase 1 steps in order. Run:
    python phase1_geometry/run_phase1.py
"""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "step1_download_boundaries.py",
    "step2_prepare_districts.py",
    "step3_build_grid_and_masks.py",
    "step4_test_zonal_stats.py",
    "step5_visualize.py",
]

for step in STEPS:
    print("\n" + "=" * 70 + f"\n  {step}\n" + "=" * 70)
    runpy.run_path(str(HERE / step), run_name="__main__")
print("\nPhase 1 complete.")
