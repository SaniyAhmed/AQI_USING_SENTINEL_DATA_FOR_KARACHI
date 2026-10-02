"""
Runs all Phase 2 steps, one calendar year at a time, and ends with the validation gate.

    python phase2_ingestion/run_phase2.py                  (START_DATE in config.py -> yesterday)
    python phase2_ingestion/run_phase2.py --start 2024-01-01 --end 2024-12-31

Order
  0. clean up temporary files an interrupted run may have left behind
  1. step1_check_gee          once
  2. step2_sentinel2_landcover once, covers the whole period (monthly composites are cached)
  3. for each 1-year chunk (config.CHUNK_YEARS):
        step3_sentinel5p
        step4_modis_aod
  4. step5_quality_report     once, over everything
  5. step7_finalize           final daily table (csv + parquet)
  6. step8_validate           the gate: exit code 1 if ANY check fails

Why chunks? Each year is a separate, smaller batch of Earth Engine requests
and fits comfortably in memory. If a year fails, the error names the year;
just run the command again. Finished months are cached in data/cache/gee/
(every cache file is written atomically and re-checked when read, so an
interrupted run can never leave a corrupt file that is trusted later).
The gap filling still looks across year boundaries: each chunk also loads the
last days of the previous year (from the cache) for the temporal window.

Only one Phase 2 process may run at a time (a lock enforces this).
"""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config
from cache_io import clean_stale_tmp, pipeline_lock
from cli import parse_dates
from gee_utils import annual_chunks

CHUNKED_STEPS = [
    "step3_sentinel5p.py",
    "step4_modis_aod.py",
]


def run_step(step, args=()):
    print("\n" + "=" * 70 + f"\n  {step} {' '.join(args)}\n" + "=" * 70)
    sys.argv = [step, *args]
    runpy.run_path(str(HERE / step), run_name="__main__")


def main():
    dates = parse_dates(__doc__.splitlines()[1])
    chunks = annual_chunks(dates.start, dates.end)
    print(f"Phase 2: {dates.start} -> {dates.end} in {len(chunks)} yearly chunk(s)")

    removed = clean_stale_tmp(config.DATA_DIR, config.FIGURE_DIR)
    if removed:
        print(f"Removed {removed} temporary file(s) left by an interrupted run.")

    run_step("step1_check_gee.py")
    run_step("step2_sentinel2_landcover.py")
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        t0 = time.time()
        print(f"\n##### Chunk {i}/{len(chunks)}: {chunk_start} -> {chunk_end} #####")
        for step in CHUNKED_STEPS:
            try:
                run_step(step, ["--start", chunk_start, "--end", chunk_end])
            except Exception:
                print(f"\nFAILED in {step} for chunk {chunk_start} -> {chunk_end}.\n"
                      "Run the same command again to resume (finished months are cached), or run\n"
                      f"    python phase2_ingestion/{step} --start {chunk_start} --end {chunk_end}")
                raise
        print(f"##### Chunk {i}/{len(chunks)} done in {(time.time() - t0) / 60:.1f} min #####")
    run_step("step5_quality_report.py")
    run_step("step7_finalize.py")
    run_step("step8_validate.py")          # raises SystemExit(1) if a check fails
    print("\nPhase 2 complete and validated.")


if __name__ == "__main__":
    with pipeline_lock():
        main()
