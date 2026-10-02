"""
PHASE 2 - STEP 3: Sentinel-5P TROPOMI gases - NO2, O3, SO2, CO and the UV
aerosol index - as daily, quality-filtered, gap-filled district series.

The quality filters are explained in products.py. The gap filling is
explained in gapfill.py.

Run (test one month first, then everything):
    python phase2_ingestion/step3_sentinel5p.py --start 2024-01-01 --end 2024-01-31
    python phase2_ingestion/step3_sentinel5p.py
Only some gases:
    python phase2_ingestion/step3_sentinel5p.py --products no2 o3

A full run from START_DATE (2022-01) makes ~60 Earth Engine requests per gas
(2-4 s each, 8 in parallel). If it stops, run it again: finished months are cached.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache_io import pipeline_lock
from cli import parse_dates
from daily_pipeline import run_product
from gee_utils import init_ee
from products import PRODUCTS, S5P_KEYS


def main():
    args = parse_dates(__doc__.splitlines()[1], S5P_KEYS)
    init_ee()
    for key in args.products:
        run_product(key, PRODUCTS[key], args.start, args.end)


if __name__ == "__main__":
    with pipeline_lock():
        main()
