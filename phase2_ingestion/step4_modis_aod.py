"""
PHASE 2 - STEP 4: MODIS MAIAC aerosol optical depth (AOD, 550 nm) - the
satellite proxy for PM2.5 / PM10 - as daily, quality-filtered, gap-filled
district series.

Only best-quality, clear-sky, land pixels are kept (AOD_QA bits, see
products.py). The IDW radius is small (5 km) because AOD varies quickly near
the coast and dust sources, so distant pixels are poor stand-ins.

Run (test one month first, then everything):
    python phase2_ingestion/step4_modis_aod.py --start 2024-01-01 --end 2024-01-31
    python phase2_ingestion/step4_modis_aod.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache_io import pipeline_lock
from cli import parse_dates
from daily_pipeline import run_product
from gee_utils import init_ee
from products import MODIS_KEYS, PRODUCTS


def main():
    args = parse_dates(__doc__.splitlines()[1])
    init_ee()
    for key in MODIS_KEYS:
        run_product(key, PRODUCTS[key], args.start, args.end)


if __name__ == "__main__":
    with pipeline_lock():
        main()
