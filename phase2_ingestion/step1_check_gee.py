"""
PHASE 2 - STEP 1: Check the Earth Engine connection, the datasets, and that
Earth Engine puts pixels exactly on our Phase 1 grid.

Run:
    python phase2_ingestion/step1_check_gee.py

What it checks
  1. You can log in (credentials.env has GEE_PROJECT).
  2. Every dataset we need exists, has data over Karachi, and has the bands
     the scripts expect. Prints the first and last date of each.
  3. Grid alignment: asks Earth Engine for the latitude/longitude of every
     cell on our 1 km grid and compares with what Phase 1 computed locally.
     They must match, otherwise every district average would be shifted.
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ee
import numpy as np
from pyproj import Transformer

import config
from gee_utils import fetch_array, init_ee, region
from spatial_utils import load_grid

DATASETS = {
    "COPERNICUS/S5P/OFFL/L3_NO2": ["tropospheric_NO2_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/NRTI/L3_NO2": ["tropospheric_NO2_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/OFFL/L3_O3": ["O3_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/NRTI/L3_O3": ["O3_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/OFFL/L3_SO2": ["SO2_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/NRTI/L3_SO2": ["SO2_column_number_density", "cloud_fraction"],
    "COPERNICUS/S5P/OFFL/L3_CO": ["CO_column_number_density"],
    "COPERNICUS/S5P/NRTI/L3_CO": ["CO_column_number_density"],
    "COPERNICUS/S5P/OFFL/L3_AER_AI": ["absorbing_aerosol_index"],
    "COPERNICUS/S5P/NRTI/L3_AER_AI": ["absorbing_aerosol_index"],
    "MODIS/061/MCD19A2_GRANULES": ["Optical_Depth_055", "AOD_QA"],
    "COPERNICUS/S2_SR_HARMONIZED": ["B4", "B8", "B11", "SCL"],
    "GOOGLE/DYNAMICWORLD/V1": ["label", "built"],
}


def main():
    project = init_ee()
    print(f"[1/3] Connected to Earth Engine (project: {project})")

    print("[2/3] Checking datasets over Karachi ...")
    geom = region()
    problems = 0
    for dataset_id, needed in DATASETS.items():
        coll = ee.ImageCollection(dataset_id).filterBounds(geom)
        info = ee.Dictionary({
            "first": coll.aggregate_min("system:time_start"),
            "last": coll.aggregate_max("system:time_start"),
            "bands": coll.first().bandNames(),
        }).getInfo()
        first = dt.datetime.fromtimestamp(info["first"] / 1000, dt.timezone.utc).date()
        last = dt.datetime.fromtimestamp(info["last"] / 1000, dt.timezone.utc).date()
        missing = [b for b in needed if b not in info["bands"]]
        status = "OK " if not missing else f"MISSING BANDS {missing}"
        problems += bool(missing)
        print(f"  {status} {dataset_id:32s} {first} -> {last}")
    if problems:
        sys.exit("Some datasets are missing bands - the scripts need updating.")

    print("[3/3] Checking grid alignment ...")
    grid = load_grid(1000)
    cube, _ = fetch_array(ee.Image.pixelLonLat(), 1000)
    ee_lon, ee_lat = cube[0], cube[1]
    t = grid["transform"]
    rows, cols = grid["shape"]
    cx, cy = np.meshgrid(t.c + t.a * (np.arange(cols) + 0.5), t.f + t.e * (np.arange(rows) + 0.5))
    lon, lat = Transformer.from_crs(config.CRS_UTM, config.CRS_WGS84, always_xy=True).transform(cx, cy)
    err_m = 111_000 * max(np.abs(ee_lon - lon).max(), np.abs(ee_lat - lat).max())
    print(f"  Largest difference between Earth Engine and Phase 1 cell centres: {err_m:.2f} m")
    if err_m > 50:
        sys.exit("Grid mismatch! Do not continue - report this.")
    print("\nAll checks passed. Earth Engine is ready.")


if __name__ == "__main__":
    main()
