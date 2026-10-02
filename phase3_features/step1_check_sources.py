"""
PHASE 3 - STEP 1: Check that Earth Engine's ERA5 and CAMS lattices line up with ours.

For one ERA5 hour and one CAMS forecast step we read the small lattice with
computePixels (what the pipeline does) and compare it, node by node, with an
independent point sample of the same image at the node's coordinates.
They must agree, otherwise every district value would come from the wrong cell.
Also prints which CAMS runs exist for a test day (00Z and 12Z, 3-hourly leads).

Run:  python phase3_features/step1_check_sources.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import ee
import numpy as np

import config
import sources
from gee_utils import fetch_array, init_ee


def compare(name, image, band, nodes):
    cube, _ = fetch_array(image.select(band), sources.ee_grid(nodes))
    lats, lons = sources.axes(nodes)
    proj = image.select(band).projection()
    worst = 0.0
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            v = image.select(band).reduceRegion(ee.Reducer.first(), ee.Geometry.Point([float(lo), float(la)]),
                                                proj.nominalScale(), crs=proj).get(band).getInfo()
            worst = max(worst, abs(v - float(cube[0, i, j])) / (abs(v) + 1e-9))
    ok = worst < 1e-5
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {cube.shape[1]}x{cube.shape[2]} nodes, worst relative difference {worst:.1e}")
    return ok


def main():
    init_ee()
    era = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate("2024-07-10T12:00", "2024-07-10T13:00").first()
    ok = compare("ERA5 temperature_2m", era, "temperature_2m", config.ERA5_NODES)
    cams = (ee.ImageCollection("ECMWF/CAMS/NRT").filter(ee.Filter.eq("model_initialization_datetime", "2024-07-09T12:00:00"))
            .filterDate("2024-07-10T00:00", "2024-07-10T01:00").first())
    ok &= compare("CAMS total AOD 550 nm", cams, "total_aerosol_optical_depth_at_550nm_surface", config.CAMS_NODES)
    day = ee.ImageCollection("ECMWF/CAMS/NRT").filterDate("2024-07-10", "2024-07-11")
    runs = day.aggregate_array("model_initialization_datetime").distinct().sort().getInfo()
    print("CAMS runs holding data valid on 2024-07-10:", runs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
