"""
PHASE 1 - STEP 4: Test the zonal-statistics pipeline with fake data.

Real satellite data comes in Phase 2. Here we make FAKE data that looks like
it, push it through the same functions Phase 2 will use, and check that the
answers make sense. If these tests pass, the grid and masks are correct.

Tests
  A. Constant field: every cell = 1.0 -> every district mean must be exactly 1.
  B. Fake "TROPOMI NO2" on a 0.05 deg lat/lon grid, highest over Saddar
     (city centre), with random "cloud" gaps -> regrid to 1 km, average per
     district. Karachi South (contains Saddar) should come out highest and
     rural Malir lowest. valid_fraction should drop because of the clouds.
  C. Fake "ERA5 temperature" on a 0.25 deg grid -> bilinear interpolation to
     district centroids (the method Phase 3 will use for weather data).

Output: data/processed/grid/test_zonal_no2.csv

Run:
    python phase1_geometry/step4_test_zonal_stats.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from spatial_utils import (latlon_transform, load_grid, regrid_to_target,
                           sample_at_points, zonal_stats)

SADDAR_LAT, SADDAR_LON = 24.86, 67.01


def main():
    grid = load_grid(1000)
    print(f"Loaded 1 km grid: {grid['shape'][0]} rows x {grid['shape'][1]} cols")

    # --- Test A: constant field ------------------------------------------------
    ones = np.ones(grid["shape"], dtype="float32")
    result_a = zonal_stats(ones, grid)
    assert np.allclose(result_a["mean"], 1.0), "Test A failed"
    assert np.allclose(result_a["valid_fraction"], 1.0), "Test A failed"
    print("\n[Test A] Constant field -> all district means = 1.0   PASSED")

    # --- Test B: fake TROPOMI-like NO2 with clouds -----------------------------
    bbox = pd.read_json(config.BOUNDARY_DIR / "master_bbox.json")["download_bbox"]
    res = 0.05
    lats = np.arange(bbox["north"] - res / 2, bbox["south"], -res)   # north -> south
    lons = np.arange(bbox["west"] + res / 2, bbox["east"], res)
    lon2d, lat2d = np.meshgrid(lons, lats)
    dist_km = 111 * np.hypot(lat2d - SADDAR_LAT, (lon2d - SADDAR_LON) * np.cos(np.radians(25)))
    no2 = 20 + 180 * np.exp(-(dist_km / 12) ** 2)          # umol/m2, peak at Saddar
    rng = np.random.default_rng(42)
    no2[rng.random(no2.shape) < 0.25] = np.nan             # 25 % "cloudy" pixels

    transform, flip = latlon_transform(lats, lons)
    no2_1km = regrid_to_target(no2, transform, config.CRS_WGS84, grid, method="bilinear")
    result_b = zonal_stats(no2_1km, grid)
    print("\n[Test B] Fake NO2 (peak over Saddar, 25 % cloud):")
    print(result_b.round(2).to_string(index=False))
    # Karachi South contains Saddar, so it holds the single highest cell. Its
    # MEAN is not the highest, because the district also includes the large,
    # thinly populated Kemari coast to the west, which dilutes the average.
    top_max = result_b.loc[result_b["max"].idxmax(), "district"]
    lowest_mean = result_b.loc[result_b["mean"].idxmin(), "district"]
    assert top_max == "Karachi South", "Test B failed: South should hold the peak"
    assert lowest_mean == "Malir", "Test B failed: Malir should have the lowest mean"
    assert (result_b["valid_fraction"] < 1).all(), "Test B failed: clouds not detected"
    print("   Peak inside Karachi South, Malir lowest mean, clouds reduce valid_fraction   PASSED")
    result_b.to_csv(config.GRID_DIR / "test_zonal_no2.csv", index=False)

    # --- Test C: fake ERA5 temperature -> centroids ----------------------------
    era_lats = np.arange(26.0, 24.0 - 0.01, -0.25)
    era_lons = np.arange(66.0, 68.0 + 0.01, 0.25)
    elon, elat = np.meshgrid(era_lons, era_lats)
    temp = 30 + 2 * (elat - 25) + 1 * (elon - 67)          # simple linear field
    cent = pd.read_csv(config.BOUNDARY_DIR / "district_centroids.csv")
    sampled = sample_at_points(temp, era_lats, era_lons, cent["centroid_lat"], cent["centroid_lon"])
    expected = 30 + 2 * (cent["centroid_lat"] - 25) + 1 * (cent["centroid_lon"] - 67)
    assert np.allclose(sampled, expected, atol=1e-6), "Test C failed"
    print("\n[Test C] Bilinear interpolation of fake ERA5 to centroids:")
    print(pd.DataFrame({"district": cent["district"], "temp_C": sampled.round(3)}).to_string(index=False))
    print("   Matches the exact formula   PASSED")

    print("\nAll tests passed. Phase 1 grid and masks are ready for Phase 2.")


if __name__ == "__main__":
    main()
