"""
PHASE 1 - STEP 3: Build the target grids (100 m and 1 km, UTM 42N) and the
district masks.

Key idea
  Satellite products all arrive on different grids (TROPOMI ~5.5 km, MODIS
  1 km, ERA5 ~28 km, Sentinel-2 10 m). In Phase 2 we will resample every
  product onto ONE common grid. This script defines that grid and records,
  for every grid cell, which district it belongs to.

What it does
  1. Builds a 100 m grid covering all districts plus a 2 km margin. Its edges
     are snapped to whole kilometres so ten 100 m cells fit exactly inside
     each 1 km cell.
  2. "Burns" (rasterizes) the district polygons onto the 100 m grid:
       district_id_100m.tif   one band, value = district id (0 = outside)
       district_mask_100m.tif six bands, band k = 1 inside district k, else 0
  3. Builds the 1 km grid by grouping 10 x 10 blocks of 100 m cells. Instead
     of a plain yes/no mask it stores the FRACTION of each 1 km cell that
     falls inside each district (0.0 - 1.0). A cell that lies half in
     Korangi and half in Karachi East gets 0.5 for each. The zonal averages
     in step 4 use these fractions as weights, which is more accurate than
     a yes/no mask for small districts like Karachi Central.
       district_fraction_1000m.tif  six float bands, fraction per district
       district_id_1000m.tif        one band, the district covering >50 %
  4. Saves grid_definition.json describing both grids.

Run:
    python phase1_geometry/step3_build_grid_and_masks.py
"""
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FINE = 100      # metres
COARSE = 1000   # metres
FACTOR = COARSE // FINE


def write_geotiff(path, array, transform, dtype, nodata=None, band_names=None):
    """Save a 2-D (one band) or 3-D (bands, rows, cols) array as a GeoTIFF."""
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    profile = {
        "driver": "GTiff", "height": array.shape[1], "width": array.shape[2],
        "count": array.shape[0], "dtype": dtype, "crs": config.CRS_UTM,
        "transform": transform, "compress": "deflate", "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)


def main():
    config.GRID_DIR.mkdir(parents=True, exist_ok=True)
    districts = gpd.read_file(config.BOUNDARY_DIR / "karachi_districts_utm42n.gpkg")
    districts = districts.sort_values("district_id").reset_index(drop=True)
    ids = districts["district_id"].tolist()
    names = districts["district"].tolist()

    # 1. Grid extent, snapped outwards to whole kilometres ---------------------
    minx, miny, maxx, maxy = districts.total_bounds
    buf = config.GRID_BUFFER_M
    xmin = math.floor((minx - buf) / COARSE) * COARSE
    ymin = math.floor((miny - buf) / COARSE) * COARSE
    xmax = math.ceil((maxx + buf) / COARSE) * COARSE
    ymax = math.ceil((maxy + buf) / COARSE) * COARSE

    width_c, height_c = (xmax - xmin) // COARSE, (ymax - ymin) // COARSE
    width_f, height_f = width_c * FACTOR, height_c * FACTOR
    # from_origin(left, top, pixel width, pixel height): row 0 is the NORTH edge.
    transform_f = from_origin(xmin, ymax, FINE, FINE)
    transform_c = from_origin(xmin, ymax, COARSE, COARSE)
    print(f"[1/4] Grid extent (UTM m): x {xmin}..{xmax}, y {ymin}..{ymax}")
    print(f"      100 m grid: {width_f} cols x {height_f} rows")
    print(f"      1 km  grid: {width_c} cols x {height_c} rows")

    # 2. Rasterize at 100 m ----------------------------------------------------
    # A cell gets a district's id if the cell CENTRE is inside the polygon.
    shapes = zip(districts.geometry, districts["district_id"])
    id_100 = rasterize(shapes, out_shape=(height_f, width_f), transform=transform_f,
                       fill=0, dtype="uint8", all_touched=False)
    masks_100 = np.stack([(id_100 == d).astype("uint8") for d in ids])
    write_geotiff(config.GRID_DIR / "district_id_100m.tif", id_100, transform_f, "uint8", nodata=0)
    write_geotiff(config.GRID_DIR / "district_mask_100m.tif", masks_100, transform_f, "uint8",
                  band_names=names)
    print("[2/4] 100 m masks written. Area check (rasterized vs polygon):")
    for d, name, poly_area in zip(ids, names, districts.geometry.area / 1e6):
        raster_area = (id_100 == d).sum() * FINE * FINE / 1e6
        print(f"      {name:16s} raster {raster_area:8.1f} km2   polygon {poly_area:8.1f} km2"
              f"   diff {100 * (raster_area - poly_area) / poly_area:+.2f} %")

    # 3. Aggregate 10x10 blocks -> 1 km fractions ------------------------------
    # reshape (bands, rows, 10, cols, 10) and average over the two "10" axes.
    frac_1000 = masks_100.reshape(len(ids), height_c, FACTOR, width_c, FACTOR).mean(axis=(2, 4))
    frac_1000 = frac_1000.astype("float32")
    best = frac_1000.argmax(axis=0)
    id_1000 = np.where(frac_1000.max(axis=0) > 0.5, np.array(ids)[best], 0).astype("uint8")
    write_geotiff(config.GRID_DIR / "district_fraction_1000m.tif", frac_1000, transform_c,
                  "float32", band_names=names)
    write_geotiff(config.GRID_DIR / "district_id_1000m.tif", id_1000, transform_c, "uint8", nodata=0)
    print("[3/4] 1 km fraction masks written. Cells per district:")
    for k, (d, name) in enumerate(zip(ids, names)):
        touched = (frac_1000[k] > 0).sum()
        full = (frac_1000[k] == 1).sum()
        print(f"      {name:16s} {touched:6d} cells touch it, {full:6d} fully inside, "
              f"weighted area {frac_1000[k].sum():8.1f} km2")

    # 4. Grid definition -------------------------------------------------------
    grid_def = {
        "crs": config.CRS_UTM,
        "bounds_utm": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        "districts": [{"district_id": d, "district": n, "band": i + 1}
                      for i, (d, n) in enumerate(zip(ids, names))],
        "grids": {
            str(FINE): {"resolution_m": FINE, "width": width_f, "height": height_f,
                        "transform": list(transform_f)[:6]},
            str(COARSE): {"resolution_m": COARSE, "width": width_c, "height": height_c,
                          "transform": list(transform_c)[:6]},
        },
    }
    with open(config.GRID_DIR / "grid_definition.json", "w") as f:
        json.dump(grid_def, f, indent=2)
    print(f"[4/4] Saved grid definition and masks to {config.GRID_DIR}")


if __name__ == "__main__":
    main()
