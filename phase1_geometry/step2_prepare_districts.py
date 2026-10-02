"""
PHASE 1 - STEP 2: Clean the district polygons, reproject to UTM 42N,
compute areas and centroids, and build the master bounding box.

What it does
  1. Keeps only the six Karachi districts and gives them clean names/ids.
  2. Repairs any broken geometry (self-intersections etc.).
  3. Reprojects from WGS84 (degrees) to UTM Zone 42N (metres).
  4. Calculates each district's area in km2 (only meaningful in metres).
  5. Calculates district centroids (used in Phase 3 to interpolate the
     coarse ERA5/GFS weather grids to each district).
  6. Builds the master bounding box and compares it with the proposed box.

Outputs (data/processed/boundaries/)
  karachi_districts_wgs84.geojson   polygons in lat/lon
  karachi_districts_utm42n.gpkg     polygons in metres
  district_centroids.csv            centroid of each district (both CRSs)
  master_bbox.json                  box to download satellite data for

Run:
    python phase1_geometry/step2_prepare_districts.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    config.BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and keep only our six districts ---------------------------------
    adm2 = gpd.read_file(config.RAW_BOUNDARY_DIR / config.HDX_ADM2_FILE)
    gdf = adm2[adm2["adm2_pcode"].isin(config.DISTRICTS)].copy()
    gdf["district_id"] = gdf["adm2_pcode"].map(lambda p: config.DISTRICTS[p][0])
    gdf["district"] = gdf["adm2_pcode"].map(lambda p: config.DISTRICTS[p][1])
    gdf = gdf[["district_id", "district", "adm2_pcode", "adm2_name", "geometry"]]
    gdf = gdf.sort_values("district_id").reset_index(drop=True)
    print(f"[1/6] Loaded {len(gdf)} districts. Source CRS: {gdf.crs}")

    # 2. Repair geometry -------------------------------------------------------
    n_invalid = (~gdf.geometry.is_valid).sum()
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    print(f"[2/6] Repaired {n_invalid} invalid geometries.")

    # Make sure the CRS is WGS84 (the file says so, but be explicit).
    gdf_wgs = gdf.to_crs(config.CRS_WGS84)

    # 3. Reproject to UTM 42N --------------------------------------------------
    gdf_utm = gdf_wgs.to_crs(config.CRS_UTM)
    print(f"[3/6] Reprojected to {config.CRS_UTM} (UTM Zone 42N, metres).")

    # 4. Areas -----------------------------------------------------------------
    gdf_utm["area_km2"] = gdf_utm.geometry.area / 1e6
    gdf_wgs["area_km2"] = gdf_utm["area_km2"].values
    print("[4/6] District areas:")
    print(gdf_utm[["district_id", "district", "area_km2"]].round(1).to_string(index=False))

    # 5. Centroids -------------------------------------------------------------
    # Centroids are computed in UTM (metres) because computing them in degrees
    # gives slightly wrong positions. A "representative point" is also stored:
    # it is always inside the polygon, even for odd shapes where the true
    # centroid falls outside.
    centroid_utm = gdf_utm.geometry.centroid
    rep_utm = gdf_utm.geometry.representative_point()
    centroid_wgs = gpd.GeoSeries(centroid_utm, crs=config.CRS_UTM).to_crs(config.CRS_WGS84)
    rep_wgs = gpd.GeoSeries(rep_utm, crs=config.CRS_UTM).to_crs(config.CRS_WGS84)
    centroids = pd.DataFrame({
        "district_id": gdf_utm["district_id"],
        "district": gdf_utm["district"],
        "centroid_lon": centroid_wgs.x.round(5),
        "centroid_lat": centroid_wgs.y.round(5),
        "centroid_x_utm": centroid_utm.x.round(1),
        "centroid_y_utm": centroid_utm.y.round(1),
        "centroid_inside": centroid_utm.within(gdf_utm.geometry),
        "rep_point_lon": rep_wgs.x.round(5),
        "rep_point_lat": rep_wgs.y.round(5),
    })
    print("[5/6] Centroids:")
    print(centroids[["district", "centroid_lat", "centroid_lon", "centroid_inside"]].to_string(index=False))

    # 6. Master bounding box ---------------------------------------------------
    west, south, east, north = gdf_wgs.total_bounds
    b = config.BBOX_BUFFER_DEG
    bbox = {
        "districts_exact": {"west": west, "south": south, "east": east, "north": north},
        "download_bbox": {  # rounded outwards to 0.01 deg
            "west": round(west - b - 0.005, 2), "south": round(south - b - 0.005, 2),
            "east": round(east + b + 0.005, 2), "north": round(north + b + 0.005, 2),
        },
        "proposed_bbox": config.PROPOSED_BBOX_WGS84,
        "crs": config.CRS_WGS84,
    }
    print("[6/6] Bounding boxes (degrees):")
    for name, box in bbox.items():
        if isinstance(box, dict):
            print(f"  {name:16s} W {box['west']:.3f}  S {box['south']:.3f}  "
                  f"E {box['east']:.3f}  N {box['north']:.3f}")

    # How much of each district the proposed box would cut off.
    p = config.PROPOSED_BBOX_WGS84
    from shapely.geometry import box as make_box
    proposed_utm = gpd.GeoSeries(
        [make_box(p["west"], p["south"], p["east"], p["north"])], crs=config.CRS_WGS84
    ).to_crs(config.CRS_UTM).iloc[0]
    outside = 100 * (1 - gdf_utm.geometry.intersection(proposed_utm).area / gdf_utm.geometry.area)
    outside = outside.clip(lower=0)  # avoid "-0.0 %" from rounding noise
    print("  Share of each district OUTSIDE the proposed box:")
    for name, pct in zip(gdf_utm["district"], outside):
        print(f"    {name:16s} {pct:5.1f} %")
    print("  -> the download_bbox (built from the real polygons) is used instead.")

    # Save ---------------------------------------------------------------------
    gdf_wgs.to_file(config.BOUNDARY_DIR / "karachi_districts_wgs84.geojson", driver="GeoJSON")
    gdf_utm.to_file(config.BOUNDARY_DIR / "karachi_districts_utm42n.gpkg", driver="GPKG")
    centroids.to_csv(config.BOUNDARY_DIR / "district_centroids.csv", index=False)
    with open(config.BOUNDARY_DIR / "master_bbox.json", "w") as f:
        json.dump(bbox, f, indent=2)
    print(f"\nSaved outputs to {config.BOUNDARY_DIR}")


if __name__ == "__main__":
    main()
