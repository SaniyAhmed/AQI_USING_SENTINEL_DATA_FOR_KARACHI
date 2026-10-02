"""
PHASE 4 - STEP 1: List every ground-monitoring station in Karachi and assign it to a district.

Uses the OpenAQ v3 API (the aggregator that carries AirNow/US Consulate, AirGradient, Hawanama,
Clarity ... data). Each station is put into the district polygon that contains it (Phase 1 boundaries,
UTM 42N). A station that lies in no district is kept in the table but marked outside and never used.

Output
    data/raw/ground/stations.csv   one row per station: id, name, provider, lat, lon, first/last data,
                                   is_reference_grade, district_id, district, metres_outside
    data/raw/ground/sensors.csv    one row per sensor (a station has one sensor per pollutant)

Run:  python phase4_ground/step1_stations.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import config
import openaq
from cache_io import atomic_csv

GROUND_DIR = config.RAW_DIR / "ground"
BBOX = (66.30, 24.30, 67.90, 25.90)                     # west, south, east, north (a little wider than the districts)
SNAP_M = 500                                             # snap stations this close to a district border
KEEP = {"pm25", "pm10", "no2", "o3", "so2", "co"}        # pollutants we may use (Karachi only has PM in practice)


def main():
    stations, sensors = [], []
    for loc in openaq.pages("/locations", {"bbox": ",".join(map(str, BBOX))}):
        c = loc["coordinates"]
        stations.append({
            "location_id": loc["id"], "name": (loc.get("name") or "").strip(),
            "provider": (loc.get("provider") or {}).get("name"), "is_reference_grade": bool(loc.get("isMonitor")),
            "lat": c["latitude"], "lon": c["longitude"],
            "first_utc": (loc.get("datetimeFirst") or {}).get("utc"), "last_utc": (loc.get("datetimeLast") or {}).get("utc"),
        })
        for s in loc.get("sensors", []):
            sensors.append({"sensor_id": s["id"], "location_id": loc["id"], "parameter": s["parameter"]["name"],
                            "units": s["parameter"]["units"]})
    st = pd.DataFrame(stations)
    sn = pd.DataFrame(sensors)
    sn = sn[sn["parameter"].isin(KEEP)].reset_index(drop=True)
    print(f"{len(st)} stations in the search box, {len(sn)} usable sensors: {sn['parameter'].value_counts().to_dict()}")

    # ---- district of every station (Phase 1 polygons, metres)
    districts = gpd.read_file(config.BOUNDARY_DIR / "karachi_districts_utm42n.gpkg")
    idcol = next(c for c in ("district_id", "id") if c in districts.columns)
    namecol = next(c for c in ("district", "name") if c in districts.columns)
    pts = gpd.GeoDataFrame(st, geometry=[Point(x, y) for x, y in zip(st["lon"], st["lat"])], crs=config.CRS_WGS84).to_crs(config.CRS_UTM)
    inside = gpd.sjoin(pts, districts[[idcol, namecol, "geometry"]], how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    st["district_id"] = inside[idcol].to_numpy()
    st["district"] = inside[namecol].to_numpy()
    # A point just outside every polygon (coastline / boundary precision) is snapped to the nearest district
    # when it is within SNAP_M metres; farther ones stay "outside" and are never used.
    dist = pd.DataFrame({r[namecol]: pts.geometry.distance(r.geometry) for _, r in districts.iterrows()})
    nearest_m = dist.min(axis=1)
    snapped = st["district_id"].isna() & (nearest_m <= SNAP_M)
    near_row = dist.idxmin(axis=1)
    by_name = districts.set_index(namecol)[idcol]
    st.loc[snapped, "district"] = near_row[snapped].to_numpy()
    st.loc[snapped, "district_id"] = by_name.reindex(near_row[snapped]).to_numpy()
    st["metres_outside"] = [0.0 if pd.notna(d) else round(float(m), 0) for d, m in zip(inside[idcol].to_numpy(), nearest_m)]
    st["snapped_to_district"] = snapped.to_numpy()
    st["district_id"] = st["district_id"].astype("Int64")
    st = st.merge(sn.groupby("location_id")["parameter"].apply(lambda s: ",".join(sorted(set(s)))).rename("parameters"),
                  on="location_id", how="left")
    st = st.sort_values(["district_id", "provider", "name"]).reset_index(drop=True)

    GROUND_DIR.mkdir(parents=True, exist_ok=True)
    atomic_csv(st, GROUND_DIR / "stations.csv")
    atomic_csv(sn, GROUND_DIR / "sensors.csv")
    used = st[st["district_id"].notna() & st["parameters"].notna()]
    print(f"\nInside a district and with a usable sensor: {len(used)} stations; outside all districts: "
          f"{int((st['district_id'].isna()).sum())} (nearest {st.loc[st['district_id'].isna(), 'metres_outside'].min():.0f} m)")
    print("\nStations per district and provider:")
    print(pd.crosstab(used["district"], used["provider"]).to_string())
    print("\nStations per district by last-data year:")
    print(pd.crosstab(used["district"], pd.to_datetime(used["last_utc"]).dt.year).to_string())
    print("\nOutside all districts:")
    print(st[st["district_id"].isna()][["location_id", "name", "provider", "lat", "lon", "metres_outside"]].to_string(index=False))


if __name__ == "__main__":
    main()
