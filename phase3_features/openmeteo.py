"""
openmeteo.py - hourly meteorology from Open-Meteo (NOAA GFS forecasts / archived GFS / ERA5 copy),
turned into the SAME local-day statistics as step2_era5.py, at the six district centroids.

Why Open-Meteo: it is the one free, key-less service that serves every field the plan needs from
NOAA GFS - including boundary-layer height and the 850 hPa temperature - as clean JSON, without
GRIB decoding. (Data: NOAA GFS / ECMWF ERA5 via open-meteo.com, CC BY 4.0.)

Endpoints used (config.py)
    historical-forecast-api  archived GFS ("analysis-like" first hours of every run)  -> T850 history
    api /v1/forecast         the latest GFS run, 16 days ahead                        -> operational T+0..T+2

Nodes: the 12 GFS 0.25 deg grid points around the centroids are requested and interpolated
bilinearly by sources.bilinear (never the provider's own interpolation).

    fetch_hourly(url, variables, start=..., end=..., models=...)   -> DataFrame (time, lat, lon, vars)
    local_day_stats(hourly)                                        -> dict stat -> array (days, lat, lon)
"""
import time

import numpy as np
import pandas as pd
import requests

import config
import sources

VARS_SURFACE = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "wind_direction_10m",
                "surface_pressure", "boundary_layer_height", "precipitation"]
NODES = {"lon0": config.GFS_NODE_LONS[0], "lat_top": config.GFS_NODE_LATS[-1], "step": 0.25,
         "n_lon": len(config.GFS_NODE_LONS), "n_lat": len(config.GFS_NODE_LATS)}
NODE_LATS = sources.axes(NODES)[0]                # north to south
NODE_LONS = sources.axes(NODES)[1]


def _get(url, params, tries=5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:      # rate limit / server trouble: wait and retry
                time.sleep(20 * 2 ** attempt)
                continue
            raise RuntimeError(f"Open-Meteo {r.status_code}: {r.text[:200]}")
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(10 * 2 ** attempt)
    raise RuntimeError("Open-Meteo did not answer after several tries")


def fetch_hourly(url, variables, nodes_lat=None, nodes_lon=None, **params):
    """Hourly values (UTC) at every lattice node -> DataFrame [time, lat, lon, <variables>]."""
    lats, lons = (NODE_LATS if nodes_lat is None else nodes_lat), (NODE_LONS if nodes_lon is None else nodes_lon)
    pairs = [(la, lo) for la in lats for lo in lons]
    out = _get(url, {"latitude": ",".join(f"{la:.4f}" for la, _ in pairs),
                     "longitude": ",".join(f"{lo:.4f}" for _, lo in pairs),
                     "hourly": ",".join(variables), "timezone": "UTC", "wind_speed_unit": "ms", **params})
    out = out if isinstance(out, list) else [out]
    if len(out) != len(pairs):
        raise RuntimeError(f"Open-Meteo returned {len(out)} locations for {len(pairs)} requested")
    frames = []
    for (la, lo), res in zip(pairs, out):
        df = pd.DataFrame(res["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["lat"], df["lon"] = la, lo
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def magnus_rh(t, td):
    f = lambda x: np.exp(17.625 * x / (243.04 + x))
    return np.clip(100 * f(td) / f(t), 0, 100)


def local_day_stats(hourly, need_hours=24):
    """
    Same statistics as step2_era5.STATS (plus t850_mean when present), from hourly node data.
    Days without `need_hours` hourly values are dropped. Returns (dates, {stat: (days, lat, lon)}).
    """
    h = hourly.copy()
    h["day"] = (h["time"] + pd.Timedelta(hours=config.LOCAL_UTC_OFFSET_H)).dt.normalize()
    rad = np.deg2rad(h["wind_direction_10m"]) if "wind_direction_10m" in h else None
    h["t2m"], h["td"] = h["temperature_2m"], h["dew_point_2m"]
    h["rh"] = magnus_rh(h["t2m"], h["td"])
    h["ws"] = h["wind_speed_10m"]
    h["u10"], h["v10"] = -h["ws"] * np.sin(rad), -h["ws"] * np.cos(rad)      # meteorological "from" direction
    h["blh"], h["sp"], h["precip"] = h["boundary_layer_height"], h["surface_pressure"], h["precipitation"]
    h["vent"] = h["blh"] * h["ws"]
    g = h.groupby(["day", "lat", "lon"])
    agg = pd.concat({
        **{f"{b}_mean": g[b].mean() for b in ["t2m", "td", "rh", "u10", "v10", "ws", "blh", "vent", "sp"]},
        **{f"{b}_max": g[b].max() for b in ["t2m", "blh", "ws"]},
        **{f"{b}_min": g[b].min() for b in ["t2m", "rh", "blh", "vent"]},
        "precip_sum": g["precip"].sum(), "n_hours": g["t2m"].count(),
        **({"t850_mean": g["temperature_850hPa"].mean()} if "temperature_850hPa" in h else {}),
    }, axis=1)
    agg = agg[agg["n_hours"] >= need_hours]
    days = pd.DatetimeIndex(sorted(agg.index.get_level_values("day").unique()))
    cube = {}
    for stat in agg.columns:
        arr = np.full((len(days), len(NODE_LATS), len(NODE_LONS)), np.nan)
        s = agg[stat].reset_index()
        di = days.get_indexer(s["day"])
        li = np.searchsorted(-NODE_LATS, -s["lat"].to_numpy())               # NODE_LATS descends
        lj = np.searchsorted(NODE_LONS, s["lon"].to_numpy())
        arr[di, li, lj] = s[stat].to_numpy()
        cube[stat] = arr
    return days, cube


def to_centroid_table(days, cube):
    """Bilinear interpolation of every stat to the six centroids -> long DataFrame."""
    ids, names, _, _ = sources.centroids()
    stats = list(cube)
    at = sources.bilinear(np.stack([cube[s] for s in stats], axis=1), NODES)   # (days, stats, 6)
    rows = []
    for k, (did, name) in enumerate(zip(ids, names)):
        df = pd.DataFrame(at[:, :, k], columns=stats)
        df.insert(0, "date", days); df.insert(1, "district_id", did); df.insert(2, "district", name)
        rows.append(df)
    return pd.concat(rows).sort_values(["date", "district_id"]).reset_index(drop=True)
