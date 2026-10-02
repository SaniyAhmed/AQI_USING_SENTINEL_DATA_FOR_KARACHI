"""
sources.py - the native lattices of the meteorological sources, and bilinear
interpolation from those lattices to the six district centroids.

A "lattice" is a small regular lat/lon grid of node values (the source's own
grid points). We read it WITHOUT resampling, then interpolate bilinearly to
each centroid ourselves, exactly as the project plan asks (ERA5/GFS ~ 28 km
-> district centroids).

    nodes = config.ERA5_NODES         {"lon0", "lat_top", "step", "n_lon", "n_lat"}
    axes(nodes)                       -> lats (north to south), lons (west to east)
    ee_grid(nodes)                    -> grid dict for ee.data.computePixels
    centroids()                       -> ids, names, lat, lon of the 6 districts
    bilinear(cube, nodes)             -> (..., 6) values at the centroids
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import config


def axes(nodes):
    lons = nodes["lon0"] + nodes["step"] * np.arange(nodes["n_lon"])
    lats = nodes["lat_top"] - nodes["step"] * np.arange(nodes["n_lat"])
    return lats, lons


def ee_grid(nodes):
    """Earth Engine pixel grid whose pixel CENTRES are exactly the lattice nodes."""
    s = nodes["step"]
    return {
        "dimensions": {"width": nodes["n_lon"], "height": nodes["n_lat"]},
        "affineTransform": {"scaleX": s, "shearX": 0, "translateX": nodes["lon0"] - s / 2,
                            "shearY": 0, "scaleY": -s, "translateY": nodes["lat_top"] + s / 2},
        "crsCode": config.CRS_WGS84,
    }


def centroids():
    """District centroids from Phase 1, ordered by district_id."""
    c = pd.read_csv(config.BOUNDARY_DIR / "district_centroids.csv").sort_values("district_id")
    return (c["district_id"].to_numpy(), c["district"].to_list(),
            c["centroid_lat"].to_numpy(), c["centroid_lon"].to_numpy())


def bilinear(cube, nodes):
    """
    cube: (..., n_lat, n_lon) values on the lattice (latitude from north to south).
    Returns (..., 6): bilinear interpolation at the six centroids.
    If any of the four surrounding nodes is NaN the result is NaN (never guessed).
    """
    lats, lons = axes(nodes)
    _, _, clat, clon = centroids()
    cube = np.asarray(cube, dtype="float64")
    lead = cube.shape[:-2]
    flat = cube.reshape(-1, *cube.shape[-2:])                       # (n, lat, lon)
    values = np.moveaxis(flat, 0, -1)[::-1]                          # (lat ascending, lon, n)
    interp = RegularGridInterpolator((lats[::-1], lons), values, method="linear",
                                     bounds_error=True)
    out = interp(np.column_stack([clat, clon]))                      # (6, n)
    return np.moveaxis(out, 0, -1).reshape(*lead, len(clat))
