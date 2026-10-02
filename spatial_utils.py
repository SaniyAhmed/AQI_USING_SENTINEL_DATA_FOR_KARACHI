"""
spatial_utils.py — reusable helpers for putting any gridded data onto our
target grid and averaging it per district (zonal statistics).

Built in Phase 1 and reused in Phases 2-4:
    from spatial_utils import load_grid, regrid_to_target, zonal_stats
"""
import json

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy.interpolate import RegularGridInterpolator

import config


def load_grid(resolution_m=1000):
    """
    Load one of the target grids created in Phase 1 step 3.

    Returns a dict with:
      transform, crs, shape (rows, cols)
      weights  array (n_districts, rows, cols) - fraction of each cell in each district
      ids, names  district ids and names, in band order
    """
    with open(config.GRID_DIR / "grid_definition.json") as f:
        grid_def = json.load(f)
    g = grid_def["grids"][str(resolution_m)]
    mask_file = ("district_fraction_1000m.tif" if resolution_m == 1000
                 else f"district_mask_{resolution_m}m.tif")
    with rasterio.open(config.GRID_DIR / mask_file) as src:
        weights = src.read().astype("float32")
    return {
        "resolution_m": resolution_m,
        "transform": Affine(*g["transform"]),
        "crs": grid_def["crs"],
        "shape": (g["height"], g["width"]),
        "weights": weights,
        "ids": [d["district_id"] for d in grid_def["districts"]],
        "names": [d["district"] for d in grid_def["districts"]],
    }


def latlon_transform(lats, lons):
    """
    Affine transform for a regular lat/lon grid given its 1-D cell-centre
    coordinate arrays (e.g. ERA5 or a Level-3 satellite grid).
    Returns (transform, flip) where flip=True means latitude was ascending
    and the data rows must be reversed so row 0 is the northernmost.
    """
    lats, lons = np.asarray(lats), np.asarray(lons)
    dx = float(lons[1] - lons[0])
    dy = float(abs(lats[1] - lats[0]))
    flip = lats[0] < lats[-1]
    north = float(lats.max()) + dy / 2
    west = float(lons.min()) - dx / 2
    return Affine(dx, 0, west, 0, -dy, north), flip


def regrid_to_target(values, src_transform, src_crs, grid, method="average"):
    """
    Resample a 2-D array onto the target grid.

    values         2-D float array; NaN = missing (cloud, bad quality, ...)
    src_transform  affine transform of `values`
    src_crs        CRS of `values`, e.g. "EPSG:4326"
    grid           dict from load_grid()
    method         "average"  - source finer than target (Sentinel-2, MODIS)
                   "bilinear" - source coarser than target (ERA5, GFS, TROPOMI)
                   "nearest"  - categorical data (land-cover classes)
    """
    out = np.full(grid["shape"], np.nan, dtype="float32")
    reproject(
        source=np.asarray(values, dtype="float32"),
        destination=out,
        src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=grid["transform"], dst_crs=grid["crs"], dst_nodata=np.nan,
        resampling=getattr(Resampling, method),
    )
    return out


def zonal_stats(values, grid):
    """
    Weighted per-district statistics of a 2-D array that is ON the target grid.

    Each cell counts in proportion to how much of it lies in the district
    (the fraction weights from step 3). NaN cells are ignored.

    Returns a DataFrame with one row per district:
      mean, std, min, max  - of the valid cells
      n_valid_cells        - number of valid cells used
      valid_fraction       - share of the district's area that had valid data
                             (1.0 = fully clear, 0.2 = mostly cloud).
                             Phase 2 uses this to decide when to gap-fill.
    """
    values = np.asarray(values, dtype="float32")
    valid = np.isfinite(values)
    rows = []
    for k, (d, name) in enumerate(zip(grid["ids"], grid["names"])):
        w = grid["weights"][k]
        inside = w > 0
        wv = np.where(valid, w, 0.0)
        total_w = w.sum()
        valid_w = wv.sum()
        if valid_w > 0:
            x = np.where(valid, values, 0.0)
            mean = float((wv * x).sum() / valid_w)
            std = float(np.sqrt((wv * (x - mean) ** 2).sum() / valid_w))
            sel = values[inside & valid]
            vmin, vmax = float(sel.min()), float(sel.max())
        else:
            mean = std = vmin = vmax = np.nan
        rows.append({
            "district_id": d, "district": name,
            "mean": mean, "std": std, "min": vmin, "max": vmax,
            "n_valid_cells": int((inside & valid).sum()),
            "valid_fraction": float(valid_w / total_w) if total_w > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def zonal_timeseries(cube, grid, extra_weight=None):
    """
    Fast weighted district means for a whole (days, rows, cols) cube.

    extra_weight: optional (rows, cols) array multiplied into the district
                  weights - e.g. the built-up fraction, so that the average
                  focuses on where people live instead of empty desert.

    Returns three (days, districts) arrays:
      mean, std       over valid cells
      valid_fraction  share of the (weighted) district area that had data
    """
    days = cube.shape[0]
    w = grid["weights"].reshape(len(grid["ids"]), -1).astype("float64")
    if extra_weight is not None:
        w = w * np.nan_to_num(extra_weight.reshape(1, -1), nan=0.0)
    x = cube.reshape(days, -1).astype("float64")
    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)

    valid_w = valid.astype("float64") @ w.T
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (x0 @ w.T) / valid_w
        mean_sq = ((x0 ** 2) @ w.T) / valid_w
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
        valid_fraction = valid_w / w.sum(axis=1)
    return mean, std, valid_fraction


def sample_at_points(values, lats, lons, points_lat, points_lon):
    """
    Bilinear interpolation of a regular lat/lon grid (ERA5, GFS) at points
    such as the district centroids. Returns one value per point.
    """
    lats, lons = np.asarray(lats), np.asarray(lons)
    values = np.asarray(values)
    if lats[0] > lats[-1]:          # interpolator needs ascending axes
        lats, values = lats[::-1], values[::-1, :]
    interp = RegularGridInterpolator((lats, lons), values, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([points_lat, points_lon]))
