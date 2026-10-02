"""
gapfill.py — fill the holes that clouds and orbit gaps leave in daily grids.

Two stages, applied to a daily cube of shape (days, rows, cols):

  1. SPATIAL - Inverse Distance Weighting (IDW)
     A missing cell gets a weighted average of the nearest clear cells on the
     SAME day. Closer cells count more (weight = 1 / distance^2). Only clear
     cells within `radius_m` are used, so a cell under a big cloud stays
     missing instead of being guessed from far away.

  2. TEMPORAL - 3-day rolling median
     Cells still missing get the median of the same cell on nearby days.
     Only real observations and IDW values are used as donors (never values
     that were themselves filled in time), so guesses do not snowball.

     mode="backward" : window = t-2, t-1, t   (uses only the past)
     mode="centered" : window = t-1, t, t+1   (uses tomorrow's data)

     Use "backward" for model training. On a real forecast day, tomorrow's
     satellite pass does not exist yet. If training features were filled with
     future data, the model would look better in testing than it really is
     (data leakage - flaw #2 in your research table).

Every cell also gets a flag saying where its value came from (FLAG_* below).
"""
import warnings

import numpy as np
from scipy.spatial import cKDTree

FLAG_OBSERVED = 0   # real satellite value
FLAG_IDW = 1        # filled from neighbours on the same day
FLAG_TEMPORAL = 2   # filled from nearby days
FLAG_MISSING = 3    # could not be filled
FLAG_OUTSIDE = 9    # cell is outside all districts (not filled, not needed)


def idw_fill_day(grid, targets, res_m, radius_m, k=8, power=2, min_neighbors=3):
    """
    Fill NaN cells of one 2-D day grid by IDW.
    targets: boolean grid, True where a value is wanted (inside districts).
    Returns (filled_grid, was_filled_boolean_grid).
    """
    out = grid.copy()
    was_filled = np.zeros(grid.shape, dtype=bool)
    valid = np.isfinite(grid)
    need = targets & ~valid
    if valid.sum() < min_neighbors or not need.any():
        return out, was_filled

    rows, cols = np.indices(grid.shape)
    tree = cKDTree(np.column_stack([rows[valid], cols[valid]]) * res_m)
    values = grid[valid]
    k = min(k, len(values))
    query = np.column_stack([rows[need], cols[need]]) * res_m
    dist, idx = tree.query(query, k=k, distance_upper_bound=radius_m)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]

    found = np.isfinite(dist)                  # neighbours inside the radius
    idx = np.where(found, idx, 0)              # dummy index for "not found"
    weights = np.where(found, 1.0 / np.maximum(dist, 1e-6) ** power, 0.0)
    enough = found.sum(axis=1) >= min_neighbors
    estimate = (weights * values[idx]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)

    need_rows, need_cols = rows[need], cols[need]
    out[need_rows[enough], need_cols[enough]] = estimate[enough]
    was_filled[need_rows[enough], need_cols[enough]] = True
    return out, was_filled


def gap_fill_cube(cube, targets, res_m, radius_m, mode="backward", window=3):
    """
    Spatial IDW, then temporal median, on a (days, rows, cols) cube.
    Returns (filled_cube, flags) - flags is int8 with the FLAG_* codes above.
    """
    days = cube.shape[0]
    filled = cube.copy()
    flags = np.full(cube.shape, FLAG_MISSING, dtype="int8")
    flags[np.isfinite(cube)] = FLAG_OBSERVED

    # 1. Spatial IDW, day by day
    for t in range(days):
        filled[t], was_filled = idw_fill_day(cube[t], targets, res_m, radius_m)
        flags[t][was_filled] = FLAG_IDW

    # 2. Temporal rolling median
    donors = np.where(flags <= FLAG_IDW, filled, np.nan)
    half = window // 2
    for t in range(days):
        if mode == "backward":
            window_days = range(t - (window - 1), t)
        elif mode == "centered":
            window_days = [s for s in range(t - half, t + half + 1) if s != t]
        else:
            raise ValueError("mode must be 'backward' or 'centered'")
        window_days = [s for s in window_days if 0 <= s < days]
        if not window_days:
            continue
        with warnings.catch_warnings():          # all-NaN cells are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(donors[window_days], axis=0)
        fill = targets & np.isnan(filled[t]) & np.isfinite(median)
        filled[t][fill] = median[fill]
        flags[t][fill] = FLAG_TEMPORAL

    flags[:, ~targets] = FLAG_OUTSIDE
    return filled, flags
