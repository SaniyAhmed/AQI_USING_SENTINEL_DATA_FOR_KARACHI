"""
gee_utils.py — talking to Google Earth Engine (GEE).

The "zero-local-storage" idea: raw satellite files never come to your PC.
Earth Engine opens them on Google's servers, applies our quality masks,
resamples them onto OUR Phase 1 grid, and sends back only the finished small
array (a 1 km grid of Karachi is 121 x 104 numbers, ~50 KB per day).

Main functions
    init_ee()              log in to Earth Engine (with a request timeout)
    region()               the Phase 1 download bounding box as an ee.Geometry
    fetch_array(image, m)  compute an ee.Image on our grid, return numpy array
    annual_chunks(s, e)    split a date range into 1-year pieces
"""
import json
import os
import time

import ee
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import config


def init_ee():
    """Initialise Earth Engine with the project id from credentials.env.

    Headless use (the daily GitHub Actions forecast): if GEE_SERVICE_ACCOUNT_JSON is set (the full
    service-account key JSON, as an environment variable / GitHub secret - never committed), it is
    used directly and no browser is needed. Local/interactive use is unchanged: ee.Authenticate()
    opens a browser once and caches the OAuth token under ~/.config/earthengine.
    """
    load_dotenv(config.CREDENTIALS_FILE)
    project = os.getenv("GEE_PROJECT")
    if not project:
        raise SystemExit(
            "GEE_PROJECT is missing. Add a line like\n"
            "    GEE_PROJECT=your-cloud-project-id\n"
            f"to {config.CREDENTIALS_FILE} (see the Phase 2 guide)."
        )
    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        info = json.loads(sa_json)
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_json)
        ee.Initialize(credentials, project=project)
        return
    for attempt in range(6):
        try:
            ee.Initialize(project=project)
            break
        except Exception as err:
            if "credentials" in str(err).lower() or "authenticate" in str(err).lower():
                print("Earth Engine login needed - a browser window will open.")
                ee.Authenticate()
                ee.Initialize(project=project)
                break
            # A dropped connection while starting up (SSL/EOF, DNS, timeout) is temporary.
            if attempt == 5 or not (isinstance(err, (ConnectionError, TimeoutError, OSError))
                                    or _is_temporary(err)):
                raise
            wait = 10 * 2 ** attempt
            print(f"Earth Engine connection problem ({type(err).__name__}); retrying in {wait}s ...")
            time.sleep(wait)
    # Every Earth Engine HTTP request now gives up after GEE_HTTP_TIMEOUT_S
    # seconds instead of waiting forever on a dead connection. A timed-out
    # request raises TimeoutError, which fetch_array() retries.
    ee.data.setDeadline(config.GEE_HTTP_TIMEOUT_S * 1000)
    return project


def annual_chunks(start, end, years=None):
    """
    Split [start, end] into consecutive pieces of `years` calendar years
    (default config.CHUNK_YEARS). Returns a list of (chunk_start, chunk_end)
    date strings, e.g. 2022-03-15 -> 2023-06-30 with years=1 gives
        [("2022-03-15", "2022-12-31"), ("2023-01-01", "2023-06-30")]
    """
    years = years or config.CHUNK_YEARS
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(pd.Timestamp(year=chunk_start.year + years - 1, month=12, day=31), end)
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return chunks


def region():
    """Download bounding box from Phase 1 (master_bbox.json) as an ee.Geometry."""
    with open(config.BOUNDARY_DIR / "master_bbox.json") as f:
        b = json.load(f)["download_bbox"]
    return ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])


def ee_grid(resolution_m):
    """Describe one of our Phase 1 grids in the format Earth Engine expects."""
    with open(config.GRID_DIR / "grid_definition.json") as f:
        grid_def = json.load(f)
    g = grid_def["grids"][str(resolution_m)]
    a, b, c, d, e, f_ = g["transform"]
    return {
        "dimensions": {"width": g["width"], "height": g["height"]},
        "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                            "shearY": d, "scaleY": e, "translateY": f_},
        "crsCode": grid_def["crs"],
    }


def fetch_array(image, resolution_m, retries=5):
    """
    Compute `image` on Earth Engine, resampled onto our grid, and download
    the result as a numpy array of shape (bands, rows, cols).
    Masked pixels (cloud, bad quality, no overpass) come back as NaN.
    """
    request = {
        # Masked pixels cannot be sent as "empty", so they are replaced with
        # the NODATA number on the server and turned into NaN here.
        "expression": image.toFloat().unmask(config.NODATA, sameFootprint=False),
        "fileFormat": "NUMPY_NDARRAY",
        # `resolution_m` is normally 100 or 1000 (our Phase 1 grids); Phase 3 passes a
        # ready-made grid dict to read a source's own native lattice (ERA5, CAMS ...).
        "grid": resolution_m if isinstance(resolution_m, dict) else ee_grid(resolution_m),
    }
    for attempt in range(retries):
        try:
            result = ee.data.computePixels(request)
            break
        except Exception as err:
            # Retry only temporary problems (network, rate limits, busy servers).
            # A mistake in the request itself fails the same way every time.
            if attempt == retries - 1 or not _is_temporary(err):
                raise
            wait = 15 * 2 ** attempt
            print(f"    Earth Engine error ({str(err)[:120]}); retrying in {wait}s ...")
            time.sleep(wait)
    names = list(result.dtype.names)
    cube = np.stack([result[n] for n in names]).astype("float32")
    cube[cube == config.NODATA] = np.nan
    return cube, names


TEMPORARY_ERRORS = ("timed out", "timeout", "too many", "rate", "quota", "429", "500", "502",
                    "503", "504", "connection", "max retries", "unavailable", "internal error")


def _is_temporary(err):
    if isinstance(err, TimeoutError):          # raised by the request timeout
        return True
    return any(word in str(err).lower() for word in TEMPORARY_ERRORS)


def empty_image(band="v"):
    """A fully masked image, used for days with no satellite overpass."""
    return ee.Image.constant(0).toFloat().rename(band).updateMask(0)


def mean_or_empty(collection, band="v"):
    """Per-pixel mean of a collection, or a fully masked image if it is empty."""
    return ee.Image(ee.Algorithms.If(collection.size().gt(0),
                                     collection.mean().rename(band),
                                     empty_image(band)))
