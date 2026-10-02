"""
products.py — the DAILY satellite products and their quality filters.

Each product has:
    build_day(day, region) -> ee.Image with one band "v" = daily mean of all
                              good-quality pixels (masked where none)
    units, idw_radius_km   -> used by the gap-filling pipeline

Everything in build_day runs on Google's servers, not on your PC.
"""
import ee

from gee_utils import mean_or_empty

# ----------------------------------------------------------------------------
# Sentinel-5P TROPOMI (gases)
# ----------------------------------------------------------------------------
# Earth Engine hosts TROPOMI as "Level-3": the Level-2 swaths already
# filtered by qa_value and gridded at ~1.1 km. Filters applied by Google at
# ingestion (from the Earth Engine catalog):
#   NO2 (tropospheric column): qa_value >= 0.75   -> exactly our rule
#   O3                       : qa_value >= 0.50 plus physical range checks
#   CO                       : qa_value >= 0.50
#   AER_AI                   : qa_value >= 0.80
# The qa_value band itself is NOT included, so we cannot raise O3 or SO2 to
# 0.75 directly. Instead we also drop pixels with cloud_fraction > 0.3, which
# removes the cloudy scenes that a stricter qa_value would remove.
# Pixels below qa 0.5 are never present, so heavy-cloud pixels (qa < 0.5)
# are always treated as missing, as the plan requires.
#
# Two streams exist:
#   OFFL - final, best quality, arrives ~5-14 days late  (used when present)
#   NRTI - near-real-time, arrives within ~3 hours       (used otherwise)
# so recent days still get data for operational forecasting.
S5P = {
    "no2": {"collection": "L3_NO2", "band": "tropospheric_NO2_column_number_density",
            "scale": 1e6, "units": "umol/m2", "cloud_max": None, "idw_radius_km": 12,
            "description": "Tropospheric NO2 column"},
    "o3": {"collection": "L3_O3", "band": "O3_column_number_density",
           "scale": 2241.15, "units": "DU", "cloud_max": 0.3, "idw_radius_km": 15,
           "description": "Total O3 column (Dobson Units)"},
    "so2": {"collection": "L3_SO2", "band": "SO2_column_number_density",
            "scale": 1e6, "units": "umol/m2", "cloud_max": 0.3, "idw_radius_km": 12,
            "description": "SO2 column (noisy: values below 0 are normal)"},
    "co": {"collection": "L3_CO", "band": "CO_column_number_density",
           "scale": 1e3, "units": "mmol/m2", "cloud_max": None, "idw_radius_km": 15,
           "description": "CO column"},
    "aer_ai": {"collection": "L3_AER_AI", "band": "absorbing_aerosol_index",
               "scale": 1.0, "units": "-", "cloud_max": None, "idw_radius_km": 15,
               "description": "UV absorbing aerosol index (dust/smoke)"},
}


def _s5p_builder(key):
    p = S5P[key]

    def prep(img):
        value = img.select(p["band"]).multiply(p["scale"])
        if p["cloud_max"] is not None:
            value = value.updateMask(img.select("cloud_fraction").lte(p["cloud_max"]))
        return value.rename("v")

    def build_day(day, region):
        end = day.advance(1, "day")
        offl = (ee.ImageCollection(f"COPERNICUS/S5P/OFFL/{p['collection']}")
                .filterBounds(region).filterDate(day, end))
        nrti = (ee.ImageCollection(f"COPERNICUS/S5P/NRTI/{p['collection']}")
                .filterBounds(region).filterDate(day, end))
        chosen = ee.ImageCollection(ee.Algorithms.If(offl.size().gt(0), offl, nrti))
        # Several orbits can cross Karachi in one day; average them per pixel.
        return mean_or_empty(chosen.map(prep))

    return build_day


# ----------------------------------------------------------------------------
# MODIS MAIAC AOD (MCD19A2, Terra + Aqua, 1 km)
# ----------------------------------------------------------------------------
# AOD_QA is a 16-bit number; groups of bits mean different things
# (Earth Engine catalog, MCD19A2.061):
#   bits 0-2  cloud mask       keep 1 = clear
#   bits 3-4  surface          keep 0 = land (drops sea/sediment glare)
#   bits 5-7  adjacency        keep 0 = not next to a cloud
#   bits 8-11 AOD quality      keep 0 = best quality
#   bit  12   glint            keep 0 = no sun glint
# `(qa >> start) & mask` reads one group of bits.
def _bits(image, start, width):
    return image.rightShift(start).bitwiseAnd((1 << width) - 1)


def _maiac_prep(img):
    qa = img.select("AOD_QA")
    good = (_bits(qa, 0, 3).eq(1)
            .And(_bits(qa, 3, 2).eq(0))
            .And(_bits(qa, 5, 3).eq(0))
            .And(_bits(qa, 8, 4).eq(0))
            .And(_bits(qa, 12, 1).eq(0)))
    return img.select("Optical_Depth_055").multiply(0.001).updateMask(good).rename("v")


def _maiac_build_day(day, region):
    granules = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
                .filterBounds(region).filterDate(day, day.advance(1, "day")))
    return mean_or_empty(granules.map(_maiac_prep))


# ----------------------------------------------------------------------------
# Registry used by the pipeline
# ----------------------------------------------------------------------------
PRODUCTS = {key: {**p, "build_day": _s5p_builder(key), "source": "Sentinel-5P"}
            for key, p in S5P.items()}
PRODUCTS["aod"] = {
    "build_day": _maiac_build_day, "source": "MODIS MAIAC",
    "units": "-", "idw_radius_km": 5,
    "description": "Aerosol optical depth at 550 nm (best-quality, clear-sky)",
}

# Physically valid range of a single pixel value, in the units above. Pixels outside it are
# retrieval artefacts and become "missing" (applied in daily_pipeline.fetch_month, so it
# works on cached months too). AOD: MAIAC's documented valid range is -0.1 .. 5.0, yet
# pixels up to 11 pass the QA bits (verified: e.g. 2022-06-17, 2026-08-23, median 5.8 over
# 5800 pixels), which would put district means at 6-7. The other limits are only guards.
VALID_RANGE = {"aod": (-0.1, 5.0), "no2": (-100, 3000), "o3": (100, 600),
               "so2": (-1000, 45000), "co": (0, 150), "aer_ai": (-20, 20)}
for _key, _range in VALID_RANGE.items():
    PRODUCTS[_key]["valid_range"] = _range

S5P_KEYS = list(S5P)
MODIS_KEYS = ["aod"]
