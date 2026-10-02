"""
config.py — settings shared by every script in the project.

Every script does `import config`, so change a setting here once and all
scripts pick it up.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders (created automatically by the scripts)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_BOUNDARY_DIR = RAW_DIR / "boundaries"
BOUNDARY_DIR = PROCESSED_DIR / "boundaries"
GRID_DIR = PROCESSED_DIR / "grid"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# ----------------------------------------------------------------------------
# Coordinate Reference Systems
# ----------------------------------------------------------------------------
# WGS84 latitude/longitude (degrees). Satellite products are delivered in it.
CRS_WGS84 = "EPSG:4326"
# UTM Zone 42N (metres). Karachi (~67 E) lies inside this zone (66 E - 72 E),
# so distances and areas measured in it are accurate.
CRS_UTM = "EPSG:32642"

# ----------------------------------------------------------------------------
# District definitions
# ----------------------------------------------------------------------------
# Source: UN OCHA "Common Operational Dataset - Administrative Boundaries"
# for Pakistan (HDX dataset id "cod-ab-pak"). It is the only free source
# found that has exactly these six districts (GADM has no Korangi).
# The key is the official P-code; the value is (our district id, our name).
# District id 0 is reserved for "outside all districts".
DISTRICTS = {
    "PK702": (1, "Karachi Central"),
    "PK704": (2, "Karachi East"),
    "PK721": (3, "Karachi South"),
    "PK729": (4, "Karachi West"),
    "PK712": (5, "Korangi"),
    "PK714": (6, "Malir"),
}

HDX_DATASET_ID = "cod-ab-pak"
HDX_API_URL = f"https://data.humdata.org/api/3/action/package_show?id={HDX_DATASET_ID}"
# Used only if the HDX API lookup fails (the URL can change between releases).
HDX_FALLBACK_URL = (
    "https://data.humdata.org/dataset/a64d1ff2-7158-48c7-887d-6af69ce21906/"
    "resource/c6521e04-75e1-41be-8e02-0554a424d9f4/download/pak_admin_boundaries.geojson.zip"
)
HDX_ZIP_NAME = "pak_admin_boundaries.geojson.zip"
HDX_ADM2_FILE = "pak_admin2.geojson"   # ADM2 = district level in Pakistan

# ----------------------------------------------------------------------------
# Bounding box and target grids
# ----------------------------------------------------------------------------
# The box proposed in the project plan. We keep it only to compare against the
# real district extent: Malir (Gadap) and Karachi West reach past it.
PROPOSED_BBOX_WGS84 = {"west": 66.8, "south": 24.7, "east": 67.5, "north": 25.5}

# Extra margin added around the district polygons.
#   - degrees: used for the lat/lon box we will download satellite data for
#     in Phase 2 (0.10 deg ~ 11 km, so coarse ERA5/GFS cells at the edge
#     are still downloaded).
#   - metres: used for the UTM analysis grid.
BBOX_BUFFER_DEG = 0.10
GRID_BUFFER_M = 2_000

# Target grid resolutions in metres.
#   1000 m -> Sentinel-5P, MODIS MAIAC, ERA5, GFS are regridded here
#    100 m -> Sentinel-2 land-cover features (NDVI, NDBI, % urban)
GRID_RESOLUTIONS_M = [1000, 100]

# ============================================================================
# PHASE 2 - Satellite ingestion (Google Earth Engine) and gap filling
# ============================================================================
# credentials.env must contain:  GEE_PROJECT=your-cloud-project-id
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.env"

# Date range to process. END_DATE = None means "up to yesterday".
# 2022-01-01 gives ~4.7 years: enough seasons (4+ monsoons and winters) for
# training while keeping the run short. The earliest possible start is
# 2019-01-01 (Sentinel-5P O3 starts 2018-09, Sentinel-2 L2A over Pakistan
# starts 2018-12) if you later want more history.
START_DATE = "2022-01-01"
END_DATE = None

# run_phase2.py processes the date range in chunks of this many years
# (1 = one calendar year per chunk), so each chunk is a smaller, independent
# batch of Earth Engine requests and a failure only affects one year.
CHUNK_YEARS = 1

# Small cache of monthly arrays downloaded from Earth Engine (a few MB per
# product per year) so an interrupted run can resume without re-downloading.
GEE_CACHE_DIR = DATA_DIR / "cache" / "gee"
# A month is cached only once it is this many days in the past, because the
# final (OFFL) Sentinel-5P files arrive a few days to two weeks late.
CACHE_AFTER_DAYS = 21
# Months downloaded in parallel from Earth Engine. Daily 1 km requests take
# 2-4 s each, so 8 in parallel stays well inside Earth Engine's concurrent
# request quota. Sentinel-2 100 m requests are heavier (10-35 s): fewer.
GEE_WORKERS = 8
GEE_WORKERS_S2 = 3

# Maximum seconds one Earth Engine HTTP request may take before it is
# abandoned (and retried). Without this a dropped connection can hang forever.
# Measured: heaviest request (Sentinel-2 month at 100 m) ~ 10-35 s.
GEE_HTTP_TIMEOUT_S = 60

# Outputs
SATELLITE_DIR = PROCESSED_DIR / "satellite"          # gap-filled 1 km daily grids (NetCDF, per month)
DISTRICT_DAILY_DIR = PROCESSED_DIR / "district_daily"  # one CSV per daily product + landcover.csv
DISTRICT_MONTHLY_DIR = PROCESSED_DIR / "district_monthly"  # Sentinel-2 monthly CSV
BUILT_WEIGHT_FILE = GRID_DIR / "built_fraction_1000m.tif"  # made by Sentinel-2 step
PHASE2_TABLE = PROCESSED_DIR / "phase2_daily_features"     # final table (.csv and .parquet), made by step 7
VALIDATION_REPORT = PROJECT_ROOT / "outputs" / "phase2_validation_report.json"

# ----------------------------------------------------------------------------
# Leak-free (causal) settings - see step8_validate.py, which tests all of them
# ----------------------------------------------------------------------------
# Built-up weights ("urban_mean") come from Sentinel-2 / Dynamic World over this
# fixed reference period, which ENDS BEFORE START_DATE. So no day in the dataset
# is weighted with land-cover information from its own future, and re-running
# later never changes weights (and so never mixes old and new rows).
BUILT_WEIGHT_PERIOD = ("2021-01", "2021-12")

# Sentinel-2 monthly composites are only complete when the month is over, so a
# day in month M gets the composite of month M-1 (never its own month).
# A cell with too little clear sky in a month is filled from earlier months only.
S2_MIN_CELL_CLEAR = 0.20     # a 1 km cell needs >= 20 % clear 100 m pixels, else it counts as cloudy
S2_MAX_FILL_MONTHS = 2       # cloudy cells may be filled from at most this many months back
S2_WARMUP_MONTHS = 2         # composites fetched before START_DATE so the first days have data

# A daily district value is "direct" when at least this share of the district
# area has (observed or interpolated) data. Below it the mean is not trusted
# and `mean_complete` uses the trailing median instead (past days only).
MIN_VALID_FRACTION = 0.5
# Trailing windows (days, past only) for `mean_complete`; the second is the
# fallback when the first has no direct value (e.g. the July-August AOD gap).
COMPLETE_WINDOWS_DAYS = (30, 90)

# Rows younger than this may still change (final Sentinel-5P files arrive 5-14
# days late, MAIAC ~ 3 days). They are flagged `provisional` - do not train on them.
PROVISIONAL_DAYS = 14

# Value used to send "no data" through Earth Engine (turned into NaN locally).
NODATA = -9999.0

# Temporal gap filling: "backward" (past days only - use this for training)
# or "centered" (uses the next day too - NOT available when forecasting).
TEMPORAL_FILL_MODE = "backward"
TEMPORAL_FILL_WINDOW = 3


# ============================================================================
# PHASE 3 - Meteorology, model aerosol and the feature table
# ============================================================================
MET_DIR = PROCESSED_DIR / "met"                      # daily meteorology per district (ERA5, GFS, CAMS)
FEATURE_DIR = PROCESSED_DIR / "features"             # final feature tables
FORECAST_DIR = DATA_DIR / "forecast"                 # operational pulls (GFS runs are archived here)
FEATURE_TABLE = FEATURE_DIR / "phase3_features"      # .parquet / .csv
PHASE3_REPORT = PROJECT_ROOT / "outputs" / "phase3_validation_report.json"

# Karachi is UTC+5 all year (no daylight saving). A local day d covers UTC
# [d-1 19:00, d 19:00). Meteorology is aggregated over local days so that it
# lines up with daily ground AQI in Phase 4.
LOCAL_UTC_OFFSET_H = 5

# The forecast is ISSUED on the morning of day D (about 10:00 local = 05:00 UTC).
# At that moment, what exists (and what features may use):
#   * satellite data up to day D-1 (Sentinel-5P/MODIS overpasses of day D come later)
#   * NOAA GFS 00Z run of day D (ready ~ 04:00 UTC) -> meteorology for D, D+1, D+2
#   * CAMS 12Z run of day D-1 (ready ~ 22:00 UTC the evening before) -> model AOD/PM for D..D+2
#     (the CAMS 00Z run of day D is NOT ready until ~ 10:00 UTC, so it is never used)
HORIZONS = (0, 1, 2)
# The live/operational forecast (step5_gfs_forecast.py, predict_today.py, the webapp) only issues
# today + tomorrow. HORIZONS above (incl. T+2) is unchanged and still governs historical feature
# building, training and evaluation (Phases 3-5) - only the live-facing output is narrower.
LIVE_HORIZONS = (0, 1)
CAMS_INIT_HOUR = 12
# CAMS runs are read for lead days 0..3 (CAMS ends at hour 120, so lead 4 = hour 126 never exists) so that, if the newest run has not reached Earth Engine yet,
# the previous run (D-2) can stand in for it. Lead day k of run I is the local day I+1+k.
CAMS_LEADS = (0, 1, 2, 3)
# How many days old the newest usable CAMS 12Z run may be. Earth Engine's ingestion lag is normally
# 1 day but was observed 2 days deep on 2026-09-27 (both D-1 and D-2 runs missing), which left every
# cams_* feature NaN for that live forecast and made LightGBM/XGBoost diverge sharply (a run that
# stale had never happened during training). Raised from 2 to 4: CAMS_LEADS tops out at 3, so a T+0
# feature can fall back as far as age 4 (run D-4, lead 3) and a T+1 feature to age 4 (run D-4, lead 3)
# before running out of forecast lead to fall back on; a T+2 feature already maxed out its usable
# fallback depth at age 2 (run D-2, lead 3), so it is unaffected by this change either way.
CAMS_MAX_RUN_AGE_DAYS = 4
FEATURE_WARMUP_DAYS = 7          # first issue date = START_DATE + this (7-day rolling windows need history)

# Native lattices of the meteorological sources (node = grid-point centre, degrees).
# The boxes bracket the six district centroids with one extra cell all around.
# Values are read on the native lattice (no resampling) and interpolated to the
# centroids bilinearly by spatial_utils.sample_at_points.
ERA5_NODES = {"lon0": 66.50, "lat_top": 25.50, "step": 0.25, "n_lon": 6, "n_lat": 5}
CAMS_NODES = {"lon0": 66.40, "lat_top": 25.60, "step": 0.40, "n_lon": 4, "n_lat": 4}
# GFS/Open-Meteo nodes: only the 12 that bracket the centroids (fewer API calls).
GFS_NODE_LATS = (24.75, 25.00, 25.25)
GFS_NODE_LONS = (66.75, 67.00, 67.25, 67.50)

OPEN_METEO_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================================
# PHASE 5 - Model training, multi-horizon evaluation, AQI conversion
# ============================================================================
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_FILE = PROCESSED_DIR / "predictions"          # .parquet / .csv (OOF + final predictions)
PHASE5_REPORT = PROJECT_ROOT / "outputs" / "phase5_validation_report.json"

RANDOM_SEED = 42
N_CV_FOLDS = 5
PRIMARY_TARGET = "pm25_target"
WEIGHT_COL = "n_stations_target"

# District display names in district_id order (1..6) - used to build a fixed,
# stable one-hot encoding (fixed category order so it never depends on which
# districts happen to be present in a given fold).
DISTRICT_NAMES = [name for _, name in sorted(DISTRICTS.values())]

# Columns that must NEVER be used as a model input: identifiers, dates, the
# targets themselves, and target-side metadata (n_stations_target is real but
# only known for the TARGET day, so it is used as a sample weight, never a
# feature). district / district_id are replaced by a one-hot encoding built
# from DISTRICT_NAMES (see phase5_models/model_lib.py) instead of being fed
# in raw. met_source and cams_run_date carry no information as raw values
# (met_source is constant "era5" in the training table; cams_run_date is
# already captured numerically by cams_run_age_days).
NON_FEATURE_COLS = [
    "issue_date", "target_date", "horizon", "district", "district_id", "met_source", "cams_run_date",
    "pm25_target", "pm10_target", "n_stations_target", "aqi_pm25", "aqi_pm10", "aqi_target",
    "aqi_pollutants_used", "aqi_dominant", "target_available",
    "cv_fold",  # step1's own CV bookkeeping - a function of issue_date, undefined for a live issue date, never a feature
]
