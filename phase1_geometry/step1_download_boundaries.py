"""
PHASE 1 - STEP 1: Download the official district boundaries of Pakistan.

What it does
  1. Asks the HDX (Humanitarian Data Exchange) website where the latest
     Pakistan boundary file is.
  2. Downloads the zip (~30 MB) into data/raw/boundaries/.
  3. Unzips it and checks that all six Karachi districts are inside.

Run it from the project folder:
    python phase1_geometry/step1_download_boundaries.py

If the download keeps failing (firewall, slow internet), download the file
by hand from https://data.humdata.org/dataset/cod-ab-pak
("pak_admin_boundaries.geojson.zip"), put it in data/raw/boundaries/,
and run this script again. It will skip the download.
"""
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

# Lets this script find config.py, which lives one folder up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def find_download_url():
    """Ask the HDX API for the current URL of the GeoJSON zip."""
    try:
        response = requests.get(config.HDX_API_URL, timeout=60)
        response.raise_for_status()
        for resource in response.json()["result"]["resources"]:
            if resource["name"] == config.HDX_ZIP_NAME:
                return resource["url"]
        print("  HDX did not list the GeoJSON zip; using the fallback URL.")
    except Exception as err:
        print(f"  Could not reach the HDX API ({err}); using the fallback URL.")
    return config.HDX_FALLBACK_URL


def download(url, destination):
    """Download a file in chunks and show progress."""
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()


def main():
    config.RAW_BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_BOUNDARY_DIR / config.HDX_ZIP_NAME
    adm2_path = config.RAW_BOUNDARY_DIR / config.HDX_ADM2_FILE

    # 1. Download (skipped if the zip is already there)
    if zip_path.exists():
        print(f"[1/3] Zip already present: {zip_path}")
    else:
        print("[1/3] Downloading Pakistan administrative boundaries from HDX ...")
        url = find_download_url()
        print(f"  URL: {url}")
        download(url, zip_path)

    # 2. Unzip
    print("[2/3] Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.RAW_BOUNDARY_DIR)
    if not adm2_path.exists():
        sys.exit(f"ERROR: {config.HDX_ADM2_FILE} not found inside the zip.")

    # 3. Check the six districts are present
    print("[3/3] Checking the six Karachi districts ...")
    adm2 = gpd.read_file(adm2_path)
    karachi = adm2[adm2["adm2_pcode"].isin(config.DISTRICTS)]
    print(karachi[["adm2_pcode", "adm2_name", "adm1_name", "area_sqkm"]].to_string(index=False))

    missing = set(config.DISTRICTS) - set(karachi["adm2_pcode"])
    if missing:
        sys.exit(f"ERROR: districts missing from the file: {missing}")
    print(f"\nOK - all 6 districts found. Raw file: {adm2_path}")


if __name__ == "__main__":
    main()
