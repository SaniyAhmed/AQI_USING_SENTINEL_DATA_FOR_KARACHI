"""
PHASE 4 - STEP 2: Download the HOURLY ground measurements of every usable sensor (OpenAQ v3).

For each sensor in data/raw/ground/sensors.csv (from its station's first data, but not before
config.START_DATE - 20 days) the hourly averages are saved as
    data/raw/ground/hourly/sensor_<id>.parquet     hour_utc, value, observed_count, expected_count, has_flags

Resumable and incremental: a sensor that is already saved is only asked for hours from 48 h before
its last saved hour (recent values can be revised), then merged. Files are written atomically.
The free plan allows 60 calls a minute, so a full first run takes a few minutes.

Run:  python phase4_ground/step2_download.py
"""
import datetime as dt
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
import config
import openaq
from cache_io import atomic_write, pipeline_lock

GROUND_DIR = config.RAW_DIR / "ground"
HOURLY_DIR = GROUND_DIR / "hourly"
WINDOW_DAYS = 30
FIRST_DAY = pd.Timestamp(config.START_DATE) - pd.Timedelta(days=20)      # a few weeks of history for lags


def parse(results):
    rows = []
    for r in results:
        cov = r.get("coverage") or {}
        rows.append({"hour_utc": pd.Timestamp(r["period"]["datetimeFrom"]["utc"]).tz_localize(None),
                     "value": r["value"], "observed_count": cov.get("observedCount"),
                     "expected_count": cov.get("expectedCount"),
                     "has_flags": bool((r.get("flagInfo") or {}).get("hasFlags"))})
    return pd.DataFrame(rows)


def download_sensor(sensor_id, first_utc, now):
    path = HOURLY_DIR / f"sensor_{sensor_id}.parquet"
    start = max(FIRST_DAY, pd.Timestamp(first_utc).tz_localize(None).normalize()) if first_utc else FIRST_DAY
    old = None
    if path.exists():
        try:
            old = pd.read_parquet(path)
            start = max(start, old["hour_utc"].max() - pd.Timedelta(hours=48))
        except Exception:
            old = None                                        # unreadable file: download everything again
    # 30-day windows (<= 720 hours = one page): long windows make the API time out (HTTP 408)
    frames = []
    for w0 in pd.date_range(start, now, freq=f"{WINDOW_DAYS}D"):
        w1 = min(w0 + pd.Timedelta(days=WINDOW_DAYS), now + pd.Timedelta(hours=1))
        j = openaq.get(f"/sensors/{sensor_id}/hours", {
            "datetime_from": w0.strftime("%Y-%m-%dT%H:%M:%SZ"), "datetime_to": w1.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 1000})
        res = j.get("results", [])
        if len(res) >= 1000:
            raise RuntimeError(f"sensor {sensor_id}: window {w0} returned a full page - shorten WINDOW_DAYS")
        if res:
            frames.append(parse(res))
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["hour_utc", "value", "observed_count", "expected_count", "has_flags"])
    df = pd.concat([old[old["hour_utc"] < start], new], ignore_index=True) if old is not None else new
    df = df.drop_duplicates("hour_utc", keep="last").sort_values("hour_utc").reset_index(drop=True)
    atomic_write(path, lambda p: df.to_parquet(p, index=False))
    return len(new), len(df)


def download_sensor_daily(sensor_id, first_utc, now):
    """
    Fallback for sensors whose /hours endpoint fails on OpenAQ's side (HTTP 500 for some newer sensors):
    /hours/daily gives the Karachi-local-day mean with the number of hours behind it. Saved as
    daily_sensor_<id>.parquet: date, value, observed_count, expected_count. No hourly flat-line check is possible.
    """
    start = max(FIRST_DAY, pd.Timestamp(first_utc).tz_localize(None).normalize()) if first_utc else FIRST_DAY
    rows = []
    for w0 in pd.date_range(start, now, freq="300D"):
        w1 = min(w0 + pd.Timedelta(days=300), now + pd.Timedelta(days=1))
        j = openaq.get(f"/sensors/{sensor_id}/hours/daily", {"datetime_from": w0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                             "datetime_to": w1.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 1000})
        for r in j.get("results", []):
            cov = r.get("coverage") or {}
            rows.append({"date": pd.Timestamp(r["period"]["datetimeFrom"]["local"][:10]), "value": r["value"],
                         "observed_count": cov.get("observedCount"), "expected_count": cov.get("expectedCount"),
                         "has_flags": bool((r.get("flagInfo") or {}).get("hasFlags"))})
    df = pd.DataFrame(rows, columns=["date", "value", "observed_count", "expected_count", "has_flags"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    atomic_write(HOURLY_DIR / f"daily_sensor_{sensor_id}.parquet", lambda p: df.to_parquet(p, index=False))
    return len(df)


def main():
    sensors = pd.read_csv(GROUND_DIR / "sensors.csv")
    stations = pd.read_csv(GROUND_DIR / "stations.csv")
    usable = stations[stations["district_id"].notna()][["location_id", "first_utc"]]
    sensors = sensors.merge(usable, on="location_id")            # only sensors of stations inside a district
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(dt.datetime.now(dt.timezone.utc)).tz_localize(None).floor("h")
    print(f"{len(sensors)} sensors to download up to {now:%Y-%m-%d %H:%M} UTC")
    # The API answers slowly (seconds per call) but the 60-calls-a-minute limit is enforced in openaq.get(),
    # so several sensors can be downloaded at once.
    failed = []

    def job(r):
        try:
            n_new, n_all = download_sensor(int(r.sensor_id), r.first_utc, now)
            print(f"  sensor {r.sensor_id} ({r.parameter}, station {r.location_id}): +{n_new} hours, {n_all} saved", flush=True)
        except Exception as err:
            try:                                       # the hourly endpoint is broken for this sensor: use daily means
                n = download_sensor_daily(int(r.sensor_id), r.first_utc, now)
                print(f"  sensor {r.sensor_id} ({r.parameter}, station {r.location_id}): hourly endpoint failed "
                      f"({str(err)[:40]}); daily fallback saved {n} days", flush=True)
            except Exception as err2:                  # one bad sensor must not stop the others
                failed.append(int(r.sensor_id))
                print(f"  sensor {r.sensor_id}: FAILED ({str(err2)[:100]}) - run again to retry", flush=True)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(job, list(sensors.itertuples())))
    if failed:
        raise SystemExit(f"{len(failed)} sensors failed: {failed}. Run this step again; finished sensors are not downloaded twice.")
    print("done")


if __name__ == "__main__":
    with pipeline_lock():
        main()
