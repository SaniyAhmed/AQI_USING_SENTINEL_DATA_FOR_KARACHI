"""
openaq.py - a small, polite client for the OpenAQ v3 API (https://docs.openaq.org).

The key is read from credentials.env (OPENAQ_API_KEY). The free plan allows 60 requests per minute,
so every call waits its turn (one call at most every 1.1 s) and retries on 429/5xx/network errors.
Nothing is ever printed that contains the key.
"""
import os
import threading
import time

import requests
from dotenv import load_dotenv

import config

BASE = "https://api.openaq.org/v3"
_lock = threading.Lock()
_last_call = [0.0]
MIN_GAP_S = 1.1


def _key():
    load_dotenv(config.CREDENTIALS_FILE)
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit(f"OPENAQ_API_KEY is missing in {config.CREDENTIALS_FILE}")
    return key.strip()


def get(path, params=None, tries=8):
    """GET one API page -> parsed JSON. Waits for the rate limit and retries temporary problems."""
    headers = {"X-API-Key": _key()}
    for attempt in range(tries):
        with _lock:                                            # one request at a time, spaced out
            wait = MIN_GAP_S - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=90)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:                               # over the limit: wait for the window to reset
            time.sleep(float(r.headers.get("x-ratelimit-reset", 30)) + 2)
            continue
        if r.status_code == 408 or r.status_code >= 500:            # timeout on the server side: try again
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        raise RuntimeError(f"OpenAQ {r.status_code} for {path}: {r.text[:200]}")
    raise RuntimeError(f"OpenAQ did not answer for {path} after {tries} tries")


def pages(path, params=None, limit=1000):
    """Yield every result of a paged endpoint."""
    page = 1
    while True:
        j = get(path, {**(params or {}), "limit": limit, "page": page})
        results = j.get("results", [])
        yield from results
        found = str(j.get("meta", {}).get("found", ""))
        if len(results) < limit or (found.isdigit() and page * limit >= int(found)):
            return
        page += 1
