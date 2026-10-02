"""
cache_io.py — crash-safe reading and writing of every file Phase 2 produces.

Why this exists
    np.savez_compressed(final_path, ...) writes straight into the final file.
    If the script is stopped (Ctrl+C, closed window, power cut) half-way, a
    truncated file stays behind and the next run finds an "unreadable" file.

What it does instead
    1. atomic_write()   writes to a temporary file next to the target, forces
                        it to disk, and only then renames it over the target.
                        A rename is all-or-nothing, so the target is always
                        either the complete old file or the complete new file.
    2. save/load_month_cache()   the month cache used by daily_pipeline.py;
                        on load it checks the schema, the dates and the shape,
                        and quarantines (moves aside) any file that fails.
    3. pipeline_lock()  stops two Phase 2 processes from running at once and
                        overwriting each other's files.
    4. clean_stale_tmp() removes temporary files left by an interrupted run.
"""
import contextlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

CACHE_SCHEMA = 2          # bump when the cache layout OR the product filters change -> files are re-downloaded
LEGACY_SCHEMA = -1        # files written before schema numbers existed (same filters, same layout)
TMP_MARK = ".tmp"         # temporary files look like  name.tmp1234.npz


# ----------------------------------------------------------------------------
# Atomic writing
# ----------------------------------------------------------------------------
def _tmp_path(path):
    return path.with_name(f"{path.stem}{TMP_MARK}{os.getpid()}{path.suffix}")


def atomic_write(path, writer):
    """
    Call writer(tmp_path) to create the file, then move it onto `path`.
    `writer` must write a file at the path it is given (the temp file keeps the
    same extension as `path`, which numpy / xarray / rasterio need).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        writer(tmp)
        with open(tmp, "r+b") as f:              # force the bytes onto the disk
            os.fsync(f.fileno())
        for attempt in range(10):                # Windows: antivirus/indexer may hold the file briefly
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def atomic_csv(df, path, **kwargs):
    atomic_write(path, lambda p: df.to_csv(p, index=False, **kwargs))


def atomic_netcdf(ds, path, **kwargs):
    atomic_write(path, lambda p: ds.to_netcdf(p, engine="netcdf4", **kwargs))


def clean_stale_tmp(*roots):
    """Delete leftover temporary files from interrupted runs. Returns how many."""
    removed = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for f in root.rglob(f"*{TMP_MARK}[0-9]*"):
            if f.is_file():
                with contextlib.suppress(OSError):
                    f.unlink()
                    removed += 1
    return removed


# ----------------------------------------------------------------------------
# Month cache (daily grids downloaded from Earth Engine)
# ----------------------------------------------------------------------------
def quarantine(path, reason):
    """Move a bad cache file to data/cache/gee/_quarantine/ instead of deleting it."""
    path = Path(path)
    dest_dir = config.GEE_CACHE_DIR / "_quarantine"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{path.parent.name}__{path.name}"
    with contextlib.suppress(OSError):
        shutil.move(str(path), str(dest))
    print(f"    {path.name}: {reason} -> moved to _quarantine, downloading again")


def save_month_cache(path, dates, cube):
    dates = pd.DatetimeIndex(dates)
    text_dates = dates.strftime("%Y-%m-%d").to_numpy().astype("U10")   # plain text, no pickling
    atomic_write(path, lambda p: np.savez_compressed(
        p, schema=np.int16(CACHE_SCHEMA), dates=text_dates, cube=np.asarray(cube, dtype="float32")))


def load_month_cache(path, expected_dates, grid_shape):
    """
    Return (dates, cube) from a cache file, or None if it is missing or fails
    ANY check (unreadable zip, wrong schema, wrong days, wrong grid size,
    non-numeric values). A failed file is quarantined so it cannot be reused.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh, np.load(fh, allow_pickle=False) as z:   # our own handle: always closed
            schema = int(z["schema"]) if "schema" in z.files else -1
            dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
            cube = z["cube"]                      # reading also verifies the zip CRC
    except Exception as err:                       # truncated / corrupt / old layout
        quarantine(path, f"unreadable ({type(err).__name__})")
        return None
    problem = None
    if schema not in (CACHE_SCHEMA, LEGACY_SCHEMA):
        problem = f"old cache layout (schema {schema})"
    elif not dates.equals(pd.DatetimeIndex(expected_dates)):
        problem = "dates do not match the month"
    elif cube.shape != (len(dates), *grid_shape) or cube.dtype != np.float32:
        problem = f"unexpected array {cube.shape} {cube.dtype}"
    elif np.isinf(cube).any():
        problem = "contains infinite values"
    if problem:
        quarantine(path, problem)
        return None
    if schema == LEGACY_SCHEMA:        # valid file from before schema numbers existed: stamp it
        save_month_cache(path, dates, cube)
    return dates, cube


# ----------------------------------------------------------------------------
# Only one pipeline process at a time
# ----------------------------------------------------------------------------
_lock_depth = 0


@contextlib.contextmanager
def pipeline_lock():
    """
    Hold an exclusive OS lock on data/.phase2.lock. The lock disappears by
    itself if the process dies, so there is never a stale lock to clean up.
    Re-entrant: steps run inside run_phase2.py (same process) may lock again.
    """
    global _lock_depth
    if _lock_depth:
        _lock_depth += 1
        try:
            yield
        finally:
            _lock_depth -= 1
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(config.DATA_DIR / ".phase2.lock", "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit("Another Phase 2 process is already running and writing the same files.\n"
                         "Wait for it to finish (or close it), then run this again.")
    _lock_depth = 1
    try:
        yield
    finally:
        _lock_depth = 0
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()
