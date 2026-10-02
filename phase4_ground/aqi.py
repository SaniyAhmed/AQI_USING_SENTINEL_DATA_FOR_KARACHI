"""
aqi.py - the US EPA piecewise-linear AQI (the formula on the project slide):

        I_p = (I_high - I_low) / (C_high - C_low) * (C_p - C_low) + I_low

and the overall AQI = max over the pollutants that have a value (dominant pollutant reported).

Breakpoints: EPA Technical Assistance Document for the AQI, PM2.5 as revised in May 2024
("epa_2024", the default); the pre-2024 PM2.5 table is kept as "epa_2012" for comparison.
Concentrations are TRUNCATED (not rounded) as EPA specifies before the lookup: PM2.5 to 0.1 ug/m3,
PM10 to 1 ug/m3, NO2 to 1 ppb, O3 to 0.001 ppm. The AQI is rounded to the nearest integer and
capped at 500.

Averaging times the AQI expects: PM2.5 and PM10 = 24-hour mean (what we use: local-day mean);
NO2 = 1-hour maximum (ppb); O3 = 8-hour maximum (ppm). Karachi has ground data for PM only, so the
NO2/O3 tables are here for later use and are never fed with invented values.

    sub_index("pm25", values)          -> array of sub-indices (NaN stays NaN)
    overall_aqi(pm25=..., pm10=...)    -> (aqi array, dominant pollutant array)
"""
import numpy as np
import pandas as pd

# (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    "pm25": {
        "epa_2024": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200),
                     (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)],
        "epa_2012": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
                     (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    },
    "pm10": {"any": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
                     (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)]},
    "no2": {"any": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200),
                    (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)]},          # ppb, 1-hour
    "o3": {"any": [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
                   (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)]},                              # ppm, 8-hour
}
DECIMALS = {"pm25": 1, "pm10": 0, "no2": 0, "o3": 3}            # truncation digits
STANDARD = "epa_2024"


def _table(pollutant, standard):
    t = BREAKPOINTS[pollutant]
    return np.array(t[standard] if standard in t else t["any"], dtype=float)


def truncate(values, digits):
    f = 10.0 ** digits
    return np.floor(np.asarray(values, dtype=float) * f + 1e-9) / f


def sub_index(pollutant, conc, standard=STANDARD):
    """Sub-index of one pollutant for an array of concentrations (NaN in, NaN out)."""
    tab = _table(pollutant, standard)
    c = truncate(conc, DECIMALS[pollutant])
    out = np.full(c.shape, np.nan)
    ok = np.isfinite(c) & (c >= 0)
    idx = np.searchsorted(tab[:, 0], c[ok], side="right") - 1          # last bin whose C_low <= c
    idx = np.clip(idx, 0, len(tab) - 1)
    clo, chi, ilo, ihi = tab[idx, 0], tab[idx, 1], tab[idx, 2], tab[idx, 3]
    val = (ihi - ilo) / (chi - clo) * (c[ok] - clo) + ilo
    out[ok] = np.minimum(np.rint(val), 500)                             # above the top breakpoint: capped at 500
    return out


def overall_aqi(standard=STANDARD, **conc):
    """AQI = max sub-index. conc: pm25=array, pm10=array, ... Returns (aqi, dominant pollutant name)."""
    names = [k for k in conc if conc[k] is not None]
    subs = np.column_stack([sub_index(k, np.asarray(conc[k], dtype=float), standard) for k in names])
    all_nan = np.isnan(subs).all(axis=1)
    aqi = np.where(all_nan, np.nan, np.nanmax(np.where(np.isnan(subs), -np.inf, subs), axis=1))
    dom = np.array(names, dtype=object)[np.where(all_nan, 0, np.nanargmax(np.where(np.isnan(subs), -np.inf, subs), axis=1))]
    dom = np.where(all_nan, None, dom)
    return aqi, dom


def category(aqi):
    bins = [-1, 50, 100, 150, 200, 300, 501]
    labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
    return pd.cut(pd.Series(aqi, dtype=float), bins=bins, labels=labels)
