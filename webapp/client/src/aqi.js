// EPA 2024 AQI bands - the same six categories phase5_models/phase4_ground/aqi.py computes from PM2.5.
// Plain hex here (not CSS var()) because JS needs to derive tints/alphas from these at render time;
// styles.css keeps the matching --aqi-* custom properties in sync for anything styled from pure CSS.
export const AQI_BANDS = [
  { key: "Good", color: "#34d399", range: "0-50" },
  { key: "Moderate", color: "#fbbf24", range: "51-100" },
  { key: "Unhealthy for Sensitive Groups", color: "#fb923c", range: "101-150", short: "USG" },
  { key: "Unhealthy", color: "#f87171", range: "151-200" },
  { key: "Very Unhealthy", color: "#c084fc", range: "201-300" },
  { key: "Hazardous", color: "#dc2626", range: "301-500" },
];

const BAND_BY_KEY = Object.fromEntries(AQI_BANDS.map((b) => [b.key, b]));

export function aqiColor(category) {
  return BAND_BY_KEY[category]?.color ?? "var(--text-dim)";
}

export function aqiShort(category) {
  return BAND_BY_KEY[category]?.short ?? category;
}

export const DISTRICT_CODE = {
  "Karachi Central": "C",
  "Karachi East": "E",
  "Karachi South": "S",
  "Karachi West": "W",
  Korangi: "K",
  Malir: "M",
};

export const MODEL_CODE = { LightGBM: "LGB", XGBoost: "XGB" };
