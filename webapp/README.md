# Hawa - Karachi AQI dashboard (MERN)

A MongoDB + Express + React + Node dashboard for the Phase 5 forecast: today / tomorrow / day-after
AQI for all six Karachi districts, which model produced each number, and the model's out-of-fold
accuracy (RMSE, R², AQI category accuracy) versus a naive baseline.

## Structure

```
webapp/
  server/   Express API + Mongoose models. Seeds MongoDB from seedData.json on boot.
  client/   React (Vite) frontend.
```

## Run it

Two terminals:

```bash
cd webapp/server && npm install && npm run dev    # http://localhost:5050
cd webapp/client && npm install && npm run dev    # http://localhost:5173
```

Open http://localhost:5173. The client dev server proxies `/api/*` to the backend (see
`client/vite.config.js`).

No standalone MongoDB install is required: `server/db.js` starts a self-contained in-memory MongoDB
(`mongodb-memory-server`, downloads a real `mongod` binary the first time it runs, then caches it) and
seeds it from `server/seedData.json`. To use a real MongoDB instance instead, set `MONGODB_URI` in
`server/.env`.

## Refreshing the forecast

`server/seedData.json` is a snapshot produced by the Python pipeline, not hand-edited. After a new day's
forecast is generated:

```bash
python phase3_features/step5_gfs_forecast.py --issue YYYY-MM-DD   # if not already run
python phase5_models/predict_today.py --issue YYYY-MM-DD
python phase5_models/export_webapp_data.py --issue YYYY-MM-DD     # writes webapp/server/seedData.json
```

Restart the server (`npm run dev` again, or let nodemon restart it) - `seed.js` compares the issue date
already in MongoDB and only re-seeds when it's different, so this is safe to run repeatedly.

## API

`GET /api/forecast` returns everything the dashboard needs in one call:

```json
{
  "issueDate": "2026-09-27",
  "districts": { "Karachi Central": [{ "horizon": 0, "label": "Today", "date": "...", "pm25": 34.6, "aqi": 98, "category": "Moderate", "model": "LightGBM" }, ...], ... },
  "districtOrder": ["Karachi Central", "Karachi East", "Karachi South", "Karachi West", "Korangi", "Malir"],
  "modelStats": { "overall|pred_lightgbm": {...}, "overall|pred_xgboost": {...}, "overall|pred_persistence": {...} },
  "bestModelPerHorizon": { "0": "LightGBM", "1": "LightGBM", "2": "XGBoost" }
}
```

## Notes

- District boundary geometry (`client/src/data/mapData.json`) is a simplified, pre-projected copy of
  `data/processed/boundaries/karachi_districts_wgs84.geojson` (Phase 1) - static presentational data,
  so it lives in the frontend rather than the database. The map view is cropped to the dense urban core;
  Malir's real extent reaches much further north and south (see the caption under the map).
- Colors: dark theme matching the provided reference (near-black surfaces, teal accent for UI chrome).
  AQI category colors (green/amber/orange/red/purple/maroon) are a separate, EPA-standard semantic
  palette from the teal brand accent, per the six bands in `phase4_ground/aqi.py`.
