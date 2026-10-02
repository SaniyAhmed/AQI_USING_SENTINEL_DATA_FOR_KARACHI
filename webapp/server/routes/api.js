const express = require("express");
const Forecast = require("../models/Forecast");
const Meta = require("../models/Meta");

const router = express.Router();

const DISTRICT_ORDER = ["Karachi Central", "Karachi East", "Karachi South", "Karachi West", "Korangi", "Malir"];

// GET /api/forecast - everything the dashboard needs in one call: every district's 2-day forecast,
// the out-of-fold model performance stats, and which model won each horizon.
router.get("/forecast", async (req, res) => {
  try {
    const [rows, meta] = await Promise.all([Forecast.find({}).lean(), Meta.findOne({ key: "latest" }).lean()]);
    if (!meta) return res.status(503).json({ error: "Forecast data has not been seeded yet." });

    const districts = {};
    for (const name of DISTRICT_ORDER) districts[name] = [];
    for (const r of rows) {
      if (!districts[r.district]) districts[r.district] = [];
      districts[r.district].push({
        horizon: r.horizon, label: r.label, date: r.date, pm25: r.pm25, aqi: r.aqi, category: r.category, model: r.model,
      });
    }
    for (const name of Object.keys(districts)) districts[name].sort((a, b) => a.horizon - b.horizon);

    res.json({
      issueDate: meta.issueDate,
      districts,
      districtOrder: DISTRICT_ORDER,
      modelStats: meta.modelStats,
      bestModelPerHorizon: meta.bestModelPerHorizon,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to load forecast data." });
  }
});

router.get("/health", (req, res) => res.json({ ok: true }));

module.exports = router;
