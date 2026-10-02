const mongoose = require("mongoose");

const forecastSchema = new mongoose.Schema(
  {
    district: { type: String, required: true, index: true },
    horizon: { type: Number, required: true }, // 0 = today, 1 = tomorrow
    label: { type: String, required: true },
    date: { type: String, required: true },
    pm25: { type: Number, required: true },
    aqi: { type: Number, required: true },
    category: { type: String, required: true },
    model: { type: String, required: true }, // LightGBM | XGBoost - whichever won that horizon out-of-fold
    issueDate: { type: String, required: true },
  },
  { versionKey: false }
);

module.exports = mongoose.model("Forecast", forecastSchema);
