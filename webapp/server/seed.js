const fs = require("fs");
const path = require("path");
const Forecast = require("./models/Forecast");
const Meta = require("./models/Meta");

const SEED_FILE = path.join(__dirname, "seedData.json");

// Idempotent: safe to call on every server boot. Re-seeds only when the source file's issue date
// differs from what is already stored (e.g. after phase5_models/predict_today.py produces a new day).
async function seedIfNeeded() {
  const raw = JSON.parse(fs.readFileSync(SEED_FILE, "utf-8"));
  const existing = await Meta.findOne({ key: "latest" });
  if (existing && existing.issueDate === raw.issue_date) {
    console.log(`Seed data already up to date (issue date ${raw.issue_date})`);
    return;
  }

  await Forecast.deleteMany({});
  const docs = [];
  for (const [district, entries] of Object.entries(raw.districts)) {
    for (const e of entries) {
      docs.push({ district, issueDate: raw.issue_date, ...e });
    }
  }
  await Forecast.insertMany(docs);
  await Meta.findOneAndUpdate(
    { key: "latest" },
    { key: "latest", issueDate: raw.issue_date, modelStats: raw.model_stats, bestModelPerHorizon: raw.best_model_per_horizon },
    { upsert: true }
  );
  console.log(`Seeded ${docs.length} forecast rows for issue date ${raw.issue_date}`);
}

module.exports = { seedIfNeeded };
