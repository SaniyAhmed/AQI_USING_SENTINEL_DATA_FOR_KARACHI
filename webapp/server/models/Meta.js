const mongoose = require("mongoose");

// Single-document collection: issue date, model out-of-fold stats, and which model won each horizon.
const metaSchema = new mongoose.Schema(
  {
    key: { type: String, required: true, unique: true, default: "latest" },
    issueDate: { type: String, required: true },
    modelStats: { type: mongoose.Schema.Types.Mixed, required: true },
    bestModelPerHorizon: { type: mongoose.Schema.Types.Mixed, required: true },
  },
  { versionKey: false }
);

module.exports = mongoose.model("Meta", metaSchema);
