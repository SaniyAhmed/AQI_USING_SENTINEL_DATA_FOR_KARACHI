const express = require("express");
const cors = require("cors");
const apiRoutes = require("./routes/api");

function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use("/api", apiRoutes);
  return app;
}

module.exports = { createApp };
