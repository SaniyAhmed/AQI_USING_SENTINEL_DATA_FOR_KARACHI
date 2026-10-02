require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { connectDB } = require("./db");
const { seedIfNeeded } = require("./seed");
const apiRoutes = require("./routes/api");

const PORT = process.env.PORT || 5050;

async function main() {
  await connectDB();
  await seedIfNeeded();

  const app = express();
  app.use(cors());
  app.use(express.json());
  app.use("/api", apiRoutes);

  app.listen(PORT, () => {
    console.log(`Karachi AQI API listening on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
