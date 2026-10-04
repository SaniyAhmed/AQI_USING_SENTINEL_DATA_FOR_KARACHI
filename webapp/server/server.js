require("dotenv").config();
const { connectDB } = require("./db");
const { seedIfNeeded } = require("./seed");
const { createApp } = require("./app");

const PORT = process.env.PORT || 5050;

async function main() {
  await connectDB();
  await seedIfNeeded();

  const app = createApp();

  app.listen(PORT, () => {
    console.log(`Karachi AQI API listening on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
