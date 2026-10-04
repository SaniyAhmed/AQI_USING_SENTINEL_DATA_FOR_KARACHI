require("dotenv").config();
const { connectDB } = require("../db");
const { seedIfNeeded } = require("../seed");
const { createApp } = require("../app");

const app = createApp();

// Vercel reuses a warm container across invocations, so cache the connect+seed
// promise instead of redoing it on every request.
let readyPromise = null;
function ensureReady() {
  if (!readyPromise) {
    readyPromise = connectDB()
      .then(() => seedIfNeeded())
      .catch((err) => {
        readyPromise = null; // allow the next invocation to retry
        throw err;
      });
  }
  return readyPromise;
}

module.exports = async (req, res) => {
  try {
    await ensureReady();
  } catch (err) {
    console.error("Failed to initialize server:", err);
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ error: "Server initialization failed." }));
    return;
  }
  app(req, res);
};
