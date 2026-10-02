const mongoose = require("mongoose");
const { MongoMemoryServer } = require("mongodb-memory-server");

// No standalone MongoDB is installed on this machine, so a self-contained in-memory MongoDB instance
// is started on demand (mongodb-memory-server downloads a real mongod binary the first time it runs
// and keeps it cached in ~/.cache for every run after). Set MONGODB_URI in .env to point at a real
// MongoDB instance instead - db.js falls back to it automatically when set.
let memoryServer = null;

async function connectDB() {
  const uri = process.env.MONGODB_URI;
  if (uri) {
    await mongoose.connect(uri);
    console.log(`MongoDB connected: ${uri}`);
    return;
  }
  memoryServer = await MongoMemoryServer.create({ instance: { dbName: "karachi_aqi" } });
  const memUri = memoryServer.getUri();
  await mongoose.connect(memUri, { dbName: "karachi_aqi" });
  console.log(`MongoDB (in-memory) connected: ${memUri}`);
}

async function disconnectDB() {
  await mongoose.disconnect();
  if (memoryServer) await memoryServer.stop();
}

module.exports = { connectDB, disconnectDB };
