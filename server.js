const express = require("express");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

// In-memory user store (replace with database in production)
const users = [];
const REFRESH_TOKENS = [];

// Configuration
const JWT_SECRET = process.env.JWT_SECRET || "your-secret-key";
const JWT_REFRESH_SECRET =
  process.env.JWT_REFRESH_SECRET || "your-refresh-secret";

// Middleware: Verify JWT Token
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ error: "Access token required" });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: "Invalid or expired token" });
    }
    req.user = user;
    next();
  });
};

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "Auth API is running" });
});

// Register new user
app.post("/api/auth/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Validation
    if (!username || !email || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    if (password.length < 6) {
      return res
        .status(400)
        .json({ error: "Password must be at least 6 characters" });
    }

    // Check if user exists
    const existingUser = users.find(
      (u) => u.email === email || u.username === username,
    );
    if (existingUser) {
      return res.status(409).json({ error: "User already exists" });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create user
    const newUser = {
      id: Date.now().toString(),
      username,
      email,
      password: hashedPassword,
      createdAt: new Date().toISOString(),
    };

    users.push(newUser);

    // Generate tokens
    const accessToken = generateAccessToken(newUser);
    const refreshToken = generateRefreshToken(newUser);

    REFRESH_TOKENS.push(refreshToken);

    res.status(201).json({
      message: "User registered successfully",
      user: { id: newUser.id, username, email },
      accessToken,
      refreshToken,
    });
  } catch (error) {
    res.status(500).json({ error: "Registration failed" });
  }
});

// Login user
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validation
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Find user
    const user = users.find((u) => u.email === email);
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Verify password
    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Generate tokens
    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);

    REFRESH_TOKENS.push(refreshToken);

    res.json({
      message: "Login successful",
      user: { id: user.id, username: user.username, email: user.email },
      accessToken,
      refreshToken,
    });
  } catch (error) {
    res.status(500).json({ error: "Login failed" });
  }
});

// Refresh token
app.post("/api/auth/refresh", (req, res) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(401).json({ error: "Refresh token required" });
  }

  if (!REFRESH_TOKENS.includes(refreshToken)) {
    return res.status(403).json({ error: "Invalid refresh token" });
  }

  jwt.verify(refreshToken, JWT_REFRESH_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: "Invalid refresh token" });
    }

    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);

    // Remove old refresh token and add new one
    const index = REFRESH_TOKENS.indexOf(refreshToken);
    REFRESH_TOKENS.splice(index, 1);
    REFRESH_TOKENS.push(newRefreshToken);

    res.json({ accessToken: newAccessToken, refreshToken: newRefreshToken });
  });
});

// Logout user
app.post("/api/auth/logout", (req, res) => {
  const { refreshToken } = req.body;

  const index = REFRESH_TOKENS.indexOf(refreshToken);
  if (index > -1) {
    REFRESH_TOKENS.splice(index, 1);
  }

  res.json({ message: "Logged out successfully" });
});

// Get user profile (protected)
app.get("/api/auth/profile", authenticateToken, (req, res) => {
  const user = users.find((u) => u.id === req.user.id);
  if (!user) {
    return res.status(404).json({ error: "User not found" });
  }

  res.json({
    user: {
      id: user.id,
      username: user.username,
      email: user.email,
      createdAt: user.createdAt,
    },
  });
});

// Update user profile (protected)
app.put("/api/auth/profile", authenticateToken, async (req, res) => {
  const { username, email, currentPassword, newPassword } = req.body;
  const userIndex = users.findIndex((u) => u.id === req.user.id);

  if (userIndex === -1) {
    return res.status(404).json({ error: "User not found" });
  }

  const user = users[userIndex];

  // Update username and email
  if (username) user.username = username;
  if (email) user.email = email;

  // Update password if provided
  if (newPassword) {
    if (!currentPassword) {
      return res.status(400).json({ error: "Current password required" });
    }

    const validPassword = await bcrypt.compare(currentPassword, user.password);
    if (!validPassword) {
      return res.status(401).json({ error: "Current password is incorrect" });
    }

    user.password = await bcrypt.hash(newPassword, 10);
  }

  user.updatedAt = new Date().toISOString();
  users[userIndex] = user;

  res.json({
    message: "Profile updated successfully",
    user: { id: user.id, username: user.username, email: user.email },
  });
});

// Change password (protected)
app.post("/api/auth/change-password", authenticateToken, async (req, res) => {
  const { currentPassword, newPassword } = req.body;
  const userIndex = users.findIndex((u) => u.id === req.user.id);

  if (userIndex === -1) {
    return res.status(404).json({ error: "User not found" });
  }

  if (!currentPassword || !newPassword) {
    return res.status(400).json({ error: "Current and new password required" });
  }

  if (newPassword.length < 6) {
    return res
      .status(400)
      .json({ error: "New password must be at least 6 characters" });
  }

  const user = users[userIndex];
  const validPassword = await bcrypt.compare(currentPassword, user.password);

  if (!validPassword) {
    return res.status(401).json({ error: "Current password is incorrect" });
  }

  user.password = await bcrypt.hash(newPassword, 10);
  user.updatedAt = new Date().toISOString();
  users[userIndex] = user;

  res.json({ message: "Password changed successfully" });
});

// Helper functions
function generateAccessToken(user) {
  return jwt.sign(
    { id: user.id, username: user.username, email: user.email },
    JWT_SECRET,
    { expiresIn: "15m" },
  );
}

function generateRefreshToken(user) {
  return jwt.sign({ id: user.id }, JWT_REFRESH_SECRET, { expiresIn: "7d" });
}

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Auth API running on http://localhost:${PORT}`);
});

module.exports = app;
