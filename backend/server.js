const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// ==========================================
// 🔒 PRODUCTION CORS CONFIGURATION
// ==========================================
const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:5173',
    'https://smartlact-52.onrender.com' // Explicit frontend deployment domain
];

app.use(cors({
    origin: function (origin, callback) {
        // Allow server-to-server requests or REST exploration tools (like Postman/Curl)
        if (!origin) return callback(null, true);
        
        if (allowedOrigins.indexOf(origin) === -1) {
            const msg = `The CORS policy for this site does not allow access from origin: ${origin}`;
            return callback(new Error(msg), false);
        }
        return callback(null, true);
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// ==========================================
// 📦 GLOBAL PARSING MIDDLEWARES
// ==========================================
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static directory for uploaded sonogram files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ==========================================
// 🔍 VERBOSE REQUEST LOGGER (For debugging 400 Errors)
// ==========================================
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] 📩 ${req.method} ${req.url}`);
    if (req.method === 'POST' || req.method === 'PUT') {
        console.log('📦 Request Payload:', JSON.stringify(req.body, null, 2));
    }
    next();
});

// ==========================================
// 🛣️ APPLICATION API ROUTING
// ==========================================
app.use('/api/auth', require('./routes/auth'));
app.use('/api/inference', require('./routes/inference'));

// Base application verification endpoint
app.get('/api/health', (req, res) => {
    res.status(200).json({ 
        status: 'ok', 
        message: 'Dairy Sonogram Core API is fully operational.' 
    });
});

// ==========================================
// 🚨 FALLBACK GLOBAL ERROR HANDLING
// ==========================================
app.use((err, req, res, next) => {
    console.error('❌ Unhandled Server Exception:', err.stack);
    res.status(err.status || 500).json({
        success: false,
        error: err.message || 'Internal Server Error instance encountered.'
    });
});

// ==========================================
// 🗄️ DATABASE PERSISTENCE & STARTUP
// ==========================================
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/dairy-sonogram';

console.log('🔄 Initializing connection sequence to MongoDB...');
mongoose.connect(MONGO_URI)
.then(() => {
    console.log('✅ MongoDB connected securely.');
    
    // Bind server listener only after a successful database handshake completes
    app.listen(PORT, () => {
        console.log(`🚀 API Microservice is live and running on port ${PORT}`);
    });
})
.catch(err => {
    console.error(`❌ Critical Database Handshake Failed: ${err.message}`);
    process.exit(1); // Force termination if application lacks database capabilities
});
