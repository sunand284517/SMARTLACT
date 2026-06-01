const mongoose = require('mongoose');

const SonogramResultSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    cowId: {
        type: String,
        default: 'Unknown Cow'
    },
    imagePath: {
        type: String,
        required: true
    },
    status: {
        type: String,
        default: 'pending' // Matches lowercase Python lifecycle states
    },
    classification: {
        type: String,
        default: 'Awaiting process...'
    },
    confidence: {
        type: Number,
        default: 0
    },
    yield_litres: {
        type: Number,
        default: 0 // ✅ FIX: Named exactly like your Python worker field
    },
    errorReason: {
        type: String
    }
}, { 
    timestamps: true // Automatically manages createdAt and updatedAt fields
});

module.exports = mongoose.model('SonogramResult', SonogramResultSchema);
