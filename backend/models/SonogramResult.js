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
        default: 'pending' // ✅ Lowercase to align with Celery updates
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
        default: 0 // ✅ Named exactly like your Python worker field
    },
    errorReason: {
        type: String
    }
}, { 
    timestamps: true 
});

module.exports = mongoose.model('SonogramResult', SonogramResultSchema);
