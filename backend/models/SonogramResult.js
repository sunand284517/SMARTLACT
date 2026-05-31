const mongoose = require('mongoose');

const SonogramResultSchema = new mongoose.Schema({
  user: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User', 
    required: true 
  },
  cowId: { 
    type: String, 
    required: false // ⚠️ Changed to false for now so it doesn't break if your form doesn't send it yet
  },
  imagePath: { 
    type: String, 
    required: true // ⚠️ Changed from imageUrl to imagePath to match your python worker argument exactly
  },
  status: { 
    type: String, 
    enum: ['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'], 
    default: 'PENDING' 
  },
  classification: { 
    type: String, 
    default: "Awaiting process..." 
  }, 
  confidence: { 
    type: Number, 
    default: 0 
  },
  predictedYield: { 
    type: Number, 
    default: 0 
  }
}, {
  timestamps: true // ⚠️ This automatically manages your createdAt and updatedAt fields for you!
});

module.exports = mongoose.model('SonogramResult', SonogramResultSchema);
