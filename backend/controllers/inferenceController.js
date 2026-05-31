const SonogramResult = require('../models/SonogramResult');
const client = require('../services/queueService');
const path = require('path');

// @desc    Upload sonogram image and trigger Celery ML worker
// @route   POST /api/inference/upload
exports.uploadSonogram = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'No image file provided' });
        }
        
        const cowId = req.body.cowId || 'Unknown Cow';
        
        // Use the absolute path so your Python script can easily locate the file on the server
        const absoluteImagePath = path.resolve(req.file.path);

        // 1. Create tracking document in MongoDB matching your schema keys
        const sonogram = new SonogramResult({
            user: req.user.id,
            cowId,
            imagePath: absoluteImagePath // ✅ Converted to match schema and Python parameters!
        });
        await sonogram.save();

        console.log(`✉️ Dispatching job to Redis queue for Record ID: ${sonogram._id}`);

        // 2. Trigger task to match your Python @app.task(name="predict_task") configuration exactly
        const task = client.createTask('predict_task'); // ✅ Fixed task name matching
        
        // Send arguments sequentially matching: predict_task(sonogram_id, image_path)
        const result = task.delay(sonogram._id.toString(), absoluteImagePath);

        res.json({ 
            message: 'Image uploaded successfully. Analysis in progress. ✅', 
            sonogramId: sonogram._id,
            taskId: result.taskId
        });
        
   } catch (error) {
        console.error('❌ UPLOAD CONTROLLER ERROR:', error);
        res.status(500).json({
            message: 'Server error during upload',
            error: error.message
        });
    }
};

// @desc    Get historical data list for the logged-in user
// @route   GET /api/inference/history
exports.getSonograms = async (req, res) => {
    try {
        // Fetch history belonging to the logged-in user, ordered by newest first
        const results = await SonogramResult.find({ user: req.user.id }).sort({ createdAt: -1 });
        res.json(results);
    } catch (error) {
        console.error('❌ FETCH HISTORY ERROR:', error.message);
        res.status(500).json({ message: 'Server error while fetching history' });
    }
};

// @desc    Remove an old sonogram record and cancel track trace
// @route   DELETE /api/inference/:id
exports.deleteSonogram = async (req, res) => {
    try {
        const result = await SonogramResult.findOneAndDelete({ _id: req.params.id, user: req.user.id });
        if (!result) {
            return res.status(404).json({ message: 'Result not found or unauthorized' });
        }
        res.json({ message: 'Sonogram deleted successfully from backend ✅' });
    } catch (error) {
        console.error('❌ DELETE RECORD ERROR:', error.message);
        res.status(500).json({ message: 'Server error during deletion' });
    }
};
