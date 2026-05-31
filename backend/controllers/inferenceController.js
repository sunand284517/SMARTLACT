const SonogramResult = require('../models/SonogramResult');
const client = require('../services/queueService');
const path = require('path');

exports.uploadSonogram = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'No image file provided' });
        }
        
        const cowId = req.body.cowId || 'Unknown Cow';
        const imageUrl = `/uploads/${req.file.filename}`;

        const sonogram = new SonogramResult({
            user: req.user.id,
            cowId,
            imageUrl
        });
        await sonogram.save();

        // Push task to Celery Queue. The python worker will pick this up
        // 'tasks.predict' must match the Python task name
        const task = client.createTask('tasks.predict');
        const absoluteImagePath = path.resolve(req.file.path);
        
        const result = task.applyAsync([sonogram._id.toString(), absoluteImagePath]);

        res.json({ 
            message: 'Image uploaded successfully. Analysis in progress.', 
            sonogramId: sonogram._id,
            taskId: result.taskId
        });
   } catch (error) {
    console.error('UPLOAD ERROR:', error);

    res.status(500).json({
        message: 'Server error during upload',
        error: error.message
    });
}
};

exports.getSonograms = async (req, res) => {
    try {
        const results = await SonogramResult.find({ user: req.user.id }).sort({ createdAt: -1 });
        res.json(results);
    } catch (error) {
        res.status(500).json({ message: 'Server error' });
    }
};

exports.deleteSonogram = async (req, res) => {
    try {
        const result = await SonogramResult.findOneAndDelete({ _id: req.params.id, user: req.user.id });
        if (!result) {
            return res.status(404).json({ message: 'Result not found or unauthorized' });
        }
        res.json({ message: 'Sonogram deleted successfully' });
    } catch (error) {
        res.status(500).json({ message: 'Server error during deletion' });
    }
};
