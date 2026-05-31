const SonogramResult = require('../models/SonogramResult');
const client = require('../services/queueService');
const path = require('path');

exports.uploadSonogram = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'No image file provided'
            });
        }

        const cowId = req.body.cowId || 'Unknown Cow';
        const absoluteImagePath = path.resolve(req.file.path);

        const sonogram = new SonogramResult({
            user: req.user.id,
            cowId,
            imagePath: absoluteImagePath
        });

        await sonogram.save();

        console.log(`✉️ Dispatching job for Record ID: ${sonogram._id}`);

        const task = client.createTask('predict_task');

        const result = await task.applyAsync([
            sonogram._id.toString(),
            absoluteImagePath
        ]);

        console.log('✅ Task queued:', result);

        return res.status(200).json({
            success: true,
            message: 'Image uploaded successfully. Analysis in progress.',
            sonogramId: sonogram._id,
            taskId: result.taskId || null
        });

    } catch (error) {
        console.error('❌ UPLOAD CONTROLLER ERROR:', error);

        return res.status(500).json({
            success: false,
            message: 'Server error during upload',
            error: error.message
        });
    }
};

exports.getSonograms = async (req, res) => {
    try {
        const results = await SonogramResult
            .find({ user: req.user.id })
            .sort({ createdAt: -1 });

        res.status(200).json(results);

    } catch (error) {
        console.error('❌ FETCH HISTORY ERROR:', error);

        res.status(500).json({
            success: false,
            message: 'Server error while fetching history'
        });
    }
};

exports.deleteSonogram = async (req, res) => {
    try {
        const result = await SonogramResult.findOneAndDelete({
            _id: req.params.id,
            user: req.user.id
        });

        if (!result) {
            return res.status(404).json({
                success: false,
                message: 'Result not found or unauthorized'
            });
        }

        res.status(200).json({
            success: true,
            message: 'Sonogram deleted successfully'
        });

    } catch (error) {
        console.error('❌ DELETE ERROR:', error);

        res.status(500).json({
            success: false,
            message: 'Server error during deletion'
        });
    }
};
