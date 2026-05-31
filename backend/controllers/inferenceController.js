const SonogramResult = require('../models/SonogramResult');
const client = require('../services/queueService');

// @desc    Upload sonogram image directly to cloud storage and pass URL to Python worker
// @route   POST /api/inference/upload
exports.uploadSonogram = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'No image file provided'
            });
        }

        const cowId = req.body.cowId || 'Unknown Cow';
        
        // ✅ req.file.path automatically contains the secure Cloudinary web link URL (https://res.cloudinary.com/...)
        const secureCloudURL = req.file.path;

        // 1. Create a tracking document in MongoDB tracking the cloud image link
        const sonogram = new SonogramResult({
            user: req.user.id,
            cowId,
            imagePath: secureCloudURL // ✅ Tracks the permanent web link directly
        });

        await sonogram.save();

        console.log(`✉️ Task generated in DB. Forwarding cloud URL to Upstash Redis for Record ID: ${sonogram._id}`);
        console.log(`🌐 Secure Cloud URL: ${secureCloudURL}`);

        // 2. Instantiate the task matching your Python worker name exactly
        const task = client.createTask('predict_task');

        // ✅ Using task.delay to cleanly send the sequential string arguments down the queue
        const result = task.delay(
            sonogram._id.toString(),
            secureCloudURL
        );

        console.log('✅ Task queued smoothly through Upstash Redis wrapper:', result);

        return res.status(200).json({
            success: true,
            message: 'Image uploaded to cloud successfully. Analysis loop triggered! ✅',
            sonogramId: sonogram._id,
            taskId: result.taskId || null,
            url: secureCloudURL
        });

    } catch (error) {
        console.error('❌ UPLOAD CONTROLLER ERROR:', error);

        return res.status(500).json({
            success: false,
            message: 'Server error during upload processing',
            error: error.message
        });
    }
};

// @desc    Get historical data list for the logged-in user
// @route   GET /api/inference/history
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

// @desc    Remove an old sonogram record
// @route   DELETE /api/inference/:id
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
            message: 'Sonogram record removed from database successfully ✅'
        });

    } catch (error) {
        console.error('❌ DELETE ERROR:', error);

        res.status(500).json({
            success: false,
            message: 'Server error during deletion processing'
        });
    }
};
