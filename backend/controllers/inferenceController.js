const SonogramResult = require('../models/SonogramResult');
const axios = require('axios');
require('dotenv').config(); // ✅ Load env variables

// @desc    Upload sonogram image and trigger Celery processing
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

        // ✅ Cloudinary URL
        const secureCloudURL = req.file.path;

        // ✅ 1. Save record in MongoDB
        const sonogram = await SonogramResult.create({
            user: req.user.id,
            cowId,
            imagePath: secureCloudURL,
            status: "PENDING",
            classification: "Awaiting process...",
            confidence: 0,
            predictedYield: 0
        });

        console.log(`✅ Saved to DB with ID: ${sonogram._id}`);
        console.log(`🌐 Image URL: ${secureCloudURL}`);

        // ✅ 2. CALL PYTHON API (FIXED)
        try {
            const PYTHON_API_URL = process.env.PYTHON_API_URL;

            console.log("🚀 Calling Python API:", PYTHON_API_URL);

            const response = await axios.post(
                `${PYTHON_API_URL}/process`,
                {
                    result_id: sonogram._id.toString(),
                    image_path: secureCloudURL
                },
                {
                    timeout: 30000 // ⏱️ prevent timeout issues
                }
            );

            console.log('✅ Celery task triggered:', response.data);

        } catch (apiError) {
            console.error('❌ Python API ERROR:');
            console.error('Message:', apiError.message);
            console.error('Data:', apiError.response?.data);

            // ❗ Mark as FAILED if API fails
            await SonogramResult.findByIdAndUpdate(sonogram._id, {
                status: "FAILED"
            });

            return res.status(500).json({
                success: false,
                message: 'Failed to trigger processing task'
            });
        }

        // ✅ 3. Success response
        return res.status(200).json({
            success: true,
            message: 'Image uploaded & processing started ✅',
            sonogramId: sonogram._id,
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
            message: 'Sonogram record removed successfully ✅'
        });

    } catch (error) {
        console.error('❌ DELETE ERROR:', error);

        res.status(500).json({
            success: false,
            message: 'Server error during deletion processing'
        });
    }
};
