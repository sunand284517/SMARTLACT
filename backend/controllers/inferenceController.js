const SonogramResult = require('../models/SonogramResult');
const axios = require('axios');
require('dotenv').config();


// =========================
// UPLOAD SONOGRAM
// =========================
exports.uploadSonogram = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'No image file provided'
            });
        }

        const cowId = req.body.cowId || 'Unknown Cow';
        const secureCloudURL = req.file.path;

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

        const PYTHON_API_URL = process.env.PYTHON_API_URL;

        if (!PYTHON_API_URL) {
            throw new Error("PYTHON_API_URL is not set in environment variables");
        }

        console.log("🚀 Calling Python API:", PYTHON_API_URL);

        const response = await axios.post(
            `${PYTHON_API_URL}/process`,
            {
                result_id: sonogram._id.toString(),
                image_path: secureCloudURL
            },
            { timeout: 30000 }
        );

        console.log('✅ Celery task triggered:', response.data);

        return res.status(200).json({
            success: true,
            message: 'Image uploaded & processing started ✅',
            sonogramId: sonogram._id,
            url: secureCloudURL
        });

    } catch (error) {
        console.error('❌ ERROR:', error.message);

        return res.status(500).json({
            success: false,
            message: error.message
        });
    }
};


// =========================
// GET SONOGRAM HISTORY
// =========================
exports.getSonograms = async (req, res) => {
    try {
        const data = await SonogramResult.find({ user: req.user.id })
            .sort({ createdAt: -1 });

        return res.status(200).json({
            success: true,
            data
        });

    } catch (error) {
        return res.status(500).json({
            success: false,
            message: error.message
        });
    }
};


// =========================
// DELETE SONOGRAM
// =========================
exports.deleteSonogram = async (req, res) => {
    try {
        const { id } = req.params;

        const deleted = await SonogramResult.findOneAndDelete({
            _id: id,
            user: req.user.id
        });

        if (!deleted) {
            return res.status(404).json({
                success: false,
                message: "Record not found"
            });
        }

        return res.status(200).json({
            success: true,
            message: "Deleted successfully"
        });

    } catch (error) {
        return res.status(500).json({
            success: false,
            message: error.message
        });
    }
};
