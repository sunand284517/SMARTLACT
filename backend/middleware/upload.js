const cloudinary = require('cloudinary').v2;
const { CloudinaryStorage } = require('multer-storage-cloudinary');
const multer = require('multer');

// 1. Configure Cloudinary with your cloud environment variables
cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET
});

// 2. Setup the Cloudinary storage pipeline engine
const storage = new CloudinaryStorage({
    cloudinary: cloudinary,
    params: {
        folder: 'dairy-sonograms', // Creates/uses this folder in your Cloudinary media library
        allowed_formats: ['jpg', 'jpeg', 'png'],
        // Explicitly format the public ID filename structure
        public_id: (req, file) => `sonogram-${Date.now()}`
    }
});

// 3. Keep your strict image-only filtering safeguard intact
const fileFilter = (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
        cb(null, true);
    } else {
        cb(new Error('Not an image! Please upload an image.'), false);
    }
};

// 4. Initialize the custom cloud multer setup
const upload = multer({ 
    storage: storage, 
    fileFilter: fileFilter 
});

module.exports = upload;
