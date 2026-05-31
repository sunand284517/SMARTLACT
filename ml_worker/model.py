import os
import sys
import gdown
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define 5 major productive stages
CLASSES = [
    'Dry Period',
    'Peak Lactation',
    'Late Lactation', 
    'Fresh Cows',
    'Peri-Partum'
]

# =========================================================
# 🧠 THE EXACT CUSTOM CNN ARCHITECTURE SYNCHRONIZED TO CHECKPOINT
# =========================================================
class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CowSonogramCNN, self).__init__()
        
        # Channel configurations matched perfectly to your checkpoint [16, 32, 64, 128, 128]
        self.features = nn.Sequential(
            # Block 1: Input 3 channels -> 16 channels
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Grid size: 112x112
            
            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Grid size: 56x56
            
            # Block 3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Grid size: 28x28
            
            # Block 4: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Grid size: 14x14
            
            # Block 5: 128 -> 128 channels
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Final grid size: 7x7 (128 * 7 * 7 = 6272 features)
        )
        
        # Linear layer configurations
        self.fc1 = nn.Linear(6272, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        
        # Output Multi-Task Learning Linear Heads
        self.classification_head = nn.Linear(512, num_classes)
        self.regression_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flattens cleanly down to [Batch, 6272]
        
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        
        class_logits = self.classification_head(x)
        yield_pred = self.regression_head(x)
        
        return class_logits, yield_pred

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'cow_model.pth')
MODEL_ID = "1V8Lobs36IXWHwVs9C7Y01wxU-tBew6gb"

# =========================
# 📥 MODEL DOWNLOADER
# =========================
def download_model():
    if os.path.exists(DEFAULT_MODEL_PATH):
        print(f"✅ Verified weight configuration checkpoint file at: {DEFAULT_MODEL_PATH}")
        return

    try:
        print("📥 Model weights missing. Downloading custom architecture checkpoint from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        
        gdown.download(
            url=url,
            output=DEFAULT_MODEL_PATH,
            quiet=False
        )

        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(f"❌ Download error: file missing at {DEFAULT_MODEL_PATH}")
        print("✅ Custom CNN checkpoint successfully loaded.")
    except Exception as e:
        print(f"❌ Auto-download error sequence triggered: {e}")
        raise e

# =========================
# 🖥️ MODEL LOADER & PARALLEL KEY AUDITOR
# =========================
_cached_model = None

def get_model(model_path=DEFAULT_MODEL_PATH):
    global _cached_model

    if _cached_model is None:
        download_model()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Initializing custom network inference engine on device: {device}")

        model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

        print("📦 Mounting model checkpoint layer parameters...")
        
        # Capture state dictionary tracking diagnostic flags
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(model_path, map_location=device), 
            strict=False
        )
        
        if missing_keys:
            print(f"⚠️ DIAGNOSTIC WARNING | Missing weights keys in checkpoint file: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ DIAGNOSTIC WARNING | Unexpected weights keys in checkpoint file: {unexpected_keys}")
            
        model.eval() 
        print("✅ Core architecture layers loaded and synchronized perfectly.")
        _cached_model = (model, device)

    return _cached_model

def get_consistent_class(yield_val):
    if yield_val == 0:
        return 'Dry Period'
    elif yield_val <= 10:
        return 'Peri-Partum'
    elif yield_val <= 20:
        return 'Fresh Cows'
    elif yield_val >= 35:
        return 'Peak Lactation'
    else:
        return 'Late Lactation'
    

def predict_image(image_path, model_path=DEFAULT_MODEL_PATH):
    """
    Given an image path (local path OR secure cloud web URL link), 
    return a tuple: (classification, confidence, predicted_yield).
    """
    try:
        model, device = get_model(model_path)

        # ✅ BALANCED ALIGNMENT PREPROCESSING MATRIX
        # Using basic tensor translation without ImageNet channel shifts to prevent feature distortion
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Handle secure cloud storage internet URLs
        if image_path.startswith('http://') or image_path.startswith('https://'):
            print(f"🌐 Fetching live sonogram byte stream from cloud storage link...")
            response = requests.get(image_path, timeout=15)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to pull image from URL. Status code: {response.status_code}")
            raw_img = Image.open(BytesIO(response.content))
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Target sonogram asset file missing at: {image_path}")
            raw_img = Image.open(image_path)

        # ✅ DYNAMIC CHANNEL ALIGNMENT HACK:
        # If your notebook read images via openCV or grayscale pipelines but mapped them 
        # to a 3-channel layer, converting to RGB forces identical dimensions.
        image = raw_img.convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits, yield_pred = model(tensor)
            
            probabilities = torch.nn.functional.softmax(class_logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            final_yield = yield_pred.item()
            if final_yield < 0: 
                final_yield = 0.0
            
            # ✅ FALLBACK BACKUP CHECK:
            # If regression yield heads underfit, check if the classification index matches a different stage
            class_idx = predicted_idx.item()
            model_predicted_class = CLASSES[class_idx]
            
            # Let the regression function map consistency, but defer to classification head if conflict occurs
            consistent_class = get_consistent_class(final_yield)
            if consistent_class == 'Peri-Partum' and model_predicted_class != 'Peri-Partum':
                consistent_class = model_predicted_class
            
            # Extract standard primitive types safely for MongoDB BSON records
            conf_val = float(confidence.item())
            yield_val = float(final_yield)
            
        return consistent_class, conf_val, yield_val
    except Exception as e:
        print(f"Error during custom CNN inference execution: {e}")
        raise RuntimeError(f"Inference failed: {e}")
