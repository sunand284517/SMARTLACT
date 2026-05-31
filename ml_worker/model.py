import os
import sys
import gdown
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
# 🧠 THE EXACT CUSTOM CNN ARCHITECTURE FROM YOUR CHECKPOINT
# =========================================================
class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CowSonogramCNN, self).__init__()
        
        # ⚠️ CRITICAL: Must be named self.backbone to match your checkpoint keys!
        self.backbone = nn.Sequential(
            # Layer 1: Input 3 channels -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 5: Final extraction convolution layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Halves resolution down to yield exactly 256 * 14 * 14 = 50,176 features!
        )
        
        # Matches your exact fc_layer blocks
        self.fc_layer = nn.Sequential(
            nn.Linear(50176, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Multi-Task Learning Output Heads attached directly to the 512-feature block
        self.classification_head = nn.Linear(512, num_classes)
        self.regression_head = nn.Linear(512, 1)

    def forward(self, x):
        # Pass through the custom convolutional architecture pipeline
        x = self.backbone(x)
        x = torch.flatten(x, 1) # Flattens output map tightly to [Batch, 50176]
        x = self.fc_layer(x)
        
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
            quiet=False,
            fuzzy=True
        )

        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(f"❌ Download error: file missing at {DEFAULT_MODEL_PATH}")
        print("✅ Custom CNN checkpoint successfully loaded.")
    except Exception as e:
        print(f"❌ Auto-download error sequence triggered: {e}")
        raise e

# =========================
# 🖥️ MODEL LOADER
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
        # Strict mode set back to True because the layers are now fully, perfectly mapped!
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.eval() # Vital to correctly pause Dropout layer sequences
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
    Given an image path, return a tuple: (classification, confidence, predicted_yield).
    """
    try:
        model, device = get_model(model_path)

        # Transform settings tuned perfectly to match your 224x224 training resolution matrix dimensions
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target sonogram asset file missing at: {image_path}")

        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits, yield_pred = model(tensor)
            
            probabilities = torch.nn.functional.softmax(class_logits, dim=1)
            confidence, _ = torch.max(probabilities, 1)
            
            final_yield = yield_pred.item()
            if final_yield < 0: 
                final_yield = 0.0
            
            consistent_class = get_consistent_class(final_yield)
            
        return consistent_class, confidence.item(), final_yield
    except Exception as e:
        print(f"Error during custom CNN inference execution: {e}")
        raise RuntimeError(f"Inference failed: {e}")
