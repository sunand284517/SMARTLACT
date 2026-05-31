import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image
import gdown
import os

# =========================
# LABELS
# =========================
CLASSES = [
    'Dry Period',
    'Peak Lactation',
    'Late Lactation',
    'Fresh Cows',
    'Peri-Partum'
]

MODEL_ID = "1V8Lobs36IXWHwVs9C7Y01wxU-tBew6gb"
MODEL_PATH = "cow_model.pth"

_cached = None


# =========================
# MODEL (TRAINING MATCH)
# =========================
class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU()
        )

        self.classification_head = nn.Linear(512, num_classes)
        self.regression_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)

        class_logits = self.classification_head(x)
        yield_pred = self.regression_head(x)

        # ALWAYS RETURN EXACTLY 2 VALUES
        return (class_logits, yield_pred)


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


# =========================
# LOAD MODEL (SAFE + CACHE)
# =========================
def load_model():
    global _cached

    if _cached is not None:
        return _cached

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # SAFE CHECKPOINT HANDLING
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _cached = (model, device)

    print("✅ Model loaded successfully")

    return _cached


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# PREDICT FUNCTION (ROBUST)
# =========================
def predict_image(image_path):
    model, device = load_model()

    # Load image safely
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Image loading error: {str(e)}"
        }

    image = transform(image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = model(image)

            # HARD SAFETY CHECK
            if not isinstance(outputs, (tuple, list)):
                return {
                    "status": "failed",
                    "error": f"Invalid output type: {type(outputs)}"
                }

            if len(outputs) != 2:
                return {
                    "status": "failed",
                    "error": f"Expected 2 outputs, got {len(outputs)}"
                }

            class_logits, yield_pred = outputs

            probs = torch.softmax(class_logits, dim=1)
            conf, idx = torch.max(probs, 1)

            return {
                "status": "success",
                "classification": CLASSES[idx.item()],
                "confidence": float(conf.item()),
                "predicted_yield": float(yield_pred.item())
            }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
