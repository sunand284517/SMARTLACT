import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image
import gdown
import os

# =========================
# CLASSES (DO NOT CHANGE ORDER)
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
# FIXED MODEL (MUST MATCH TRAINING)
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

        self.shared = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU()
        )

        self.class_head = nn.Linear(512, num_classes)
        self.yield_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)

        class_logits = self.class_head(x)
        yield_pred = self.yield_head(x)

        return class_logits, yield_pred


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


# =========================
# LOAD MODEL (SAFE)
# =========================
def load_model():
    global _cached

    if _cached is not None:
        return _cached

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(len(CLASSES)).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # FIXED: handle both formats safely
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # IMPORTANT: strict=False avoids crashes after small mismatches
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    _cached = (model, device)

    print("✅ Model loaded and ready")
    return _cached


# =========================
# TRANSFORM (MUST MATCH TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# PREDICTION (ROBUST)
# =========================
def predict_image(image_path):
    try:
        model, device = load_model()

        # load image
        if image_path.startswith("http"):
            r = requests.get(image_path, timeout=10)
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            class_logits, yield_pred = model(x)

            probs = torch.softmax(class_logits, dim=1)
            conf, idx = torch.max(probs, 1)

            yield_value = yield_pred.item()

            # SAFETY FIX: prevent negative/zero nonsense
            yield_value = max(0.1, float(yield_value))

            return {
                "classification": CLASSES[idx.item()],
                "confidence": round(float(conf.item()), 4),
                "yield_litres": round(yield_value, 2)
            }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
