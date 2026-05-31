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
MODEL_PATH = "/tmp/cow_model.pth"

_cached = None


# =========================
# MODEL (FIXED STABLE VERSION)
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
            nn.MaxPool2d(2),
        )

        # FIX: avoids flatten mismatch issues
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.shared = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.class_head = nn.Linear(512, num_classes)

        # 🔥 FIXED REGRESSION HEAD (stabilized output 0–1)
        self.yield_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)

        class_logits = self.class_head(x)
        yield_pred = self.yield_head(x)

        return class_logits, yield_pred


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
# LOAD MODEL
# =========================
def load_model():
    global _cached

    if _cached is not None:
        return _cached

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

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
# IMAGE TRANSFORM (MATCH TRAINING)
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
# PREDICTION (FIXED OUTPUT)
# =========================
def predict_image(image_path):
    model, device = load_model()

    # Load image
    try:
        if image_path.startswith("http"):
            r = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return ("failed", 0.0, 0.0)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, yield_pred = model(image)

        probs = torch.softmax(class_logits, dim=1)
        conf, idx = torch.max(probs, 1)

        # =========================
        # 🔥 FIXED MILK YIELD SCALING
        # =========================
        yield_litres = yield_pred.item() * 30.0   # IMPORTANT FIX

        # clamp unrealistic values
        yield_litres = max(0.0, min(yield_litres, 40.0))

        return (
            CLASSES[idx.item()],
            float(conf.item()),
            float(yield_litres)
        )
