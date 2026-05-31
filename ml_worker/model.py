import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gdown
import os
import requests
from io import BytesIO
from PIL import Image

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
# 🔥 MUST MATCH TRAINED MODEL EXACTLY
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

        # ⚠️ THIS NAME MUST MATCH YOUR TRAINED MODEL
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

        return class_logits, yield_pred


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


# =========================
# LOAD MODEL SAFE
# =========================
def load_model():
    global _cached

    if _cached:
        return _cached

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(len(CLASSES)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()

    _cached = (model, device)
    return _cached


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# PREDICT
# =========================
def predict_image(image_url):
    model, device = load_model()

    if image_url.startswith("http"):
        img = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    else:
        img = Image.open(image_url).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, yield_pred = model(x)

        probs = torch.softmax(class_logits, dim=1)
        conf, idx = torch.max(probs, 1)

        milk = float(yield_pred.item())

        # safety clamp
        milk = max(0.5, min(milk, 50.0))

        return {
            "class": CLASSES[idx.item()],
            "confidence": float(conf.item()),
            "milk_yield_liters": milk
        }
