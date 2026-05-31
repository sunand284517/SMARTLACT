import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image
import gdown

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

# =========================
# MODEL ARCHITECTURE (MUST MATCH TRAINED MODEL)
# =========================
class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CowSonogramCNN, self).__init__()

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

        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =========================
# GOOGLE DRIVE MODEL CONFIG
# =========================
MODEL_ID = "1V8Lobs36IXWHwVs9C7Y01wxU-tBew6gb"
MODEL_PATH = "cow_model.pth"

_cached_model = None


# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if os.path.exists(MODEL_PATH):
        return

    print("📥 Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise Exception("Model download failed")


# =========================
# LOAD MODEL
# =========================
def load_model():
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # IMPORTANT: must match checkpoint EXACTLY
    model.load_state_dict(state, strict=True)

    model.eval()

    _cached_model = (model, device)

    return _cached_model


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# PREDICT FUNCTION
# =========================
def predict_image(image_path):
    model, device = load_model()

    # Load image
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)

        conf, idx = torch.max(probs, 1)

    return CLASSES[idx.item()], float(conf.item())
