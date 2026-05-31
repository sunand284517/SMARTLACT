import os
import gdown
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
import requests

# =========================
# CLASS LABELS
# =========================
CLASSES = [
    'Dry Period',
    'Peak Lactation',
    'Late Lactation',
    'Fresh Cows',
    'Peri-Partum'
]

# =========================
# MODEL ARCHITECTURE
# =========================
class CowSonogramCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CowSonogramCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(6272, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.classification_head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))

        return self.classification_head(x)


# =========================
# MODEL CONFIG
# =========================
MODEL_ID = "1V8Lobs36IXWHwVs9C7Y01wxU-tBew6gb"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cow_model.pth")

_cached_model = None


# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists locally")
        return

    print("📥 Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={MODEL_ID}"

    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model download failed from Google Drive")

    print("✅ Model downloaded successfully")


# =========================
# LOAD MODEL (CACHE SAFE)
# =========================
def get_model():
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # IMPORTANT: strict=True ensures correct architecture match
    model.load_state_dict(state, strict=True)

    model.eval()

    _cached_model = (model, device)

    return _cached_model


# =========================
# IMAGE PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image_path):
    model, device = get_model()

    # Load image (URL or local)
    if image_path.startswith("http"):
        response = requests.get(image_path, timeout=15)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, idx = torch.max(probs, 1)

        label = CLASSES[idx.item()]
        conf = float(confidence.item())

    return label, conf
