import os
import requests
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

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

        class_logits = self.classification_head(x)
        return class_logits


# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cow_model.pth")

_cached = None

# =========================
# LOAD MODEL
# =========================
def get_model():
    global _cached

    if _cached is not None:
        return _cached

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please upload correct cow_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CowSonogramCNN(num_classes=len(CLASSES)).to(device)

    state = torch.load(MODEL_PATH, map_location=device)

    # STRICT LOADING (IMPORTANT FIX)
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
# PREDICTION FUNCTION
# =========================
def predict_image(image_path):
    model, device = get_model()

    # Load image
    if image_path.startswith("http"):
        response = requests.get(image_path, timeout=15)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(probs, 1)

        label = CLASSES[predicted_idx.item()]
        conf = float(confidence.item())

    return label, conf
