"""
Zero-Disk External Generalization Benchmark (RAM Streaming)
==========================================================
Tests the ResNet-18 model against completely unseen, real-world natural scene images
streamed directly into RAM via Unsplash CDN without downloading or cluttering local disk storage.

Evaluates:
- ResNet-18 Student Prediction
- Softmax Confidence Score (%)
- Real-World Out-of-Domain Generalization Accuracy
"""

import io
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Multithreading
torch.set_num_threads(4)

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


EXTERNAL_TEST_SUITE = [
    # BUILDINGS (0)
    {"url": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=400", "ground_truth": 0, "description": "Modern glass skyscraper highrise"},
    {"url": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=400", "ground_truth": 0, "description": "Classic brick architecture buildings"},
    {"url": "https://images.unsplash.com/photo-1541888946425-d0fbb18086f6?w=400", "ground_truth": 0, "description": "Historic European city palace building"},
    {"url": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=400", "ground_truth": 0, "description": "Modern residential luxury estate villa"},

    # FOREST (1)
    {"url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=400", "ground_truth": 1, "description": "Dense sunlit green forest trees"},
    {"url": "https://images.unsplash.com/photo-1511497584788-87676104235f?w=400", "ground_truth": 1, "description": "Misty foggy woodland forest canopy"},
    {"url": "https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=400", "ground_truth": 1, "description": "Golden autumn forest foliage canopy"},
    {"url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400", "ground_truth": 1, "description": "Towering redwood forest evergreen"},

    # GLACIER (2)
    {"url": "https://images.unsplash.com/photo-1517760444937-f6397edcbbcd?w=400", "ground_truth": 2, "description": "Blue glacial ice shelf and frozen icebergs"},
    {"url": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=400", "ground_truth": 2, "description": "Polar arctic glacier ice sheet landscape"},
    {"url": "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=400", "ground_truth": 2, "description": "Glacial ice wall with icy water lagoon"},
    {"url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400", "ground_truth": 3, "description": "Rugged alpine mountain peaks and ridges"},

    # MOUNTAIN (3)
    {"url": "https://images.unsplash.com/photo-1486870591958-9b9d0d1dda99?w=400", "ground_truth": 3, "description": "Majestic snowy mountain summit range"},
    {"url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400", "ground_truth": 3, "description": "Yosemite granite mountain cliffs"},
    {"url": "https://images.unsplash.com/photo-1465919292275-c60ba49da6ae?w=400", "ground_truth": 3, "description": "Sharp mountain ridge rock formation"},
    {"url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400", "ground_truth": 1, "description": "Misty mountain forest wilderness valley"},

    # SEA (4)
    {"url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400", "ground_truth": 4, "description": "Tropical sandy beach turquoise sea waves"},
    {"url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400", "ground_truth": 4, "description": "Deep blue open ocean sea water ripples"},
    {"url": "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=400", "ground_truth": 4, "description": "Coastal ocean sea cliffs and surf"},
    {"url": "https://images.unsplash.com/photo-1468581264429-2548ef9eb732?w=400", "ground_truth": 4, "description": "Sunny sea horizon with gentle waves"},

    # STREET (5)
    {"url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=400", "ground_truth": 5, "description": "Urban city street road with crosswalk and traffic"},
    {"url": "https://images.unsplash.com/photo-1477959858617-67f30bc75b82?w=400", "ground_truth": 5, "description": "Downtown Chicago street avenue at dusk"},
    {"url": "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=400", "ground_truth": 5, "description": "Neon lit night city street with cars"},
    {"url": "https://images.unsplash.com/photo-1508873696983-2df5293cb32b?w=400", "ground_truth": 5, "description": "European cobblestone street alleyway"}
]


def run_benchmark():
    print("=" * 82)
    print("  Zero-Disk External Generalization Benchmark (In-Memory Streaming)")
    print("=" * 82)

    model = ResNet18Classifier(num_classes=6)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    print("[OK] ResNet-18 Student Model loaded from best_model.pth (83.08% checkpoint)")

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    results = []
    correct_count = 0
    total_evaluated = 0

    print(f"\nStreaming {len(EXTERNAL_TEST_SUITE)} external unseen benchmark images into RAM...")
    print("-" * 82)
    print(f"{'#':<3} | {'Class (GT)':<12} | {'Predicted':<12} | {'Conf %':<8} | {'Match':<6} | {'Scene Description'}")
    print("-" * 82)

    for idx, item in enumerate(EXTERNAL_TEST_SUITE, 1):
        url = item["url"]
        gt = item["ground_truth"]
        gt_name = CLASSES[gt]
        desc = item["description"]

        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                img_bytes = response.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"[{idx:02d}] Fetch error for {desc}: {e}")
            continue

        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item() * 100.0

        pred_name = CLASSES[pred_idx]
        is_match = (pred_idx == gt)
        if is_match:
            correct_count += 1
        total_evaluated += 1

        match_str = "✅ YES" if is_match else "❌ NO"
        print(f"{idx:<3} | {gt_name:<12} | {pred_name:<12} | {conf:5.1f}%  | {match_str:<6} | {desc}")
        results.append({
            "gt": gt_name,
            "pred": pred_name,
            "conf": conf,
            "match": is_match,
            "desc": desc
        })

    print("-" * 82)
    accuracy = (correct_count / total_evaluated * 100.0) if total_evaluated > 0 else 0
    print(f"\n🎯 External Benchmark Generalization Accuracy: {accuracy:.2f}% ({correct_count}/{total_evaluated} correct)")
    print("=" * 82)


if __name__ == "__main__":
    run_benchmark()
