"""
Test Custom Image / Photos with the Trained Scene Classifier
============================================================
Usage:
    python predict_custom.py <path_to_image_or_folder>

Example:
    python predict_custom.py my_photo.jpg
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_PATH = Path("best_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def predict_image(image_path, model, transform):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    print("\n" + "=" * 50)
    print(f"  Image: {image_path}")
    print(f"  🎯 PREDICTED CLASS: {CLASSES[top_idx].upper()} ({probs[top_idx]*100:.1f}%)")
    print("=" * 50)
    print("  Full Probability Distribution:")
    for i, c in enumerate(CLASSES):
        bar = "█" * int(probs[i] * 25)
        print(f"   [{i}] {c:<10}: {probs[i]*100:5.1f}%  {bar}")
    print("=" * 50 + "\n")


def main():
    if not MODEL_PATH.exists():
        print(f"Model {MODEL_PATH} not found. Train the model first.")
        return

    model = ResNet18Classifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if len(sys.argv) < 2:
        print("Usage: python predict_custom.py <path_to_image_or_folder>")
        print("Testing on a sample validation image...")
        val_sample = Path("data/val/mountain").glob("*.jpg")
        first_img = next(val_sample, None)
        if first_img:
            predict_image(first_img, model, transform)
        return

    target = Path(sys.argv[1])
    if target.is_file():
        predict_image(target, model, transform)
    elif target.is_dir():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_file in target.glob(ext):
                predict_image(img_file, model, transform)
    else:
        print(f"Path not found: {target}")


if __name__ == "__main__":
    main()
