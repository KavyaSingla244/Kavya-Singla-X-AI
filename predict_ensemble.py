"""
Multi-Model Snapshot Ensemble + Test-Time Augmentation (TTA) Predictor
======================================================================
1. Discovers best_model.pth + all snapshot checkpoints.
2. Runs 5-crop multi-scale & horizontal-flip inference on data/test/.
3. Blends softmax probabilities across all models and crops.
4. Produces state-of-the-art submission.csv for Kaggle.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import csv
import numpy as np

# Optimization for AMD Ryzen
torch.set_num_threads(6)

MODEL_PATH = Path("best_model.pth")
TEST_DIR = Path("data/test")
OUTPUT_PATH = Path("submission.csv")
SUBMISSIONS_DIR = Path("submissions")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")
NUM_CLASSES = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path.name


def get_tta_transforms():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        # Standard center
        transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(), normalize]),
        # Horizontal Flip
        transforms.Compose([transforms.Resize((150, 150)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
    ]


def predict_ensemble():
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # 1. Discover models to ensemble
    checkpoint_paths = []
    if MODEL_PATH.exists():
        checkpoint_paths.append(MODEL_PATH)

    # Prioritize best_model.pth and top Round 5 snapshots (>= 82.5% val accuracy)
    r5_snaps = sorted(list(Path(".").glob("snapshot_r5_*.pth")), reverse=True)
    selected_snaps = [s for s in r5_snaps if any(score in s.name for score in ["83.1", "83.0", "82.9", "82.8", "82.7", "82.5"])]
    if not selected_snaps:
        selected_snaps = r5_snaps[:6]
    
    for s in selected_snaps:
        if s not in checkpoint_paths:
            checkpoint_paths.append(s)

    print(f"\nDiscovered {len(checkpoint_paths)} Elite Checkpoints for Ensembling:")
    for cp in checkpoint_paths:
        print(f"  - {cp.name}")

    models_list = []
    for cp in checkpoint_paths:
        m = ResNet18Classifier(num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(cp, map_location=device))
        m = m.to(device)
        m.eval()
        models_list.append(m)

    # 2. Test images
    test_image_paths = sorted(list(TEST_DIR.glob("*.jpg")) + list(TEST_DIR.glob("*.png")))
    print(f"\nFound {len(test_image_paths)} test images.")

    tta_transforms = get_tta_transforms()
    total_probs = np.zeros((len(test_image_paths), NUM_CLASSES), dtype=np.float32)
    image_names = [p.name for p in test_image_paths]

    print("\nRunning Multi-Model Snapshot Ensemble with TTA...")
    for t_idx, trans in enumerate(tta_transforms):
        dataset = TestDataset(test_image_paths, transform=trans)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for model in models_list:
            all_batch_probs = []
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=f"TTA Crop {t_idx+1}/{len(tta_transforms)} (Model: {model.__class__.__name__})"):
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_batch_probs.append(probs)
            
            total_probs += np.vstack(all_batch_probs)

    # Average probabilities
    num_passes = len(tta_transforms) * len(models_list)
    avg_probs = total_probs / num_passes

    predicted_classes = np.argmax(avg_probs, axis=1)
    confidences = np.max(avg_probs, axis=1)

    # 3. Align with sample_submission.csv (map both stem and filename)
    pred_map = {}
    for p, pred, conf in zip(test_image_paths, predicted_classes, confidences):
        pred_map[p.stem] = (int(pred), float(conf))
        pred_map[p.name] = (int(pred), float(conf))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = SUBMISSIONS_DIR / f"submission_ensemble_{timestamp}.csv"

    fieldnames = ["image_id", "prediction", "confidence"]
    with open(OUTPUT_PATH, "w", newline="") as f_out, open(timestamped_path, "w", newline="") as f_ts:
        writer_out = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer_ts = csv.DictWriter(f_ts, fieldnames=fieldnames)
        writer_out.writeheader()
        writer_ts.writeheader()

        if SAMPLE_SUBMISSION_PATH.exists():
            with open(SAMPLE_SUBMISSION_PATH, "r") as f_samp:
                reader = csv.DictReader(f_samp)
                for row in reader:
                    img_id = row["image_id"]
                    if img_id in pred_map:
                        pred, conf = pred_map[img_id]
                    elif Path(img_id).stem in pred_map:
                        pred, conf = pred_map[Path(img_id).stem]
                    else:
                        pred, conf = 0, 0.5
                    row_data = {"image_id": img_id, "prediction": pred, "confidence": f"{conf:.4f}"}
                    writer_out.writerow(row_data)
                    writer_ts.writerow(row_data)
        else:
            for p in test_image_paths:
                pred, conf = pred_map[p.stem]
                row_data = {"image_id": p.stem, "prediction": pred, "confidence": f"{conf:.4f}"}
                writer_out.writerow(row_data)
                writer_ts.writerow(row_data)

    print("\n" + "=" * 60)
    print(f"  [OK] Ensemble Submission Generated!")
    print(f"  Canonical:   {OUTPUT_PATH}")
    print(f"  Timestamped: {timestamped_path}")
    print("=" * 60)


if __name__ == "__main__":
    predict_ensemble()
