"""
Active Data Curation & Purification for 3LC Intel Scene Challenge (Round 4)
===========================================================================
1. Loads the latest high-accuracy checkpoint (best_model.pth, 80.25% Val Acc).
2. Scores all 6,000 undefined pool images with batch acceleration.
3. Computes top prediction probability P1, second probability P2, and decision margin (P1 - P2).
4. Selects the top 400 purest, highest-margin candidates per class (2,400 total).
5. Writes a new 3LC Table revision with exactly 3,000 active rows (600 seed + 2,400 curated).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tlc
import sys

PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
TABLE_NAME = "train"
NUM_CLASSES = 6
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]
MODEL_PATH = Path("best_model.pth")
CURATE_PER_CLASS = 400  # 400 * 6 = 2,400 added -> 600 seed + 2,400 = 3,000 total
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


class UndefinedPoolDataset(Dataset):
    def __init__(self, table, indices, transform=None):
        self.table = table
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_idx = self.indices[i]
        sample = self.table[orig_idx]
        img_path = sample["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (150, 150), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, orig_idx, str(img_path)


def curate_and_commit():
    print("=" * 60)
    print("  Starting Active Data Curation & Self-Distillation (Round 4)")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model {MODEL_PATH} not found. Run train.py first.")

    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[OK] 80.25% Model loaded from {MODEL_PATH}")

    base_path = Path(__file__).parent
    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(base_path.absolute()),
        project=PROJECT_NAME,
    )

    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=TABLE_NAME,
    ).latest()
    print(f"[OK] Loaded train table with {len(train_table)} rows (URL: {train_table.url})")

    eval_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    undefined_indices = list(range(600, len(train_table)))
    dataset = UndefinedPoolDataset(train_table, undefined_indices, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nScoring {len(undefined_indices)} pool images in batches...")
    scored_candidates = []

    with torch.no_grad():
        for images, orig_indices, img_paths in tqdm(dataloader, desc="Scoring Pool (Round 4)"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for prob, orig_idx, img_p in zip(probs, orig_indices.numpy(), img_paths):
                sorted_idx = np.argsort(prob)[::-1]
                top1_c = int(sorted_idx[0])
                top1_p = float(prob[top1_c])
                top2_p = float(prob[sorted_idx[1]])
                margin = top1_p - top2_p

                # Confidence + Margin composite score
                composite_score = (top1_p * 0.6) + (margin * 0.4)

                scored_candidates.append({
                    "table_idx": int(orig_idx),
                    "image_path": img_p,
                    "predicted_label": top1_c,
                    "confidence": top1_p,
                    "margin": margin,
                    "score": composite_score,
                })

    print(f"\nSuccessfully scored {len(scored_candidates)} samples.")
    print(f"Selecting top {CURATE_PER_CLASS} purest samples per class...")

    selected_indices_set = set()
    new_labels_map = {}

    for c in range(NUM_CLASSES):
        class_c_candidates = [cand for cand in scored_candidates if cand["predicted_label"] == c]
        class_c_candidates.sort(key=lambda x: x["score"], reverse=True)
        chosen = class_c_candidates[:CURATE_PER_CLASS]
        avg_conf = np.mean([x['confidence'] for x in chosen]) if chosen else 0.0
        avg_margin = np.mean([x['margin'] for x in chosen]) if chosen else 0.0
        print(f"  Class {c} ({CLASSES[c]:<9}): {len(class_c_candidates):4d} candidates -> selected {len(chosen):3d} (avg conf: {avg_conf:.3f}, avg margin: {avg_margin:.3f})")

        for item in chosen:
            selected_indices_set.add(item["table_idx"])
            new_labels_map[item["table_idx"]] = item["predicted_label"]

    print(f"\nTotal newly labeled samples: {len(selected_indices_set)}")
    total_active = 600 + len(selected_indices_set)
    print(f"Total active training samples: {total_active} / 3000 budget cap")

    schemas = {
        "id": tlc.Schema(value=tlc.Int32Value(), writable=False),
        "image": tlc.ImagePath,
        "label": tlc.CategoricalLabel("label", classes=CLASSES),
        "weight": tlc.SampleWeightSchema(),
    }

    table_writer = tlc.TableWriter(
        table_name=TABLE_NAME,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Intel Scene train set with 600 seed + 2,400 purified active samples (Round 4)",
        column_schemas=schemas,
        if_exists="overwrite",
    )

    table_rows_list = list(train_table.table_rows)
    weight1_count = 0

    for idx, raw_row in enumerate(table_rows_list):
        img_val = raw_row["image"]

        if idx < 600:
            lbl = raw_row["label"]
            wt = 1.0
        elif idx in selected_indices_set:
            lbl = new_labels_map[idx]
            wt = 1.0
        else:
            lbl = 6  # undefined
            wt = 0.0

        if wt > 0:
            weight1_count += 1

        table_writer.add_row({
            "id": idx,
            "image": img_val,
            "label": lbl,
            "weight": wt,
        })

    new_table = table_writer.finalize()
    print(f"\n[OK] New Table Revision Created!")
    print(f"  URL: {new_table.url}")
    print(f"  Total weight=1.0 rows: {weight1_count} (Cap: 3000)")
    assert weight1_count <= 3000, f"Error: Exceeded budget cap! {weight1_count} > 3000"
    print("=" * 60)
    print("  Round 4 Dataset Revision Ready!")
    print("=" * 60)


if __name__ == "__main__":
    curate_and_commit()
