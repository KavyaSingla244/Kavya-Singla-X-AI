"""
Round 5 - Warm-Start Continuous Learning & Fine-Tuning
=====================================================
1. Initializes directly from our 81.25% checkpoint (best_model.pth).
2. Uses the Foundation-Model Purified 3,000 Dataset (3LC Table Revision).
3. Applies RandAugment + ColorJitter + CutMix/MixUp Dual Regularization.
4. SGD + Nesterov Momentum with gentle Cosine Decay (0.012 -> 0.0001).
5. Dynamic Model EMA tracking for maximum test generalization.
"""

import math
import copy
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
import tlc
from PIL import Image

# CPU SIMD Optimization
torch.set_num_threads(6)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
TABLE_NAME = "train"
VAL_TABLE_NAME = "val"
BEST_MODEL_FILENAME = "best_model.pth"
NUM_CLASSES = 6
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]

BATCH_SIZE = 32
EPOCHS = 26
INITIAL_LR = 0.012      # Warm-start fine-tuning rate
MIN_LR = 0.0001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EMA_DECAY = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# MODEL
# ============================================================================
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


# ============================================================================
# MODEL EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    def update(self, model):
        self.step += 1
        # Warm-start step decay
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(d).add_(param.data, alpha=1 - d)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.data.copy_(buffer.data)


# ============================================================================
# CUTMIX & MIXUP
# ============================================================================
cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=1.0)
mixup = v2.MixUp(num_classes=NUM_CLASSES, alpha=0.3)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


# ============================================================================
# DATASET & TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomCrop(150, padding=8, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, table, transform=None, filter_weight=True):
        self.table = table
        self.transform = transform
        rows = list(table.table_rows)
        if filter_weight:
            self.indices = [i for i, r in enumerate(rows) if r.get("weight", 1.0) > 0 and r.get("label", 6) < 6]
        else:
            self.indices = [i for i, r in enumerate(rows) if r.get("label", 6) < 6]
        print(f"Loaded TableDataset: {len(self.indices)} active samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.table[self.indices[idx]]
        img_path = row["image"]
        label = row["label"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (150, 150), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(eval_model, dataloader):
    eval_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = eval_model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train_warmstart():
    print("=" * 65)
    print("  Round 5: Warm-Start Fine-Tuning from 81.25% Checkpoint")
    print("=" * 65)

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

    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=VAL_TABLE_NAME,
    ).latest()

    train_dataset = TableDataset(train_table, transform=train_transform, filter_weight=True)
    val_dataset = TableDataset(val_table, transform=val_transform, filter_weight=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. Warm-Start Model Initialization
    model = ResNet18Classifier(num_classes=NUM_CLASSES)
    if Path(BEST_MODEL_FILENAME).exists():
        model.load_state_dict(torch.load(BEST_MODEL_FILENAME, map_location=device))
        print(f"[OK] Initialized weights from {BEST_MODEL_FILENAME} (81.25% baseline)")
    else:
        print(f"[WARN] {BEST_MODEL_FILENAME} not found. Initializing fresh.")
    
    model = model.to(device)
    ema = ModelEMA(model, decay=EMA_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR,
    )

    # Evaluate baseline accuracy before fine-tuning
    initial_val_acc = evaluate_model(model, val_loader)
    print(f"\nInitial Warm-Start Baseline Val Acc: {initial_val_acc:.2f}%\n")

    best_val_acc = initial_val_acc
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (lr={current_lr:.5f})")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            images, labels = cutmix_or_mixup(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Evaluate Raw and EMA
        raw_val_acc = evaluate_model(model, val_loader)
        ema_val_acc = evaluate_model(ema.ema_model, val_loader)
        chosen_acc = max(raw_val_acc, ema_val_acc)
        chosen_type = "EMA" if ema_val_acc >= raw_val_acc else "Raw"

        print(f"Epoch {epoch}/{EPOCHS} - Val Acc: {chosen_acc:.2f}% ({chosen_type}) [Raw: {raw_val_acc:.2f}%, EMA: {ema_val_acc:.2f}%] (Peak: {best_val_acc:.2f}%)")

        chosen_state = copy.deepcopy(ema.ema_model.state_dict() if chosen_type == "EMA" else model.state_dict())

        if chosen_acc > best_val_acc:
            best_val_acc = chosen_acc
            best_model_state = copy.deepcopy(chosen_state)
            torch.save(best_model_state, BEST_MODEL_FILENAME)
            print(f"  --> 🌟 [NEW BEST] Saved checkpoint ({best_val_acc:.2f}%) to {BEST_MODEL_FILENAME}")

        # Save snapshot if high performing (>= 81.0%)
        if chosen_acc >= 81.0:
            snap_path = f"snapshot_r5_epoch_{epoch}_{chosen_acc:.1f}.pth"
            torch.save(chosen_state, snap_path)
            print(f"  --> 📸 [SNAPSHOT] Saved {snap_path}")

    # Final save
    torch.save(best_model_state, BEST_MODEL_FILENAME)
    print("\n" + "=" * 65)
    print(f"  Round 5 Warm-Start Finished! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    train_warmstart()
