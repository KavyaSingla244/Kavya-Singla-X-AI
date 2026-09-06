"""
Round 6 - Multi-Teacher Foundation Knowledge Distillation Pipeline (Native Soft-CE)
===================================================================================
1. Teacher Committee: OpenCLIP ViT-B/16 + OpenCLIP ViT-B/32 (91%+ zero-shot accuracy).
2. Generates calibrated teacher probability distributions for all 3,000 active training samples.
3. Student: Competition ResNet18Classifier initialized from best_model.pth (83.08%).
4. Loss: Native Soft-Target Cross-Entropy Distillation + Label-Smoothed Supervised Loss.
5. Invariance Stack: RandAugment + ColorJitter + CutMix/MixUp Dual Regularization.
6. Optimizer: SGD + Nesterov Momentum + Dynamic Model EMA (decay = 0.999).
"""

import math
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import open_clip
import tlc
from PIL import Image

# SIMD Multithreading
torch.set_num_threads(6)

BASE_DIR = Path(__file__).parent.absolute()
PROJECT_NAME = "Intel-Scene"
DATASET_NAME = "intel-scene"
TABLE_NAME = "train"
VAL_TABLE_NAME = "val"
BEST_MODEL_FILENAME = "best_model.pth"
NUM_CLASSES = 6
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street", "undefined"]

BATCH_SIZE = 32
EPOCHS = 28
INITIAL_LR = 0.005      # Stable fine-tuning learning rate
MIN_LR = 0.00005
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EMA_DECAY = 0.999
ALPHA_KD = 0.35         # 35% Soft Teacher Guidance + 65% Supervised CE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def resolve_image_path(p):
    s = str(p)
    if "<INTEL_SCENE_DATA>" in s:
        s = s.replace("<INTEL_SCENE_DATA>", str(BASE_DIR))
    return s


# ============================================================================
# STUDENT MODEL (Strict 3LC / Kaggle ResNet18 Architecture)
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
# DYNAMIC MODEL EMA
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
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(d).add_(param.data, alpha=1 - d)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.data.copy_(buffer.data)


# ============================================================================
# TEACHER KNOWLEDGE EXTRACTION (LOAD / EXTRACT)
# ============================================================================
def extract_teacher_probs(train_samples):
    cache_path = Path("teacher_probs.npy")
    if cache_path.exists():
        probs = np.load(cache_path)
        if len(probs) == len(train_samples):
            print(f"[OK] Loaded cached Teacher Probability Targets from {cache_path}: shape {probs.shape}")
            return probs

    logits_cache = Path("teacher_logits.npy")
    if logits_cache.exists():
        logits = np.load(logits_cache)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).astype(np.float32)
        np.save(cache_path, probs)
        print(f"[OK] Generated and saved Teacher Probs to {cache_path}: shape {probs.shape}")
        return probs

    print("\n" + "=" * 65)
    print("  Extracting Multi-Teacher OpenCLIP Dark Knowledge (Batched)...")
    print("=" * 65)

    teachers = []
    m32, _, prep32 = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    teachers.append((m32.to(device).eval(), prep32, 'ViT-B-32'))

    m16, _, prep16 = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    teachers.append((m16.to(device).eval(), prep16, 'ViT-B-16'))

    class_prompts = [
        'a photo of buildings, houses, residential and commercial city architecture',
        'a photo of a lush green forest, pine trees, woods, and wilderness foliage',
        'a photo of a glacier, blue icebergs, and cold frozen polar landscape',
        'a photo of a tall mountain peak, rocky cliffs, and alpine ridge',
        'a photo of the open sea, ocean water, coastal horizon, and waves',
        'a photo of an urban street, city avenue, road with traffic and sidewalks'
    ]

    tok = open_clip.get_tokenizer('ViT-B-32')
    text_tokens = tok(class_prompts).to(device)

    teacher_text_feats = []
    for model, _, _ in teachers:
        with torch.no_grad():
            tf = model.encode_text(text_tokens)
            tf /= tf.norm(dim=-1, keepdim=True)
            teacher_text_feats.append(tf)

    batch_size = 48
    all_probs = []

    for start_idx in tqdm(range(0, len(train_samples), batch_size), desc="Teacher Distillation"):
        batch_samples = train_samples[start_idx:start_idx + batch_size]
        pil_batch = []
        for s in batch_samples:
            img_path = resolve_image_path(s["image"])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (150, 150), (128, 128, 128))
            pil_batch.append(img)

        batch_ensemble_probs = torch.zeros(len(pil_batch), NUM_CLASSES, device=device)
        with torch.no_grad():
            for (t_model, t_prep, _), t_tf in zip(teachers, teacher_text_feats):
                t_tensors = torch.stack([t_prep(img) for img in pil_batch]).to(device)
                img_f = t_model.encode_image(t_tensors)
                img_f /= img_f.norm(dim=-1, keepdim=True)
                logits = 100.0 * (img_f @ t_tf.T)
                probs = torch.softmax(logits, dim=-1)
                batch_ensemble_probs += probs

            batch_ensemble_probs /= len(teachers)
        all_probs.append(batch_ensemble_probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    np.save(cache_path, all_probs)
    return all_probs


# ============================================================================
# DISTILLATION DATASET & TRANSFORMS
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


class DistillationDataset(Dataset):
    def __init__(self, samples, teacher_probs, transform=None):
        self.samples = samples
        self.teacher_probs = teacher_probs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = resolve_image_path(sample["image"])
        hard_label = sample["label"]
        t_probs = self.teacher_probs[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (150, 150), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, hard_label, t_probs


class ValDataset(Dataset):
    def __init__(self, table, transform=None):
        self.rows = [r for r in table.table_rows if r.get("label", 6) < 6]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = resolve_image_path(row["image"])
        label = row["label"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (150, 150), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_model(eval_model, dataloader):
    eval_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = eval_model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def train_knowledge_distillation():
    print("=" * 65)
    print("  Round 6: Teacher-Student Knowledge Distillation Training")
    print("=" * 65)

    tlc.register_project_url_alias(
        token="INTEL_SCENE_DATA",
        path=str(BASE_DIR),
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

    all_train_rows = list(train_table.table_rows)
    active_samples = [r for r in all_train_rows if r.get("weight", 1.0) > 0 and r.get("label", 6) < 6]
    print(f"Loaded {len(active_samples)} active samples for Knowledge Distillation.")

    # 1. Precompute/Extract Teacher Soft Probability Targets
    teacher_probs = extract_teacher_probs(active_samples)

    train_dataset = DistillationDataset(active_samples, teacher_probs, transform=train_transform)
    val_dataset = ValDataset(val_table, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Student from snapshot_r5_epoch_22_83.1.pth / best_model.pth
    student = ResNet18Classifier(num_classes=NUM_CLASSES)
    init_weights = "snapshot_r5_epoch_22_83.1.pth" if Path("snapshot_r5_epoch_22_83.1.pth").exists() else BEST_MODEL_FILENAME
    student.load_state_dict(torch.load(init_weights, map_location=device))
    print(f"[OK] Student initialized from {init_weights} (Warm Start)")
    student = student.to(device)
    ema = ModelEMA(student, decay=EMA_DECAY)

    optimizer = torch.optim.SGD(
        student.parameters(),
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

    # Loss Functions: Native Soft-Target CrossEntropyLoss
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    kd_criterion = nn.CrossEntropyLoss()

    initial_acc = evaluate_model(student, val_loader)
    print(f"\nBaseline Starting Val Acc: {initial_acc:.2f}%\n")

    best_val_acc = initial_acc
    best_student_state = copy.deepcopy(student.state_dict())

    for epoch in range(1, EPOCHS + 1):
        student.train()
        running_loss = 0.0
        running_kd = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} (lr={current_lr:.5f})")
        for images, hard_labels, t_probs in pbar:
            images = images.to(device)
            hard_labels = hard_labels.to(device)
            t_probs = t_probs.to(device)

            optimizer.zero_grad()
            student_logits = student(images)

            # 1. Supervised Cross-Entropy Loss
            loss_ce = ce_criterion(student_logits, hard_labels)

            # 2. Native Soft Target Distillation Loss
            loss_kd = kd_criterion(student_logits, t_probs)

            # Combined Total Loss
            total_loss = (1 - ALPHA_KD) * loss_ce + ALPHA_KD * loss_kd
            total_loss.backward()

            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            ema.update(student)

            running_loss += total_loss.item() * images.size(0)
            running_kd += loss_kd.item() * images.size(0)
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", kd=f"{loss_kd.item():.4f}")

        scheduler.step()

        # Evaluate Raw and EMA
        raw_val_acc = evaluate_model(student, val_loader)
        ema_val_acc = evaluate_model(ema.ema_model, val_loader)
        chosen_acc = max(raw_val_acc, ema_val_acc)
        chosen_type = "EMA" if ema_val_acc >= raw_val_acc else "Raw"

        print(f"Epoch {epoch:02d}/{EPOCHS} | Val Acc: {chosen_acc:.2f}% ({chosen_type}) [Raw: {raw_val_acc:.2f}%, EMA: {ema_val_acc:.2f}%] | (Peak: {best_val_acc:.2f}%)")

        chosen_state = copy.deepcopy(ema.ema_model.state_dict() if chosen_type == "EMA" else student.state_dict())

        if chosen_acc > best_val_acc:
            best_val_acc = chosen_acc
            best_student_state = copy.deepcopy(chosen_state)
            torch.save(best_student_state, BEST_MODEL_FILENAME)
            print(f"  --> 🌟 [NEW BEST] Saved checkpoint ({best_val_acc:.2f}%) to {BEST_MODEL_FILENAME}")

        # Save snapshot if high performing (>= 82.5%)
        if chosen_acc >= 82.5:
            snap_path = f"snapshot_r6_distill_epoch_{epoch}_{chosen_acc:.1f}.pth"
            torch.save(chosen_state, snap_path)
            print(f"  --> 📸 [SNAPSHOT] Saved {snap_path}")

    # Final save
    torch.save(best_student_state, BEST_MODEL_FILENAME)
    print("\n" + "=" * 65)
    print(f"  Round 6 Distillation Finished! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    train_knowledge_distillation()
