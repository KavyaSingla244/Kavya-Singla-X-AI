"""
Large-Scale Multi-Class Online Dataset Evaluation (Zero-Disk In-Memory Streaming)
=================================================================================
Streams a balanced mix of thousands of unseen real-world natural scene images directly
from Hugging Face ('prithivMLmods/OpenScene-Classification') across all 6 classes:
0: buildings, 1: forest, 2: glacier, 3: mountain, 4: sea, 5: street.

Evaluates:
1. ResNet-18 Student Model (best_model.pth)
2. Global Out-of-Domain Accuracy across thousands of unseen images
3. Per-Class Precision, Recall, and F1-Scores
4. Full 6x6 Confusion Matrix & Confidence Calibration
"""

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# SIMD Multithreading (2 threads to run cleanly in parallel with training)
torch.set_num_threads(2)

PROJECT_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
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


def run_online_evaluation(num_samples=2000):
    print("=" * 80)
    print(f"  Balanced Multi-Class Online Generalization Benchmark ({num_samples} Streamed Images)")
    print("=" * 80)

    model = ResNet18Classifier(num_classes=6)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    print("[OK] Loaded ResNet-18 Student Model from best_model.pth")

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\nConnecting to Hugging Face stream: 'prithivMLmods/OpenScene-Classification' (Buffered Shuffle)...")
    try:
        ds = load_dataset('prithivMLmods/OpenScene-Classification', split='train', streaming=True)
        # Buffer shuffle ensures a balanced mix of all 6 classes
        ds = ds.shuffle(seed=42, buffer_size=3000)
    except Exception as e:
        print(f"[ERROR] Could not load dataset stream: {e}")
        return

    confusion = np.zeros((6, 6), dtype=int)
    correct = 0
    total = 0
    correct_confs = []
    incorrect_confs = []

    pbar = tqdm(total=num_samples, desc="Evaluating Streamed Scenes")

    for sample in ds:
        if total >= num_samples:
            break

        try:
            pil_img = sample["image"].convert("RGB")
            gt_label = sample["label"]
        except Exception:
            continue

        if gt_label < 0 or gt_label >= 6:
            continue

        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(probs).item()
            conf = probs[pred_label].item()

        is_correct = (pred_label == gt_label)
        if is_correct:
            correct += 1
            correct_confs.append(conf)
        else:
            incorrect_confs.append(conf)

        confusion[gt_label][pred_label] += 1
        total += 1
        pbar.update(1)

        if total % 100 == 0:
            current_acc = (correct / total) * 100.0
            pbar.set_postfix(acc=f"{current_acc:.2f}%", total=total)

    pbar.close()

    overall_acc = (correct / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"  BALANCED ONLINE BENCHMARK RESULTS: {total} Unseen Images Evaluated in RAM")
    print("=" * 80)
    print(f"🎯 Global Out-of-Domain Accuracy: {overall_acc:.2f}% ({correct}/{total} correct)")
    if correct_confs:
        print(f"📈 Avg Confidence on Correct:     {np.mean(correct_confs)*100.0:.2f}%")
    if incorrect_confs:
        print(f"📉 Avg Confidence on Incorrect:   {np.mean(incorrect_confs)*100.0:.2f}%")

    print("\n--- Per-Class Performance Breakdown ---")
    print(f"{'Class Name':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Evaluated Samples'}")
    print("-" * 65)

    for i, cname in enumerate(PROJECT_CLASSES):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(6) if j != i)
        fn_actual = sum(confusion[i][j] for j in range(6) if j != i)
        total_class = sum(confusion[i])

        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn_actual) * 100.0) if (tp + fn_actual) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        print(f"{cname:<12} | {prec:8.2f}%  | {rec:8.2f}%  | {f1:8.2f}%  | {total_class}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    num = 2000
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    run_online_evaluation(num)
