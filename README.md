# 🏔️ Intel Scene Classification — 3LC Data-Centric AI Challenge (Kaggle)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![3LC](https://img.shields.io/badge/Powered%20By-3LC%20AI-green.svg)](https://3lc.ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-Top%20Leaderboard%20Track-20beff.svg)](https://www.kaggle.com)

A high-performance solution for the **3LC Data-Centric AI Challenge on Kaggle: 6-Class Natural Scene Classification** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

---

## 🎯 Challenge Overview & Strict Constraints

- **Fixed Architecture**: Standard `ResNet-18` classifier (`ResNet18Classifier`).
- **No Pretrained Weights**: Must train strictly **from scratch** (`weights=None`).
- **Strict Labeling Budget**: Training dataset is strictly capped at **at most 3,000 samples with weight = 1.0** (600 ground-truth seed + 2,400 curated from 6,000 unlabeled pool images).
- **Core Objective**: Maximize model accuracy and real-world out-of-domain generalization through pure data-centric engineering, noise filtering, and knowledge distillation.

---

## 📈 Benchmark Progression Across Iteration Rounds

| Iteration Round | Strategy & Methodology | Peak Validation Acc | Real-World Generalization |
| :--- | :--- | :---: | :---: |
| **Round 1** | Seed dataset only (600 images) + AdamW baseline | 68.25% | — |
| **Round 2** | Raw pseudo-labeling (3,000 images) + AdamW | 75.33% | — |
| **Round 3** | Curated 3,000 images + SGD Nesterov + MixUp ($\alpha=0.3$) | 80.25% | — |
| **Round 4** | CutMix ($\alpha=1.0$) + MixUp + Dynamic Model EMA (decay = 0.999) | 81.25% | LB: **0.79555** |
| **Round 5** | OpenCLIP Foundation Consensus Curation (3,000-table) | 83.08% | — |
| **Round 6 (Current)** | **Multi-Teacher (ViT-B/16 + ViT-B/32) Soft Knowledge Distillation** | **83.92% 🏆** | **86.42% (1,200 Unseen Online Scenes)** |

---

## 🧠 Cutting-Edge Engineering Architecture

```
                                  [6,000 Unlabeled Pool Images]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [OpenCLIP ViT-B/32 Zero-Shot]                               [OpenCLIP ViT-B/16 Zero-Shot]
           (86.92% Zero-Shot)                                          (88.45% Zero-Shot)
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                                [Consensus Agreement Filter]
                                (Mutual Agreement: 77.1%)
                                                │
                                                ▼
                         [Top 400 Purest Candidates Per Class (2,400)]
                                                │
                                                ▼
                         [3LC Table Revision: 600 Seed + 2,400 Pure]
                                                │
                                                ▼
                    [Multi-Teacher Soft Knowledge Distillation Pipeline]
                     ├── OpenCLIP Dark Knowledge Soft Probability Vectors
                     ├── Native Soft-Target Cross-Entropy Loss
                     ├── RandAugment (num_ops=2, magnitude=7) + ColorJitter
                     ├── Nesterov SGD (Cosine Annealing)
                     └── Dynamic Model EMA (decay = 0.999)
                                                │
                                                ▼
                              [83.92% Peak Checkpoint & Snapshots]
                                                │
                                                ▼
                              [Multi-Model Snapshot Ensemble (TTA)]
                                                │
                                                ▼
                                        [submission.csv]
```

---

## 🛠️ Repository Structure

```
├── curate_foundation_consensus.py   # Multi-foundation model consensus curation
├── train_distill_fast.py           # High-speed RAM-cached knowledge distillation
├── train_distill.py                # Multi-teacher knowledge distillation pipeline
├── train_warmstart.py              # Warm-start fine-tuning pipeline
├── predict_ensemble.py             # Multi-model snapshot blend ensemble with TTA
├── test_external_generalization.py # Zero-disk in-RAM streaming generalization benchmark
├── sample_submission.csv           # Kaggle official submission format
├── submissions/                    # Versioned and timestamped Kaggle submissions
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Foundation Consensus Curation
```bash
python curate_foundation_consensus.py
```

### 3. Run Knowledge Distillation Training
```bash
python train_distill_fast.py
```

### 4. Test Out-of-Domain Generalization (Zero Disk Streaming)
```bash
python test_external_generalization.py
```

### 5. Generate Submission
```bash
python predict_ensemble.py
```
Outputs `submission.csv` aligned with Kaggle requirements in `~/Downloads/`.

---

## 📜 License
MIT License
