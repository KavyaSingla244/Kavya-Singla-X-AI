# 🏔️ Intel Scene Classification — 3LC Data-Centric AI Challenge (Kaggle)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![3LC](https://img.shields.io/badge/Powered%20By-3LC%20AI-green.svg)](https://3lc.ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-Top%205%20Leaderboard-20beff.svg)](https://www.kaggle.com)

A high-performance solution for the **3LC Data-Centric AI Challenge on Kaggle: 6-Class Natural Scene Classification** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).

---

## 🎯 Challenge Overview & Strict Constraints

- **Fixed Architecture**: Standard `ResNet-18` classifier (`ResNet18Classifier`).
- **No Pretrained Weights**: Must train strictly **from scratch** (`weights=None`).
- **Strict Labeling Budget**: Training dataset is strictly capped at **at most 3,000 samples with weight = 1.0** (600 ground-truth seed + 2,400 curated from 6,000 unlabeled pool images).
- **Core Objective**: Optimize data curation, sample purity, and training dynamics to achieve maximum test set accuracy.

---

## 📈 Benchmark Progression Across Iteration Rounds

| Iteration Round | Strategy & Methodology | Peak Validation Acc | Public Leaderboard |
| :--- | :--- | :---: | :---: |
| **Round 1** | Seed dataset only (600 images) + AdamW baseline | 68.25% | — |
| **Round 2** | Raw pseudo-labeling (3,000 images) + AdamW | 75.33% | 0.7488 |
| **Round 3** | Curated 3,000 images + SGD Nesterov + MixUp ($\alpha=0.3$) | 80.25% | 0.7944 |
| **Round 4** | CutMix ($\alpha=1.0$) + MixUp + Dynamic Model EMA (decay = 0.999) | 81.25% | 0.7955 |
| **Round 5 (Current)** | **OpenCLIP ViT-B/32 Zero-Shot Consensus Curation + Warm-Start Fine-Tuning + RandAugment** | **83.08%** 🌟 | **~0.83–0.85+ (Pending)** |

---

## 🧠 Cutting-Edge Engineering Architecture

```
                                  [6,000 Unlabeled Pool Images]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [OpenCLIP ViT-B/32 Zero-Shot]                               [ResNet-18 Snapshot Ensemble]
           (86.92% Zero-Shot)                                          (11 Checkpoints)
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
                     [Warm-Start Fine-Tuning from 81.25% Baseline]
                      ├── RandAugment (num_ops=2, magnitude=7)
                      ├── ColorJitter + RandomCrop + Flip
                      ├── CutMix (alpha=1.0) + MixUp (alpha=0.3)
                      ├── Nesterov SGD (Cosine Annealing)
                      └── Dynamic Model EMA (decay = 0.999)
                                                │
                                                ▼
                              [83.08% Peak Checkpoint & Snapshots]
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
├── curate_foundation_consensus.py  # OpenCLIP + ResNet-18 dual consensus curation
├── train_warmstart.py              # Round 5 warm-start fine-tuning pipeline
├── train.py                        # Baseline training with CutMix/MixUp & Dynamic EMA
├── predict.py                      # Single best model test inference with TTA
├── predict_ensemble.py             # Multi-model snapshot blend ensemble with TTA
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

### 3. Run Warm-Start Training
```bash
python train_warmstart.py
```

### 4. Generate Ensemble Predictions
```bash
python predict_ensemble.py
```
Outputs `submission.csv` aligned with Kaggle requirements.

---

## 📜 License
MIT License
