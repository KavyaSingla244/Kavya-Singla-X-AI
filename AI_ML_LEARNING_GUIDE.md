# 📚 Comprehensive AI / ML Learning & Reference Guide
> **Curated from the Intel Scene Classification 3LC Data-Centric AI Project**  
> A structured glossary of all concepts, neural networks, optimizers, loss functions, augmentations, and libraries used throughout the competition to guide your future Machine Learning learning journey.

---

## 📑 Table of Contents
1. [Core AI & Machine Learning Paradigms](#1-core-ai--machine-learning-paradigms)
2. [Neural Network Architectures & Models](#2-neural-network-architectures--models)
3. [Optimizers & Optimization Algorithms](#3-optimizers--optimization-algorithms)
4. [Loss Functions & Mathematical Objectives](#4-loss-functions--mathematical-objectives)
5. [Regularization & Training Stabilization Techniques](#5-regularization--training-stabilization-techniques)
6. [Data Augmentation & Computer Vision Techniques](#6-data-augmentation--computer-vision-techniques)
7. [Evaluation Metrics & Diagnostics](#7-evaluation-metrics--diagnostics)
8. [Software, Frameworks & Libraries](#8-software-frameworks--libraries)
9. [Recommended Learning Path](#9-recommended-learning-path)

---

## 1. Core AI & Machine Learning Paradigms

### 🔹 Data-Centric AI vs. Model-Centric AI
- **Model-Centric AI**: Keeping the dataset fixed and changing neural network architectures, hyperparameter tuning, or adding layers to improve accuracy.
- **Data-Centric AI**: Keeping the model architecture fixed (e.g., standard ResNet-18) and focusing on engineering, curating, cleaning, noise-filtering, and weighting the training dataset to improve performance.

### 🔹 Supervised Learning
- Training a neural network using labeled input-output pairs $(x, y)$, where $x$ is the input image and $y$ is the ground-truth class label (e.g., *Forest*).

### 🔹 Self-Supervised Learning (SSL)
- A training technique where models learn rich representations directly from raw unlabeled data without human annotations (e.g., predicting missing patches or predicting invariant views in **DINOv2**).

### 🔹 Contrastive Multimodal Learning
- Training models on image-text pairs by pulling matching image-text embeddings closer together in latent space and pushing non-matching pairs apart (the core mechanism behind **CLIP / OpenCLIP**).

### 🔹 Knowledge Distillation (Teacher-Student Framework)
- An information-theoretic method where a large, capable model (the **Teacher**, like DINOv2 + OpenCLIP) transfers its learned knowledge to a smaller or constrained model (the **Student**, our ResNet-18).

### 🔹 Dark Knowledge & Soft Targets
- **Hard Labels**: A binary one-hot vector (e.g., $[0, 0, 1, 0, 0, 0]$ for Glacier).
- **Soft Targets (Dark Knowledge)**: Continuous probabilities output by a teacher model (e.g., $[0.01, 0.02, 0.82, 0.14, 0.01, 0.00]$). This reveals that this glacier photo shares 14% visual resemblance to a mountain, teaching the student nuanced inter-class relationships.

### 🔹 Out-of-Distribution (OOD) Generalization
- Testing how well a trained model performs on completely unseen images collected from different cameras, environments, or lighting conditions (e.g., internet photos from Unsplash/Hugging Face).

### 🔹 Data Leakage
- When information from the validation or test dataset accidentally leaks into the training pipeline, artificially inflating performance without true learning.

---

## 2. Neural Network Architectures & Models

### 🔹 Convolutional Neural Networks (CNNs)
- Neural networks designed for image processing using spatial filters (kernels) that slide over image pixels to detect edges, textures, patterns, and complex objects.

### 🔹 ResNet-18 (Residual Network with 18 Layers)
- A classic, robust CNN architecture featuring **Skip Connections (Residual Blocks)** that allow gradients to flow directly through identity shortcuts ($F(x) + x$), preventing the vanishing gradient problem.

### 🔹 Vision Transformers (ViTs)
- Attention-based neural networks that treat an image as a sequence of patches (similar to words in a sentence) and compute self-attention across all patches to capture global image context.

### 🔹 DINOv2 (Self-Supervised ViT by Meta AI)
- A state-of-the-art self-supervised Vision Transformer foundation model trained on 142M images. It excels at extracting fine-grained spatial geometry, surface depth, and patch textures without text bias.

### 🔹 OpenCLIP (Open-Source Contrastive Language-Image Pretraining)
- Multimodal neural networks (e.g., `ViT-B/16` and `ViT-B/32`) trained on LAION-2B image-text pairs that map visual scenes to rich semantic concepts.

### 🔹 Linear Probe / Feature Extractor
- Freezing the backbone of a pretrained model and training only a single linear classification layer on top to evaluate the quality of the learned representations.

---

## 3. Optimizers & Optimization Algorithms

### 🔹 Stochastic Gradient Descent (SGD) with Momentum
- An optimizer that updates weights in the direction of the loss gradient, using a velocity vector (momentum $\beta = 0.9$) to accelerate through flat regions and dampen oscillations.

### 🔹 Nesterov Accelerated Gradient (NAG)
- An advanced variation of momentum that calculates the gradient not at the current position, but slightly ahead in the direction of momentum, leading to faster convergence and better stability.

### 🔹 AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)
- A popular adaptive optimizer that maintains individual learning rates for every parameter based on first and second gradient moments, while cleanly decoupling L2 weight regularization.

### 🔹 Learning Rate Scheduling: Cosine Annealing with Warmup
- **Linear Warmup**: Gradually increasing the learning rate from 0 during early epochs to prevent violent initial updates.
- **Cosine Annealing**: Smoothly decaying the learning rate following a cosine curve down to a minimum value, allowing the model to settle into the deepest region of a loss valley.

### 🔹 Sharp vs. Flat Loss Minima
- **Sharp Minima**: Narrow crevices on the loss surface. High training accuracy, but small shifts in test data cause large test errors.
- **Flat Minima**: Wide, gentle valleys. Models in flat minima generalize significantly better to new unseen images (favored by SGD + Nesterov).

---

## 4. Loss Functions & Mathematical Objectives

### 🔹 Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$)
- The standard classification loss measuring the distance between predicted probability distribution $\hat{y}$ and true one-hot label $y$:
  $$\mathcal{L}_{\text{CE}} = -\sum_{i} y_i \log(\hat{y}_i)$$

### 🔹 Kullback-Leibler (KL) Divergence ($\mathcal{L}_{\text{KL}}$)
- An information-theoretic loss measuring the relative entropy between the student model's predicted distribution $P_s$ and the teacher model's soft target distribution $P_t$:
  $$\mathcal{L}_{\text{KL}}(P_t \parallel P_s) = \sum_{i} P_t(i) \log\left(\frac{P_t(i)}{P_s(i)}\right)$$

### 🔹 Temperature Scaling ($T$)
- A hyperparameter applied to logits before softmax ($\frac{z}{T}$). Higher temperatures ($T=3.0$) smooth out extreme confidence peaks, exposing the "dark knowledge" in low-probability classes.

### 🔹 Softmax Function & Confidence Calibration
- Converts raw unnormalized model outputs (logits) into a valid probability distribution summing to 1.0:
  $$\sigma(z)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

---

## 5. Regularization & Training Stabilization Techniques

### 🔹 Model EMA (Exponential Moving Average)
- Maintaining a "shadow model" whose weights update slowly as a weighted average over training steps:
  $$\theta_{\text{EMA}} \leftarrow \beta \cdot \theta_{\text{EMA}} + (1 - \beta) \cdot \theta_{\text{model}}$$
- Smooths out batch-to-batch gradient jitter, producing our peak **84.92%** single checkpoint.

### 🔹 Stochastic Weight Averaging (SWA)
- Averaging the model weights across multiple checkpoints during late training epochs to find the geometric centroid of flat loss basins.

### 🔹 Label Smoothing ($\epsilon = 0.1$)
- Replacing hard binary targets ($1.0$) with softened targets ($1 - \epsilon + \frac{\epsilon}{K}$) to prevent overconfidence.

### 🔹 Weight Decay / L2 Regularization
- Penalizing large weights in the loss function ($\frac{1}{2} \lambda \|w\|^2$) to prevent individual neurons from memorizing noise.

### 🔹 Dropout
- Randomly zeroing out a percentage of neurons (e.g., 30%) during forward passes in training, forcing the network to learn redundant, co-adapted features.

### 🔹 Batch Normalization (BatchNorm)
- Normalizing layer inputs across mini-batches to maintain zero mean and unit variance, speeding up convergence and stabilizing internal covariate shift.

---

## 6. Data Augmentation & Computer Vision Techniques

### 🔹 CutMix Augmentation
- Cutting a random rectangular patch from image $B$ and pasting it onto image $A$. In our pipeline, we linearly interpolated the teacher continuous probability vectors according to the bounding box area ratio $\lambda$.

### 🔹 MixUp Augmentation
- Linearly overlaying two images: $x = \lambda x_A + (1 - \lambda) x_B$ and blending their labels proportionally.

### 🔹 Receptive Field & Kernel Strides
- The specific area in the input image that a neuron in a neural network layer "sees". ResNet-18’s stem $7\times7$ kernel is mathematically matched to $224\times224$ pixels.

### 🔹 Bicubic Anti-Aliasing Interpolation
- High-quality image resizing that samples a $4\times4$ grid of neighboring pixels using cubic polynomial interpolation, preserving high-frequency textures (ice, rocks, leaves).

### 🔹 Test-Time Augmentation (TTA)
- During inference, feeding multiple augmented views (e.g., original + horizontal flip + multi-crop) of the test image through the model and averaging predictions to reduce single-view variance.

### 🔹 Multi-Model Ensembling
- Combining probability predictions from multiple independently trained snapshot checkpoints to produce a single superior prediction (`submission.csv`).

---

## 7. Evaluation Metrics & Diagnostics

### 🔹 Precision, Recall & F1-Score
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (When the model predicts *Glacier*, how often is it right?).
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Out of all real *Glacier* images, how many did it find?).
- **F1-Score**: Harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

### 🔹 Confusion Matrix
- A $6\times6$ table showing actual vs. predicted classes, identifying specific blind spots (e.g., Mountain misclassified as Glacier).

### 🔹 UMAP / t-SNE Dimensionality Reduction
- Mathematical algorithms that project high-dimensional latent feature vectors (e.g., 512-D) down to 2D coordinates for visual cluster inspection in the 3LC dashboard.

---

## 8. Software, Frameworks & Libraries

| Tool | Category | What it does |
| :--- | :--- | :--- |
| **3LC (`tlc`)** | Data-Centric AI Platform | Manages dataset tables, versioning, sample weighting (`weight=1.0` vs `0.0`), and interactive dashboard visualizations. |
| **PyTorch (`torch`)** | Deep Learning Framework | Core computational graph engine, automatic differentiation (`autograd`), and GPU/CPU tensor execution. |
| **Torchvision** | Vision Toolkit | Pre-built datasets, image transforms (RandomResizedCrop, ColorJitter), and standard CNN architectures. |
| **TIMM (`timm`)** | PyTorch Image Models | Open-source library containing cutting-edge vision architectures (used for loading **DINOv2**). |
| **OpenCLIP (`open_clip_torch`)** | Multimodal Foundation Library | Pretrained CLIP models for zero-shot text-image semantic embedding. |
| **Hugging Face Datasets (`datasets`)** | Remote Dataset Streaming | Zero-disk streaming of parquet datasets directly over HTTP into memory. |
| **Scikit-Learn (`sklearn`)** | Classical ML Toolkit | Computing classification reports, confusion matrices, and metrics. |
| **NumPy & Pillow (PIL)** | Numerical & Imaging Core | Array manipulations, `.npy` probability serialization, and image decoding. |

---

## 9. Recommended Learning Path for Future Mastery

1. **Foundations**:
   - Python for Scientific Computing (`numpy`, `pandas`, `matplotlib`).
   - Linear Algebra (Vectors, Matrices, Dot Products, Eigenvalues).
   - Calculus (Derivatives, Gradients, Chain Rule).
2. **Deep Learning Core**:
   - Deep Learning with PyTorch (Tensors, Autograd, `nn.Module`, `DataLoader`).
   - CNN Fundamentals (Convolutions, Pooling, Padding, Strides, Receptive Fields).
3. **Advanced Computer Vision**:
   - ResNets, DenseNets, ConvNeXt, and Vision Transformers (ViT).
   - Self-Supervised Learning (SimCLR, MoCo, DINOv2).
   - Multimodal Models (CLIP, OpenCLIP, SigLIP).
4. **Data-Centric AI & MLOps**:
   - Active Learning, Label Noise Curation, and Data Cleaning (using **3LC**).
   - Knowledge Distillation & Model Compression.
   - Experiment Tracking & Ensembling.

---
*Created for Kavya Singla — 3LC Data-Centric AI Competition Reference Guide.*
