# Multimodal Image–Text Retrieval with CLIP-Style Fine-Tuning

**A PyTorch project that fine-tunes a CLIP-style model on MS COCO 2014 for bidirectional image–text retrieval, with a full ablation study and interactive search.**

---

## Highlights

- **Contrastive learning**: Train an image encoder (ResNet50 + projection head) to align with a frozen CLIP text encoder using InfoNCE loss in a shared 512-d embedding space.
- **Production-minded data pipeline**: Cached CLIP text embeddings for faster training; COCO 2014 image–caption pairs with configurable preprocessing and verification.
- **Systematic ablation study**: Compare baseline vs. data augmentation, layer/batch norm, dropout, LR warmup, and combined configs—with automated training and evaluation.
- **Rigorous evaluation**: Recall@1/5/10 and mean rank for both image→text and text→image retrieval; chunked similarity computation for large validation sets.
- **Interactive demo**: CLI for text-to-image search over the validation set with optional visualization and save.

---

## Overview

This project implements **multimodal retrieval**: given a text query, retrieve the most relevant images (and vice versa) by learning a joint embedding space. The approach follows the CLIP paradigm: a **trainable image encoder** (ResNet50 backbone + projection head) is aligned via contrastive loss to a **frozen CLIP text encoder** so that matching image–caption pairs sit close in the 512-d space.

**Why it matters:** Cross-modal retrieval powers search, recommendation, and accessibility. This repo shows end-to-end ML engineering: dataset preparation, model design, training loops, evaluation metrics, ablation experiments, and an interactive application—all in one codebase.

---

## Key Features

| Feature                 | Description                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset preparation** | Load COCO 2014 images and captions; encode captions with CLIP; cache embeddings (`.pt`) and metadata for fast training.                                         |
| **Baseline training**   | ResNet50 (ImageNet) + 2-layer projection (Linear→GELU→Linear)→512-d; InfoNCE loss; AdamW + cosine LR schedule.                                                  |
| **Improved training**   | Optional data augmentation (crop, flip, color jitter), LayerNorm/BatchNorm, dropout, LR warmup, gradient clipping; configurable via `--config`.                 |
| **Evaluation**          | Recall@1/5/10 and mean rank for image→text and text→image; similarity computed in chunks to handle large sets.                                                  |
| **Visualizations**      | Training/validation loss curves; text-query retrieval grids; image classification scores vs. class labels.                                                      |
| **Ablation study**      | Train multiple configurations (e.g. `augmentation`, `layer_norm`, `dropout`, `warmup`, `all`) and evaluate them in one run; results and comparison plots saved. |
| **Interactive search**  | `text_to_image_search.py`: type text queries and get top-K images with similarity scores and optional saved figures.                                            |

---

## Demo / Screenshots

### Training: best validation loss across ablation configs

Comparison of best validation loss for baseline and improved configurations (augmentation, layer norm, dropout, warmup, and combined).

![Best validation loss comparison across ablation configs](https://raw.githubusercontent.com/WallaceSuu/multimodal-retrieval/main/loss_curves/best_val_loss_comparison.png "Best Val Loss — Ablation Comparison")

### Evaluation: image classification

The model scores an input image against candidate class labels (e.g. "a person", "an animal", "food") in the shared embedding space.

![Image classification — class scores](https://raw.githubusercontent.com/WallaceSuu/multimodal-retrieval/main/evaluation_results/classification_3690.png "Image classification scores")

![Image classification — another example](https://raw.githubusercontent.com/WallaceSuu/multimodal-retrieval/main/evaluation_results/classification_3620.png "Image classification example")

### Text-to-image retrieval

Top retrieved images for natural-language queries (e.g. "sport", "animal") from the validation set.

![Text-to-image retrieval: sport](https://raw.githubusercontent.com/WallaceSuu/multimodal-retrieval/main/evaluation_results/text_to_image_sport.png "Text-to-Image: 'sport'")

![Text-to-image retrieval: animal](https://raw.githubusercontent.com/WallaceSuu/multimodal-retrieval/main/evaluation_results/text_to_image_animal.png "Text-to-Image: 'animal'")

*All images linked from the repo; replace paths if your run outputs differ. Use [blob URLs](https://github.com/WallaceSuu/multimodal-retrieval/blob/main/evaluation_results/) to open files on GitHub.*

---

## Tech Stack

| Category      | Technologies                                                  |
| ------------- | ------------------------------------------------------------- |
| **Framework** | PyTorch 2.x, torchvision                                      |
| **Models**    | ResNet50 (ImageNet), HuggingFace Transformers (CLIP ViT-B/32) |
| **Data**      | MS COCO 2014, PIL, CLIP normalization (224×224)               |
| **Training**  | InfoNCE loss, AdamW, optional cosine annealing / warmup       |
| **Utilities** | tqdm, matplotlib, numpy, safetensors                          |

---

## Architecture / How It Works

1. **Data**
   - COCO 2014 images + caption JSON → `dataset_preparation.py` builds image–caption pairs, encodes captions with pretrained CLIP, and saves text embeddings and metadata under `cache/`.

2. **Training**
   - **Image branch**: Image → ResNet50 → 2048-d → projection head → 512-d, L2-normalized.
   - **Text branch**: Captions are already embedded (cached); no text encoder training.
   - **Loss**: InfoNCE over batch: maximize similarity of (image, caption) pairs, minimize with in-batch negatives; temperature 0.07.

3. **Evaluation**
   - Encode all validation images and use cached text embeddings → full similarity matrix (in chunks) → compute Recall@K and mean rank for both directions.

4. **Search**
   - Encode a text query with CLIP → cosine similarity with all image embeddings → return top-K images.

---

## Repository Structure

```
multimodal-retrieval/
├── dataset_preparation.py   # COCO load, CLIP encode, cache embeddings + metadata
├── train_clip.py            # Baseline: ResNet50 + projection, InfoNCE, single config
├── train_clip_improved.py   # Extended: augmentation, norm, dropout, warmup, ablation
├── evaluate_clip.py         # Recall@K, mean rank, retrieval/classification visuals
├── text_to_image_search.py  # Interactive text-query search over validation set
├── ablation_study.py        # Evaluate multiple checkpoints, plot comparison
├── requirements.txt         # PyTorch, torchvision, transformers, etc.
├── Train.txt                # Training command reference
├── Test.txt                 # Evaluation/search command reference
├── cache/                   # Cached train/val embeddings and metadata (created by dataset_preparation)
├── checkpoints_*/           # Per-config checkpoints and training_history.json (gitignored)
├── evaluation_results*/    # metrics.json and visualizations per run
└── ablation_results.json    # Training summary from ablation runs (if generated)
```

---

## Setup / Installation

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/YOUR_USERNAME/multimodal-retrieval.git
   cd multimodal-retrieval
   pip install -r requirements.txt
   ```

2. **Obtain MS COCO 2014**
   - Images: [COCO 2014](https://cocodataset.org/#download) (train/val).
   - Annotations: `captions_train2014.json`, `captions_val2014.json` from the same site (or e.g. [Kaggle COCO 2014](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3)).
   - Place files so that:
     - Images: `coco2014/images/train2014/`, `coco2014/images/val2014/`
     - Annotations: `coco2014/annotations/captions_train2014.json`, `coco2014/annotations/captions_val2014.json`

   The repo does not include a download script; `dataset_preparation.py` prints instructions if annotation files are missing.

---

## How to Run

**1. Prepare data (one-time)**

```bash
python dataset_preparation.py
```

This creates `cache/train_embeddings.pt`, `cache/train_metadata.pt`, and the same for `val`. Default root is `coco2014`; you can change paths inside the script if needed.

**2. Train**

- Baseline (single config):

  ```bash
  python train_clip.py
  ```

- Configurable + ablation (recommended):

  ```bash
  python train_clip_improved.py --config baseline
  python train_clip_improved.py --config augmentation
  python train_clip_improved.py --config all
  # Full ablation (trains several configs):
  python train_clip_improved.py --config ablation
  ```

  Checkpoints and `training_history.json` go to `checkpoints_<config>/`.

**3. Evaluate**

```bash
python evaluate_clip.py --checkpoint checkpoints_baseline/best_model.pt
python evaluate_clip.py --checkpoint checkpoints_baseline/best_model.pt --subset_size 5000 --output_dir evaluation_results_baseline
```

**4. Compare ablations**

```bash
python ablation_study.py --subset_size 5000
```

**5. Interactive text-to-image search**

```bash
python text_to_image_search.py --checkpoint checkpoints_baseline/best_model.pt --top_k 5
```

---

## Environment Variables

None required. All paths and options are set via script defaults or CLI flags (`--data_root`, `--cache_dir`, `--checkpoint`, etc.). Use a GPU if available; the code falls back to CPU.

---

## Example Workflow / Usage

```bash
# One-time data prep
python dataset_preparation.py

# Train baseline and one improved config
python train_clip_improved.py --config baseline
python train_clip_improved.py --config augmentation

# Evaluate and save metrics
python evaluate_clip.py --checkpoint checkpoints_baseline/best_model.pt --output_dir evaluation_results_baseline
python evaluate_clip.py --checkpoint checkpoints_augmentation/best_model.pt --output_dir evaluation_results_augmentation

# Compare all trained configs (if you ran ablation or multiple configs)
python ablation_study.py --subset_size 5000

# Try interactive search
python text_to_image_search.py --checkpoint checkpoints_baseline/best_model.pt
# Then e.g. type: "person playing sport" and "dog on grass"
```

---

## Technical Highlights

- **Efficient training**: Cached text embeddings avoid recomputing CLIP text features every epoch; only the image encoder and projection head are trained.
- **Stable training**: Gradient clipping (in improved trainer), optional warmup, and AdamW with weight decay.
- **Scalable evaluation**: Similarity matrix computed in configurable chunks to control memory on large validation sets.
- **Reproducibility**: Configs and training history saved per run (`config.json`, `training_history.json`); fixed seeds can be added in scripts for full reproducibility.
- **Modular design**: Shared `CLIPImageEncoder`, `COCODataset`, and `info_nce_loss` across training and evaluation; improved encoder and trainer extend the baseline without duplicating core logic.

---

## Challenges and Engineering Decisions

| Challenge                                        | Decision                                                                                                                            |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Matching image encoder to frozen CLIP text space | Use a projection head (Linear→GELU→Linear) from ResNet50’s 2048-d to 512-d with L2 normalization, and train with symmetric InfoNCE. |
| Training speed                                   | Precompute and cache CLIP text embeddings once in `dataset_preparation.py`; training only loads cached tensors.                     |
| Generalization vs. overfitting                   | Added optional augmentation, dropout, LayerNorm, and LR warmup in `train_clip_improved.py`; ablation study to compare.              |
| Large-N evaluation                               | Compute similarity in chunks (e.g. 1000×1000) in `evaluate_clip.py` to avoid O(N²) memory.                                          |
| Multiple experiments                             | Named configs (`--config`) and separate `checkpoints_*` directories so multiple runs coexist; ablation script evaluates all.        |

---

## Results / What I Learned

- **Metrics (example)**: On the validation set, a baseline run can reach roughly **Recall@1 ~17% (image→text) and ~22% (text→image)** and **Recall@10 ~54% / ~62%**, with mean rank in the low 30s. Exact numbers depend on data subset and training length (see `evaluation_results/metrics.json` if you run locally).
- **Ablation**: The codebase supports comparing loss and Recall across configs (e.g. `ablation_results.json` and `ablation_study.py` plots); the best combo is dataset-dependent and worth reporting in your README or docs once you have runs.
- **Takeaways**: Contrastive training with a frozen text encoder is sufficient for strong retrieval; caching and chunked eval make iteration and evaluation practical; ablation structure makes it easy to justify design choices in interviews.

---
