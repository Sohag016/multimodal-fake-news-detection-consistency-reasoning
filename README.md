# 🔥 Multimodal Consistency Reasoning Framework for Fake News Detection

> **Research Paper Implementation** — A novel multimodal framework for fake news detection combining **RoBERTa + ResNet-50**, **CLIP-based text-image alignment**, **OCR consistency scoring**, and **learned MLP fusion** on the **Fakeddit** dataset (6-way classification).

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Core Novelty](#-core-novelty)
3. [Research Contributions](#-research-contributions)
4. [Dataset](#-dataset)
5. [Framework Architecture](#-framework-architecture)
6. [Project Steps](#-project-steps)
   - [Step 1 — Environment Setup](#step-1--environment-setup)
   - [Step 2 — Install Libraries](#step-2--install-libraries)
   - [Step 3 — Seeds & Project Folders](#step-3--seeds--project-folders)
   - [Step 4 — Download Fakeddit Dataset](#step-4--download-fakeddit-dataset)
   - [Step 5 — Load Train / Val / Test TSV Files](#step-5--load-train--val--test-tsv-files)
   - [Step 6 — Create Balanced Dataset (400 per class)](#step-6--create-balanced-dataset-400-per-class)
   - [Step 7 — Download & Cache Images](#step-7--download--cache-images)
   - [Step 8 — Train / Validation Split](#step-8--train--validation-split)
   - [Step 9 — Tokenizer & Image Transforms](#step-9--tokenizer--image-transforms)
   - [Step 10 — Multimodal Dataset Class](#step-10--multimodal-dataset-class)
   - [Step 11 — DataLoaders](#step-11--dataloaders)
   - [Step 12 — Multimodal Model (RoBERTa + ResNet-50)](#step-12--multimodal-model-roberta--resnet-50)
   - [Step 13 — Loss, Optimizer & LR Scheduler](#step-13--loss-optimizer--lr-scheduler)
   - [Step 14 — Training & Evaluation Functions](#step-14--training--evaluation-functions)
   - [Step 15 — Training Loop with Early Stopping](#step-15--training-loop-with-early-stopping)
   - [Step 16 — Load Best Model & Classification Report](#step-16--load-best-model--classification-report)
   - [Step 17 — Training Curves](#step-17--training-curves)
   - [Step 18 — CLIP-Based Text-Image Alignment Model](#step-18--clip-based-text-image-alignment-model)
   - [Step 19 — CLIP Dataset & DataLoaders](#step-19--clip-dataset--dataloaders)
   - [Step 20 — Train CLIP Model](#step-20--train-clip-model)
   - [Step 21 — OCR Text Extraction & Similarity](#step-21--ocr-text-extraction--similarity)
   - [Step 22 — Image Metadata Feature Extraction](#step-22--image-metadata-feature-extraction)
   - [Step 23 — XGBoost Metadata Model](#step-23--xgboost-metadata-model)
   - [Step 24 — Learned MLP Fusion](#step-24--learned-mlp-fusion)
   - [Step 25 — Full Model Comparison](#step-25--full-model-comparison)
   - [Step 26 — Consistency Score Computation](#step-26--consistency-score-computation)
   - [Step 27 — Full Prediction + Explanation + Verification Pipeline](#step-27--full-prediction--explanation--verification-pipeline)
   - [Step 28 — Consistency Analysis on Validation Set](#step-28--consistency-analysis-on-validation-set)
   - [Step 29 — Consistency Visualizations](#step-29--consistency-visualizations)
   - [Step 30 — Consistency Threshold Analysis](#step-30--consistency-threshold-analysis)
   - [Step 31 — Ablation Study](#step-31--ablation-study)
   - [Step 32 — Confusion Matrices](#step-32--confusion-matrices)
   - [Step 33 — Save All Results](#step-33--save-all-results)
   - [Step 34 — Final Summary & Publication Figures](#step-34--final-summary--publication-figures)
7. [Models & Components](#-models--components)
8. [Key Upgrades Over Baseline](#-key-upgrades-over-baseline)
9. [Publication Figures](#-publication-figures)
10. [Installation & Setup](#-installation--setup)
11. [How to Run](#-how-to-run)
12. [Results](#-results)
13. [Technologies Used](#-technologies-used)
14. [Citation](#-citation)

---

## 🔍 Project Overview

This is the official implementation of the **Multimodal Consistency Reasoning Framework** for fake news detection. The system classifies Reddit posts (text + image pairs) into **6 fake news categories** using the **Fakeddit** dataset.

The framework combines four complementary components:
1. **RoBERTa + ResNet-50** — Joint text-image encoder for semantic classification
2. **CLIP** — Zero-shot-style text-image alignment scoring
3. **OCR + Metadata** — Extracted visual text and image statistics fed into XGBoost
4. **MLP Fusion** — Learned late fusion of all model probability outputs

The key novelty is the **Consistency Score** — a metric that quantifies how well the text and image of a post agree with each other, used as an interpretable reasoning signal for fake news detection.

---

## 💡 Core Novelty

> **Consistency Score** — A post-level signal computed from the agreement between the text modality (RoBERTa predictions), image modality (ResNet predictions), and cross-modal alignment (CLIP similarity). Posts where text and image are inconsistent are flagged as likely fake.

This score enables:
- **Interpretable predictions** — explains *why* a post is flagged
- **Threshold-based filtering** — retains only high-confidence predictions
- **Ablation analysis** — measures the contribution of each modality

---

## 🏆 Research Contributions

1. Novel **Consistency Score** for multimodal fake news reasoning
2. **Learned MLP Fusion** replacing fixed-weight ensemble methods
3. **CLIP-based alignment** as an auxiliary fake news signal
4. **OCR + image metadata** integration via XGBoost for non-semantic cues
5. Full **ablation study** isolating each component's contribution
6. **6 publication-ready figures** generated automatically

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Dataset** | Fakeddit (Reddit multimodal fake news) |
| **Task** | 6-way fake news classification |
| **Label Column** | `6_way_label` |
| **Modalities** | Post title (text) + linked image |
| **Balanced Sample** | 400 samples per class (2,400 total) |
| **Split** | Stratified train / validation / test |

### 6-Way Label Classes
| Label | Category |
|---|---|
| 0 | True |
| 1 | Satire / Parody |
| 2 | Misleading Content |
| 3 | Imposter Content |
| 4 | False Connection |
| 5 | Manipulated Content |

> The dataset is downloaded directly from the **Fakeddit** public repository via the notebook. Images are downloaded and cached locally per post ID.

---

## 🔄 Framework Architecture

```
Reddit Post (Title + Image)
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
  TEXT                                   IMAGE
(Post Title)                          (Linked Image)
    │                                       │
    ▼                                       ▼
RoBERTa-base                          ResNet-50
(tokenized,                         (224×224, augmented:
 max_len=128)                        ColorJitter, Rotation,
    │                                ResizedCrop)
    └──────────┬────────────────────────────┘
               ▼
     Multimodal Fusion Head
     (concat → Dense → Dropout → 6-class softmax)
               │
               ▼
          Model 1 Predictions (P1)
               │
    ┌──────────┴─────────────────────────────┐
    ▼                                        ▼
CLIP Model                            OCR + Metadata
(ViT-B/32 or similar)                (Tesseract OCR text
Text-Image Alignment                  + image stats:
Score → 6-class head                  brightness, contrast,
    │                                  aspect ratio, entropy)
    ▼                                        │
Model 2 Predictions (P2)                     ▼
                                      XGBoost Classifier
                                    Model 3 Predictions (P3)
    └──────────┬─────────────────────────────┘
               ▼
        Learned MLP Fusion
        (P1 ⊕ P2 ⊕ P3 → 64-D → 6-class)
               │
               ▼
        Final Prediction
               │
               ▼
        Consistency Score
   (text-image agreement signal)
               │
          ┌────┴────┐
          ▼         ▼
     Explanation  Threshold
     + Reasoning  Filtering
```

---

## 📁 Project Steps

### Step 1 — Environment Setup
Detects GPU availability via `torch.cuda.is_available()`. Sets device to `cuda` or `cpu`. Prints environment info (PyTorch version, CUDA version, GPU name).

### Step 2 — Install Libraries
Installs all required packages: `transformers`, `timm`, `pytesseract`, `opencv-python`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `requests`, and `tensorflow-datasets` via pip.

### Step 3 — Seeds & Project Folders
Sets global random seeds (`random`, `numpy`, `torch`) to `42` for full reproducibility. Creates output directories for models, figures, and results.

### Step 4 — Download Fakeddit Dataset
Downloads the **Fakeddit** multimodal fake news dataset TSV files (train, validate, test) from the public GitHub repository. Fakeddit contains Reddit posts with their linked images and 6-way fake news labels.

### Step 5 — Load Train / Val / Test TSV Files
Reads the three split TSV files into pandas DataFrames. Inspects class distribution across the `6_way_label` column. Prints sample counts per class for each split.

### Step 6 — Create Balanced Dataset (400 per class)
Performs **stratified undersampling** to create a balanced working dataset with exactly **400 samples per class** (2,400 total). Stratification by `6_way_label` ensures equal class representation.

> ✅ **Upgrade:** 400 per class (was 100) — more data for better generalisation.

### Step 7 — Download & Cache Images
Downloads Reddit-linked images for all 2,400 posts using `requests`. Images are saved locally by post ID. Failed downloads are tracked and those rows are removed from the working dataset. Handles timeouts and connection errors gracefully.

### Step 8 — Train / Validation Split
Performs a **stratified train/validation split** (80/20) on the balanced working dataset using `sklearn.model_selection.train_test_split` with `stratify=6_way_label`.

### Step 9 — Tokenizer & Image Transforms
**Text:** Loads `roberta-base` tokenizer from HuggingFace Transformers. Tokenizes post titles with `max_length=128`, padding, and truncation.

**Images — Training transforms:**
- `RandomResizedCrop(224)`
- `RandomHorizontalFlip`
- `ColorJitter` (brightness, contrast, saturation, hue)
- `RandomRotation(15°)`
- `Normalize` (ImageNet mean/std)

**Images — Validation/Test transforms:** `Resize(256)` → `CenterCrop(224)` → `Normalize`

> ✅ **Upgrade:** RoBERTa-base replaces DistilBERT; stronger augmentation added.

### Step 10 — Multimodal Dataset Class
Implements a custom `torch.utils.data.Dataset` that:
- Loads the post title and tokenizes it via RoBERTa tokenizer
- Opens and transforms the cached image via PIL
- Returns a dictionary: `{input_ids, attention_mask, pixel_values, labels}`

### Step 11 — DataLoaders
Wraps the Dataset into `torch.utils.data.DataLoader` objects:
- **Train:** `shuffle=True`, `batch_size=32`, `num_workers=2`
- **Val/Test:** `shuffle=False`, `batch_size=64`

### Step 12 — Multimodal Model (RoBERTa + ResNet-50)
Builds the primary multimodal classifier:

```
Text Branch:    RoBERTa-base → [CLS] token → 768-D
Image Branch:   ResNet-50 (pretrained) → avg pool → 2048-D
Fusion:         concat(768 + 2048) = 2816-D
                → Linear(2816, 512) → ReLU → Dropout(0.3)
                → Linear(512, 6) → Softmax
```

Both backbone encoders have their weights fine-tuned during training.

### Step 13 — Loss, Optimizer & LR Scheduler
- **Loss:** Weighted `CrossEntropyLoss` to handle any residual class imbalance
- **Optimizer:** `AdamW` with **separate learning rates** — lower LR for pretrained backbones (`1e-5`), higher LR for fusion head (`1e-3`)
- **Scheduler:** **Cosine decay with linear warmup** over training steps

> ✅ **Upgrade:** Differential LR + cosine warmup scheduler (was flat Adam).

### Step 14 — Training & Evaluation Functions
Implements `train_epoch()` and `eval_epoch()` functions:
- `train_epoch`: forward pass → loss → backward → gradient clipping → optimizer step → scheduler step
- `eval_epoch`: no-grad forward pass → accuracy + macro F1 computation

### Step 15 — Training Loop with Early Stopping
Trains for up to N epochs with **EarlyStopping** on `val_accuracy` (patience configurable). Saves the best checkpoint (`best_model.pt`) based on validation F1. Prints per-epoch: train loss, train acc, val loss, val acc, val F1.

> ✅ **Upgrade:** EarlyStopping added to prevent overfitting.

### Step 16 — Load Best Model & Classification Report
Reloads the best saved checkpoint. Runs inference on the **test set**. Prints full `sklearn.metrics.classification_report` with per-class Precision, Recall, F1, and support.

### Step 17 — Training Curves
Plots **Figure 1** — Training and validation loss/accuracy curves across epochs for the RoBERTa + ResNet-50 model.

### Step 18 — CLIP-Based Text-Image Alignment Model
Loads a **CLIP** model (`openai/clip-vit-base-patch32` or equivalent). Builds a custom classification head on top of CLIP's joint text-image embedding:

```
CLIP(text, image) → cosine similarity embedding → 512-D
→ Linear(512, 6) → Softmax
```

CLIP provides a fundamentally different signal — it measures **semantic alignment** between the post title and its image.

### Step 19 — CLIP Dataset & DataLoaders
Implements a separate Dataset class for CLIP that applies CLIP's own preprocessing (image processor + tokenizer) rather than standard ImageNet transforms.

### Step 20 — Train CLIP Model
Trains the CLIP classification head (backbone frozen or fine-tuned) using the same training loop structure as Step 14–15. Saves best CLIP checkpoint.

### Step 21 — OCR Text Extraction & Similarity
Uses **Tesseract OCR** (`pytesseract`) to extract any text embedded within the post images. Computes **cosine similarity** between:
- Extracted OCR text (TF-IDF vectorized)
- Post title text

This catches cases where image text contradicts or copies the headline — a strong fake news signal.

### Step 22 — Image Metadata Feature Extraction
Extracts hand-crafted visual features from each image:
- **Brightness** — mean pixel intensity
- **Contrast** — standard deviation of pixel intensity
- **Aspect ratio** — width / height
- **Entropy** — Shannon entropy of pixel distribution
- **Color channel statistics** — per-channel mean and std (RGB)

These metadata features capture compression artifacts, meme-style formatting, and other low-level cues associated with fake imagery.

### Step 23 — XGBoost Metadata Model
Trains an **XGBoost classifier** on the image metadata features (Step 22) combined with OCR similarity score (Step 21). Produces probability outputs for all 6 classes.

> ✅ **Upgrade:** XGBoost replaces the baseline RandomForest classifier.

### Step 24 — Learned MLP Fusion
Reloads best checkpoints for RoBERTa+ResNet and CLIP models. Extracts their **probability vectors** on the validation set. Builds a **learned MLP fusion** model:

```
Input: P1 (6-D, RoBERTa+ResNet) ⊕ P2 (6-D, CLIP) = 12-D
→ Linear(12, 64) → ReLU → Dropout(0.3)
→ Linear(64, 6) → Softmax
```

Trained end-to-end on the validation set predictions.

> ✅ **Upgrade:** Learned MLP replaces fixed-weight alpha blending.

### Step 25 — Full Model Comparison
Compiles accuracy and macro F1 scores for all models:
- RoBERTa + ResNet-50
- CLIP Alignment Model
- XGBoost Metadata
- Learned MLP Fusion

Prints a comparison table and generates **Figure 2** — grouped bar chart.

### Step 26 — Consistency Score Computation
**Core novelty of the framework.** Computes a **Consistency Score** for each post defined as:

```
Consistency = f(
    agreement(RoBERTa_pred, ResNet_pred),
    CLIP_text_image_similarity,
    OCR_title_similarity
)
```

A low score indicates text-image inconsistency — a strong indicator of manipulated or misleading content.

### Step 27 — Full Prediction + Explanation + Verification Pipeline
Implements an end-to-end pipeline that for any input post:
1. Runs all models to get predictions
2. Computes consistency score
3. Returns predicted label + confidence + explanation string + verification verdict

### Step 28 — Consistency Analysis on Validation Set
Runs the consistency pipeline on the full validation set. Computes consistency scores for all posts. Analyzes the distribution across:
- Correct vs. incorrect predictions
- Per true label class
- Score vs. accuracy correlation

### Step 29 — Consistency Visualizations
Generates **Figure 4** — three consistency analysis plots:
- **Histogram:** Consistency score distribution (correct vs. wrong predictions)
- **Boxplot:** Per-class consistency score spread
- Statistical significance annotation between correct/wrong groups

### Step 30 — Consistency Threshold Analysis
Sweeps consistency score thresholds from 0.0 to 1.0. For each threshold, computes accuracy and fraction of retained samples. Generates **Figure 5** — threshold vs. accuracy & retained samples curve. Identifies the optimal threshold for high-confidence prediction mode.

### Step 31 — Ablation Study
Systematically removes each component and measures performance drop:

| Ablation | Component Removed |
|---|---|
| Full model | — (baseline) |
| w/o CLIP | Remove CLIP alignment signal |
| w/o OCR | Remove OCR similarity feature |
| w/o Metadata | Remove XGBoost metadata branch |
| w/o MLP Fusion | Use fixed-weight ensemble instead |
| Text only | Remove image modality |
| Image only | Remove text modality |

### Step 32 — Confusion Matrices
Generates **Figure 3** — confusion matrices (Seaborn heatmaps) for each model across all 6 fake news classes. Reveals per-class misclassification patterns.

### Step 33 — Save All Results
Saves all outputs to disk:
- Best model checkpoints (`.pt` files)
- Classification reports (`.txt`)
- Results comparison table (`.csv`)
- Consistency scores DataFrame (`.csv`)
- All figures (`.pdf` and `.png`, publication-ready)

### Step 34 — Final Summary & Publication Figures
Generates all **6 publication-ready figures** with consistent styling:

| Figure | Content |
|---|---|
| Figure 1 | Training curves (loss + accuracy) |
| Figure 2 | Model comparison bar chart |
| Figure 3 | Confusion matrices (all models) |
| Figure 4 | Consistency score analysis |
| Figure 5 | Threshold vs. accuracy curve |
| Figure 6 | Per-class F1 radar chart |

---

## 🤖 Models & Components

| Component | Architecture | Role |
|---|---|---|
| **Text Encoder** | RoBERTa-base (125M params) | Semantic text understanding |
| **Image Encoder** | ResNet-50 (pretrained, ImageNet) | Visual feature extraction |
| **CLIP Model** | ViT-B/32 CLIP | Cross-modal alignment scoring |
| **OCR** | Tesseract v4 | Embedded text extraction from images |
| **Metadata Classifier** | XGBoost | Low-level visual cue classification |
| **Fusion** | Learned MLP | Late fusion of all model outputs |

---

## ⚡ Key Upgrades Over Baseline

| Component | Baseline | This Work |
|---|---|---|
| Text model | DistilBERT | **RoBERTa-base** |
| Dataset size | 100 per class | **400 per class** |
| Augmentation | Basic resize | **ColorJitter + Rotation + ResizedCrop** |
| LR strategy | Flat Adam | **Differential LR + cosine warmup** |
| Early stopping | ❌ | ✅ |
| Metadata model | RandomForest | **XGBoost** |
| Fusion | Fixed alpha | **Learned MLP** |
| Novelty | — | **Consistency Score** |

---

## 📊 Publication Figures

| # | Figure | Description |
|---|---|---|
| 1 | Training Curves | Loss & accuracy over epochs |
| 2 | Model Comparison | Bar chart: Accuracy & F1 across all models |
| 3 | Confusion Matrices | 6×6 heatmaps for each model |
| 4 | Consistency Analysis | Score distribution: correct vs. wrong |
| 5 | Threshold Curve | Accuracy vs. retained samples |
| 6 | Radar Chart | Per-class F1 for all models |

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+
- CUDA-compatible GPU (strongly recommended)
- Google Colab (recommended) or local environment

### Install Dependencies

```bash
pip install torch torchvision transformers timm
pip install pytesseract opencv-python Pillow
pip install xgboost scikit-learn pandas numpy
pip install matplotlib seaborn requests
```

### Install Tesseract OCR Engine

```bash
# Ubuntu / Colab
sudo apt-get install tesseract-ocr -y

# macOS
brew install tesseract
```

---

## ▶️ How to Run

1. **Open the notebook** — Upload `Fake_News_paper_Code.ipynb` to Google Colab.

2. **Enable GPU** — `Runtime → Change runtime type → A100 / T4 GPU`

3. **Run all cells** — `Runtime → Run all`

   The notebook will:
   - Auto-download Fakeddit TSV files
   - Download and cache post images
   - Train all models sequentially
   - Generate all figures and save results

4. **Outputs** — All figures, checkpoints, and CSV results are saved to the output directory automatically.

> ⚠️ Image downloading (Step 7) may take **20–40 minutes** depending on network speed. Cached images are reused on subsequent runs.

---

## 📈 Results

Full results are generated at runtime in **Step 25 (Model Comparison)** and **Step 34 (Final Summary)**. All 6 publication-ready figures are saved automatically.

| Metric | Reported |
|---|---|
| Accuracy | Per model (all 4 models) |
| Macro F1 | Per model |
| Per-class F1 | Via classification report + radar chart |
| Consistency Score | Distribution + threshold analysis |
| Ablation Impact | Per-component contribution |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Vision** | ResNet-50 (torchvision), CLIP (OpenAI) |
| **NLP** | RoBERTa-base (HuggingFace) |
| **OCR** | Tesseract v4 (`pytesseract`) |
| **Classical ML** | XGBoost, scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Data Handling** | pandas, NumPy |
| **Environment** | Google Colab (GPU) / Jupyter Notebook |

---

## 📄 Citation

This repository contains the implementation for our research paper. Citation details will be updated upon publication.

```bibtex
@article{fakenews_mcrf_2025,
  title   = {Multimodal Consistency Reasoning Framework for Fake News Detection},
  note    = {Under Review},
  year    = {2025}
}
```

---

*Research Implementation — Multimodal Fake News Detection with Consistency Reasoning*
