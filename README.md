# Osteoarthritis Imaging Analysis — Research Decision-Support System

> ⚠️ **Medical Disclaimer**: This is a **research/educational prototype**. It is **not a medical device**, has **not been approved by any regulatory body**, and must **not be used for clinical diagnosis**. All outputs require review and interpretation by a qualified radiologist or clinician.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Medical/Research Objective](#2-medicalresearch-objective)
3. [System Architecture](#3-system-architecture)
4. [Installation](#4-installation)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Model Selection & Strategy](#6-model-selection--strategy)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Running the Application](#9-running-the-application)
10. [Example Input/Output](#10-example-inputoutput)
11. [Explainability](#11-explainability)
12. [Limitations](#12-limitations)
13. [Ethical Considerations](#13-ethical-considerations)
14. [Privacy Considerations](#14-privacy-considerations)
15. [Reproducibility](#15-reproducibility)
16. [License](#16-license)

---

## 1. Project Overview

This project is a **research prototype** for detecting osteoarthritis (OA) related radiographic risk markers in knee X-ray images. It combines classical computer-vision preprocessing (OpenCV), deep-learning inference (PyTorch / EfficientNet-B4), gradient-based explainability (Grad-CAM), and a Streamlit web interface into a single end-to-end pipeline.

The system does **not** produce medical diagnoses. Instead, it identifies imaging features commonly associated with OA and presents them as model-derived risk markers requiring clinical review.

### Key Features

- 📁 Supports JPG, PNG, and DICOM (.dcm) input
- 🔬 OpenCV preprocessing: grayscale, denoise, CLAHE, sharpen, resize, normalize
- 🦴 Joint-region localization via morphological CV
- 🧠 EfficientNet-B4 backbone with a two-class OA classification head
- 🗺️ Grad-CAM and Grad-CAM++ heatmaps for explainability
- 📊 Structured risk markers: joint-space narrowing, osteophytes, subchondral sclerosis, bone deformity, alignment abnormality, cystic changes
- 📝 JSON + HTML report generation
- 🏋️ Full training pipeline with transfer learning, focal loss, early stopping
- ✅ pytest test suite using only synthetic images

---

## 2. Medical/Research Objective

Osteoarthritis is the most common joint disease worldwide, characterized radiographically by joint-space narrowing, osteophyte formation, subchondral sclerosis, and bone deformity. The Kellgren-Lawrence (KL) grading system (0–4) is the standard radiographic classification.

**Objective**: Provide a reproducible, open-source research pipeline that:
1. Accepts knee X-rays as input.
2. Applies a validated preprocessing pipeline.
3. Runs a pretrained CNN to score OA-related abnormality.
4. Presents structured, uncertainty-aware risk-marker findings.
5. Generates Grad-CAM explanations to show which image regions drove the prediction.
6. Produces downloadable research reports.

**This system is not intended to replace radiologist assessment.** It is designed for research reproducibility, academic study, and as a scaffold for future fine-tuning on labeled OA datasets.

---

## 3. System Architecture

```
X-ray Image (JPG / PNG / DICOM)
         ↓
  ImageLoader  [preprocessing/image_loader.py]
  - Format detection + validation
  - DICOM windowing + PII stripping
         ↓
  OpenCV Pipeline  [preprocessing/enhancement.py]
  - Grayscale → Denoise (NLM) → CLAHE → Sharpen → Resize → Normalize
         ↓
  Joint Region Localization  [models/segmentation.py]
  - Morphological ops + contour detection → ROI bounding box
         ↓
  EfficientNet-B4  [models/classifier.py]
  - Pretrained backbone (ImageNet) + OA classification head
  - Outputs: OA probability, confidence level
         ↓
  Risk Marker Extraction  [inference/risk_markers.py]
  - Model score + CV proxy measurements per marker
  - status: detected / possible / uncertain / not_detected
         ↓
  Grad-CAM Explainability  [models/explainability.py]
  - Gradient hooks on last conv block
  - Heatmap + colored overlay
         ↓
  Structured Report  [reports/report_generator.py]
  - JSON (machine-readable) + HTML (human-readable)
  - All outputs include medical disclaimers
```

### Directory Layout

```
oa_detection/
├── app/               # Streamlit UI
│   ├── main.py
│   ├── ui/            # upload, results, sidebar components
│   └── components/    # disclaimer component
├── preprocessing/     # OpenCV pipeline + DICOM loader
├── models/            # base_model, classifier, segmentation, explainability
├── inference/         # predictor orchestrator + risk_markers
├── training/          # dataset, train.py, evaluate.py
├── evaluation/        # metrics + visualization
├── reports/           # report_generator + HTML template
├── tests/             # pytest test suite (synthetic data only)
├── configs/           # oa_model.yaml
├── models/            # saved checkpoints (gitignored)
├── dataset/           # training data (gitignored)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 4. Installation

### Prerequisites

- Python 3.11+
- pip
- (Optional) CUDA-capable GPU for training

### Local Installation

```bash
# Clone the repository
git clone <repo-url>
cd oa_detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package in development mode
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t oa-detection:latest .
docker run -p 8501:8501 oa-detection:latest
```

Then open http://localhost:8501 in your browser.

---

## 5. Dataset Preparation

The system is designed to work with publicly available research datasets.

### Recommended Datasets

| Dataset | Description | License |
|---|---|---|
| **OAI** (Osteoarthritis Initiative) | ~4,800 patients, bilateral knee X-rays, KL grades | Free for research (registration required) |
| **MOST** | Multi-center OA study, knee X-rays | Free for research |
| **Kaggle Knee X-ray** | Community dataset, ~9,000 images | CC BY 4.0 |

### Directory Structure

Create the following structure:

```
dataset/
├── train/
│   ├── normal/          # KL 0-1 images
│   │   ├── patient001_left.png
│   │   └── ...
│   └── oa_related/      # KL 2-4 images
│       ├── patient042_right.png
│       └── ...
├── validation/
│   ├── normal/
│   └── oa_related/
└── test/
    ├── normal/
    └── oa_related/
```

### Patient-Level Split

**Critical**: The system automatically enforces patient-level splits. Filenames must include the patient ID as the prefix (before the first underscore): `{patient_id}_{side}_{visit}.png`. This prevents data leakage when a patient has multiple images.

### Class Balancing

The `OADataset.get_class_weights()` method computes inverse-frequency class weights automatically. These are passed to the loss function during training.

---

## 6. Model Selection & Strategy

### Primary Model: EfficientNet-B4

**Architecture**: EfficientNet-B4 (pretrained on ImageNet-1K via torchvision)  
**Backbone license**: BSD 3-Clause (torchvision)  
**Input size**: 384×384 pixels (configurable)  
**Head**: `Linear(1792, 512) → BatchNorm → Dropout(0.3) → Linear(512, 2)`  
**Output classes**: `[normal, oa_related]`

### ⚠️ Important: Pretrained Weights Disclaimer

The default model uses **ImageNet** pretrained weights. **These weights have NOT been trained or validated on X-ray or OA data.** All predictions in this default mode reflect general image features and must be treated as demonstrations requiring task-specific validation.

To use the model for meaningful OA assessment, you must fine-tune it on a labeled OA dataset (see [Training](#7-training)).

### Model Abstraction Layer

All models implement the `OAModel` abstract base class:

```python
class OAModel(ABC):
    def predict(self, preprocessed: PreprocessingResult) -> ModelPrediction: ...
    def explain_prediction(self, preprocessed, prediction) -> ExplainabilityResult: ...
    def load(self, checkpoint_path: Optional[str] = None) -> None: ...
```

Adding a new model (e.g., ResNet-50, DenseNet-121, ViT) requires only implementing this interface and registering it in `configs/oa_model.yaml`.

### Kellgren-Lawrence Grading

KL grading is **not implemented in the default model**. The `kl_grade` field in all reports is `null` unless a fine-tuned, KL-validated model is loaded. Do not interpret the absence of KL grading as a limitation — it is an honest reflection of the model's validation state.

---

## 7. Training

### Quick Start

```bash
# Prepare dataset (see Section 5)
# Then run:
python training/train.py --config configs/oa_model.yaml

# Resume from checkpoint
python training/train.py --config configs/oa_model.yaml --resume models/checkpoint_last.pth

# Dry run (validate dataset + config without training)
python training/train.py --config configs/oa_model.yaml --dry-run
```

### Training Strategy

| Phase | Epochs | Backbone | LR |
|---|---|---|---|
| Phase 1 (warmup) | 5 (configurable) | Frozen | 10× base LR |
| Phase 2 (fine-tune) | Up to 45 | Last 2 blocks unfrozen | Base LR → 0.01× (cosine) |

### Configurable Parameters (`configs/oa_model.yaml`)

```yaml
training:
  batch_size: 16
  learning_rate: 0.0001
  epochs: 50
  warmup_epochs: 5
  early_stopping_patience: 10
  use_focal_loss: true
  focal_loss_gamma: 2.0
  random_seed: 42
```

### Outputs

- `models/best_model.pth` — best checkpoint by validation loss
- `models/checkpoint_last.pth` — last epoch checkpoint
- `models/training_history.json` — loss/accuracy per epoch

---

## 8. Evaluation

```bash
python training/evaluate.py \
    --config configs/oa_model.yaml \
    --checkpoint models/best_model.pth \
    --output-dir reports/
```

### Reported Metrics

| Metric | Notes |
|---|---|
| Accuracy | Overall correctness |
| Precision | Positive predictive value |
| **Recall / Sensitivity** | **Critical for screening — missed OA cases** |
| **Specificity** | **True negative rate** |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Discriminative ability across thresholds |
| PR-AUC | Performance under class imbalance |
| **False Negative Rate** | **Clinically most important — missed disease** |
| False Positive Rate | Over-referral rate |
| ECE (calibration) | Confidence calibration quality |

### Evaluation Plots

Saved to `reports/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `calibration_curve.png`
- `evaluation_report.json`

> ⚠️ For medical screening, **sensitivity** should be prioritized over accuracy. A high false-negative rate (missed OA) is clinically more dangerous than a high false-positive rate.

---

## 9. Running the Application

### Streamlit Web App

```bash
streamlit run app/main.py
```

Open http://localhost:8501

### Application Workflow

1. **Upload** — Drag and drop a knee X-ray (JPG, PNG, or DICOM).
2. **Preview** — Original image is shown immediately.
3. **Analyze** — Click "Analyze" to run the full pipeline.
4. **Review** — See:
   - Original vs. preprocessed images (side by side)
   - OA-related abnormality score (0–100%)
   - Confidence level (low / moderate / high)
   - Risk marker table (6 markers with status and confidence)
   - Grad-CAM heatmap (labeled as model explanation)
5. **Download** — JSON report or HTML report.

### Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html
```

---

## 10. Example Input/Output

### Input
- Single knee X-ray, AP (anteroposterior) view
- File: `knee_xray.png` (any resolution, will be resized to 384×384)

### JSON Report Output

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "generated_at": "2026-09-29T12:00:00Z",
  "study_type": "knee_xray",
  "joint_type": "knee",
  "model": "efficientnet_b4_oa",
  "model_version": "0.1.0",
  "requires_finetuning": true,
  "oa_related_score": 0.78,
  "confidence_level": "moderate",
  "risk_markers": {
    "joint_space_narrowing": {
      "status": "possible",
      "confidence": 0.72,
      "note": "Proxy measurement from image intensity analysis."
    },
    "osteophytes": {
      "status": "detected",
      "confidence": 0.81,
      "note": "Elevated edge density near bone margins."
    },
    "subchondral_sclerosis": {
      "status": "uncertain",
      "confidence": 0.51,
      "note": "Intensity analysis inconclusive."
    },
    "bone_deformity": { "status": "not_detected", "confidence": 0.31 },
    "alignment_abnormality": { "status": "uncertain", "confidence": 0.44 },
    "cystic_changes": { "status": "not_detected", "confidence": 0.28 }
  },
  "kl_grade": null,
  "kl_grade_confidence": null,
  "requires_clinical_review": true,
  "disclaimer": "Research/decision-support prototype. Not a medical diagnosis..."
}
```

---

## 11. Explainability

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) computes a heatmap showing which spatial regions of the input image most influenced the model's prediction.

**How it works**:
1. Forward pass registers feature maps from the target conv layer.
2. Backward pass computes gradients of the predicted class score w.r.t. those feature maps.
3. Global average pooling of gradients → neuron importance weights.
4. Weighted sum of feature maps → ReLU → upsampled heatmap.

### Grad-CAM++

An extension of Grad-CAM using second-order gradient information for improved localization of multiple objects in the same image.

### ⚠️ Critical Disclaimer

> Grad-CAM heatmaps reflect **internal neural network activations**, not radiological anatomy. A highlighted region does **not** prove the presence of an anatomical abnormality. Heatmaps should be used to understand model behavior, not to locate disease.

---

## 12. Limitations

1. **No clinical validation**: The default model uses ImageNet weights and has not been validated on any OA dataset.
2. **No KL grading**: KL grading requires a model fine-tuned on labeled KL-grade data.
3. **Single joint**: Currently optimized for knee X-rays. Other joints require re-training.
4. **View dependence**: Designed for AP (anteroposterior) views. Lateral or oblique views may produce unreliable outputs.
5. **Image quality**: Very low-quality, under-exposed, or over-exposed images will produce warnings and potentially unreliable predictions.
6. **Population generalization**: Performance may vary across different X-ray acquisition protocols, patient populations, and imaging equipment.
7. **No DICOM metadata validation**: Modality detection is heuristic; incorrect modality inputs are flagged but not blocked.
8. **Risk markers are proxies**: CV-derived marker estimates (edge density, intensity profiles) are proxies, not validated radiological measurements.

---

## 13. Ethical Considerations

- **Non-maleficence**: The system is designed to assist, not replace, clinical judgment. All outputs include mandatory disclaimers.
- **Transparency**: The model backbone (EfficientNet-B4, ImageNet), its limitations, and uncertainty are always reported.
- **Equity**: Performance across demographic groups (age, sex, ethnicity, BMI) has not been evaluated. The system may perform differently across populations.
- **Autonomy**: Clinicians retain full decision-making authority. The system provides information, not recommendations.
- **Accountability**: The system is clearly labeled as a research prototype. Developers are not responsible for clinical decisions.

---

## 14. Privacy Considerations

- **DICOM PII stripping**: All PII fields (PatientName, PatientID, PatientBirthDate, etc.) are stripped before processing and are never logged or displayed.
- **No data storage**: The web application does not persist uploaded images. Files are processed in-memory and deleted after the session.
- **Local processing**: All inference runs locally. No images are sent to external servers.
- **Logs**: Application logs contain only image paths and processing metadata — no patient identifiers.
- **Report redaction**: Generated reports display "REDACTED" for all patient fields.

---

## 15. Reproducibility

All experiments use:
- **Python**: 3.11+
- **Random seed**: 42 (configurable in `configs/oa_model.yaml`)
- **Seeds set for**: Python random, NumPy, PyTorch, CUDA
- **Deterministic training**: `torch.backends.cudnn.deterministic = True`
- **Pinned dependencies**: See `requirements.txt`

To reproduce a training run:
```bash
python training/train.py --config configs/oa_model.yaml
# The training_history.json will record exact epoch metrics
```

---

## 16. License

- **Project code**: MIT License
- **EfficientNet-B4 weights** (torchvision): BSD 3-Clause License
- **PyTorch**: BSD License
- **OpenCV**: Apache 2.0 License

If you use a public dataset, cite it according to its terms:
- OAI: cite the OAI publication and acknowledge NIH funding
- MOST: cite the MOST consortium
- Kaggle datasets: follow the dataset-specific license

---

*OA Imaging Analysis Research System v0.1.0 — For research use only.*
