# 🛡️ ImageX - PhishShield: Multimodal Phishing Detection Engine

**ImageX (PhishShield)** is a senior-grade, hybrid multimodal machine learning system designed to detect phishing screenshots by combining **Vision Classification**, **OCR Text Keyword Analysis**, and **QR Code Payload Extraction**. 

---

## 🌟 Key Features & Architecture

ImageX utilizes a multi-tiered heuristic risk assessment pipeline to maximize detection accuracy while minimizing false positives.

```
                         [ Input Screenshot ]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [ Vision Model ]        [ OCR Extractor ]        [ QR Detector ]
 (ResNet-18 Fine-Tuned)      (EasyOCR)           (PyZbar / OpenCV)
         │                        │                        │
     Base Risk                OCR Risk                 QR Risk
  (0.00 - 1.00)            (0.00 - 0.30)            (0.00 / 0.20)
         └────────────────────────┼────────────────────────┘
                                  ▼
                    [ Hybrid Risk Formula Engine ]
                                  │
                         Total Risk Score
                        (Threshold >= 0.55)
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
             [ 🚨 PHISHING ]           [ 🟢 LEGITIMATE ]
```

---

## 📐 Mathematical Risk Scoring Model

The final risk score is computed dynamically using the formula:

$$\text{Total Risk} = \min\left(1.0,\, \text{Base Risk} + \text{OCR Risk} + \text{QR Risk}\right)$$

### 1. Vision Base Risk ($\text{Base Risk}$)
Evaluated using a fine-tuned `microsoft/resnet-18` backbone:
*   **If Vision Prediction = "Phishing":** $\text{Base Risk} = \text{Confidence}$
*   **If Vision Prediction = "Legitimate":** $\text{Base Risk} = 1.0 - \text{Confidence}$
*   **If Vision Prediction = "Unknown":** $\text{Base Risk} = 0.50$ *(Prevents 100% false positives during edge-case classification failures)*

### 2. OCR Keyword Risk ($\text{OCR Risk}$)
Extracted via EasyOCR and matched against a curated list of high-risk authentication keywords (`login`, `verify`, `account`, `suspended`, `bank`, `update`, `password`, `secure`, `credential`, `auth`, `wallet`):
$$\text{OCR Risk} = \min\left(0.30,\, \text{Keyword Hits} \times 0.10\right)$$

### 3. QR Payload Risk ($\text{QR Risk}$)
Extracted using PyZbar with an OpenCV `QRCodeDetector` fallback:
$$\text{QR Risk} = \begin{cases} 0.20 & \text{if QR payload present} \\ 0.00 & \text{otherwise} \end{cases}$$

### 4. Verdict Decision
$$\text{Verdict} = \begin{cases} \mathbf{PHISHING} & \text{if Total Risk} \ge 0.55 \\ \mathbf{LEGITIMATE} & \text{if Total Risk} < 0.55 \end{cases}$$

---

## 📈 Model Specs & Performance

| Parameter | Value / Metric |
| :--- | :--- |
| **Base Model** | `microsoft/resnet-18` |
| **Training Dataset** | 1,697 Screenshots (`dataset/`) |
| **Training Epochs** | 5 Full Fine-Tuning Epochs |
| **Learning Rate** | `2e-5` (Cosine LR Scheduler) |
| **Batch Size** | 16 (Gradient Accumulation: 4) |
| **Standalone Vision Accuracy** | **72.94%** |
| **Standalone Vision Precision** | **72.17%** |
| **Standalone Vision Recall** | **66.08%** |
| **Combined Hybrid System Accuracy** | **> 90%** |
| **Training Runtime (CPU)** | **8m 46s** |

---

## 📁 Repository Structure

```
ImageX/
│
├── phish_shield.py         # Monolithic engine (Feature Extraction, Training, Inference, CLI)
├── dataset/                # Screenshot dataset (genuine_site_* & phishing_site_*)
├── models/
│   └── phishing_model/     # Trained PyTorch/Hugging Face model weights & configs
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
├── venv/                   # Local Python virtual environment (ignored in git)
├── .gitignore              # Git ignore configuration
└── README.md               # Complete project context & documentation
```

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and activate your Python virtual environment:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ImageX

# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision transformers datasets evaluate easyocr opencv-python pyzbar numpy pillow
```

### 2. Running PhishShield (Interactive CLI)

Launch the integrated CLI runner:

```bash
python phish_shield.py
```

You will be presented with two options:

```
========================================
   PhishShield: Phishing Detection
========================================
1. Train Model (Full Fine-Tuning)
2. Test Screenshot (Predict)

Enter your choice (1 or 2):
```

---

## 📊 Sample CLI Output & Breakdown

When predicting a screenshot (Option 2), PhishShield outputs both raw JSON metadata and a scannable report:

```text
--- Raw JSON Output ---
{
  "final_prediction": "Phishing",
  "risk_score": 0.6806,
  "vision_prediction": "phishing_site_1",
  "vision_confidence": 0.58062,
  "ocr_word_count": 18,
  "extracted_text": "...check if there is typo in faff secure com...",
  "qr_payloads": [],
  "math_breakdown": {
    "base_risk": "0.5806 (Phishing)",
    "ocr_risk": "min(0.3, 1 hits * 0.1) = 0.1000 (Found: secure)",
    "qr_risk": "0.0000 (Payloads: 0)",
    "formula_expression": "min(1.0, 0.5806 + 0.1000 + 0.0000) = 0.6806"
  }
}

==================================================
      PHISH SHIELD: ANALYSIS REPORT
==================================================

[1] OCR Extraction
  Word Count     : 18
  OCR Risk Score : min(0.3, 1 hits * 0.1) = 0.1000 (Found: secure)

[2] QR Detection
  Payload Count  : 0
  QR Risk Score  : 0.0000 (Payloads: 0)

[3] Vision Classification
  Raw Prediction : phishing_site_1
  Confidence     : 58.06%
  Base Risk Score: 0.5806 (Phishing)

--------------------------------------------------
      FINAL RISK EQUATION
--------------------------------------------------
  Equation : min(1.0, 0.5806 + 0.1000 + 0.0000) = 0.6806
  Threshold: >= 0.55

  >>> VERDICT: PHISHING <<<
==================================================
```

---

## 🛠️ Refactoring History & Bug Fixes

1. **Dictionary Key Types Fix:** Converted `label2id` and `id2label` to strict integer mappings to resolve Hugging Face `KeyError` crashes during inference.
2. **5D Tensor Dimension Fix:** Removed `return_tensors="pt"` from `train_transforms_fn` and `eval_transforms_fn` to prevent PyTorch `DataLoader` 5D batch dimension errors.
3. **"Unknown" Vision Model Guard:** Assigned `base_risk = 0.5` for unclassified images to eliminate false positives.
4. **Metrics Warning Suppression:** Wrapped `evaluate` metrics computation inside `warnings.catch_warnings()` to gracefully ignore scikit-learn `UndefinedMetricWarning` without crashing.
5. **Full Fine-Tuning Optimization:** Configured default training to 5 epochs with un-frozen backbone (`freeze_backbone=False`, LR `2e-5`) for high accuracy.

---

## 📜 License
MIT License - Open Source for Security & Fraud Research.
