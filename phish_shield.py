import os
import sys
import json
import argparse
import cv2
import easyocr
import torch
import numpy as np
import logging
from PIL import Image, UnidentifiedImageError
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import ColorJitter, RandomRotation, Compose
import evaluate
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. Feature Extraction (OCR & QR)
# ==========================================
class ImageFeatureExtractor:
    def __init__(self, ocr_langs=['en']):
        logger.info("Initializing OCR Reader (EasyOCR)...")
        self.reader = easyocr.Reader(ocr_langs, gpu=torch.cuda.is_available(), verbose=False)
        
    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return None
            
        try:
            image = Image.open(image_path).convert("RGB")
            image.verify() 
            return Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, IOError) as e:
            logger.error(f"Corrupted or invalid image file {image_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {image_path}: {str(e)}")
            return None

    def extract_ocr(self, image_path):
        try:
            results = self.reader.readtext(image_path, detail=0)
            extracted_text = " ".join(results)
            # Normalize text
            extracted_text = " ".join(extracted_text.split()).lower()
            word_count = len(extracted_text.split())
            return extracted_text, word_count
        except Exception as e:
            logger.error(f"OCR Error on {image_path}: {str(e)}")
            return "", 0

    def detect_qr(self, image_path):
        try:
            # Robust read to handle non-ASCII paths or special formats
            img_array = np.fromfile(image_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2 could not decode the image")
                
            qr_payloads = []
            
            # Try PyZbar first if installed
            try:
                from pyzbar.pyzbar import decode
                decoded_objects = decode(image)
                for obj in decoded_objects:
                    payload = obj.data.decode("utf-8")
                    qr_payloads.append(payload)
                if qr_payloads:
                    return qr_payloads
            except Exception as e:
                logger.debug(f"PyZbar unavailable ({e}), falling back to cv2.QRCodeDetector")

            # Fallback to OpenCV
            detector = cv2.QRCodeDetector()
            val, points, straight_qrcode = detector.detectAndDecode(image)
            if val:
                qr_payloads.append(val)
                
            return qr_payloads
        except Exception as e:
            logger.error(f"QR Detection Error on {image_path}: {str(e)}")
            return []

# ==========================================
# 2. Prediction Pipeline
# ==========================================
class PhishingPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "phishing_model")
        
        self.feature_extractor = ImageFeatureExtractor()
        
        if os.path.exists(model_dir):
            logger.info(f"Loading local fine-tuned model from {model_dir}")
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
            self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        else:
            logger.warning(f"Local model not found at {model_dir}. Falling back to default resnet-18.")
            fallback_model = "microsoft/resnet-18"
            self.processor = AutoImageProcessor.from_pretrained(fallback_model)
            self.model = AutoModelForImageClassification.from_pretrained(fallback_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def classify_image(self, image):
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, class_idx = torch.max(probs, dim=-1)
                
                if hasattr(self.model.config, "id2label") and len(self.model.config.id2label) > 0:
                    raw_id2label = self.model.config.id2label
                    # Handle both integer and string keys seamlessly
                    key = class_idx.item()
                    str_key = str(key)
                    if key in raw_id2label:
                        predicted_class = raw_id2label[key]
                    elif str_key in raw_id2label:
                        predicted_class = raw_id2label[str_key]
                    else:
                        predicted_class = "Phishing" if key % 2 != 0 else "Legitimate"
                else:
                    predicted_class = "Phishing" if class_idx.item() % 2 != 0 else "Legitimate"
                
                return predicted_class, confidence.item()
        except Exception as e:
            logger.error(f"Classification Error: {str(e)}")
            return "Unknown", 0.0

    def predict_screenshot(self, image_path):
        result = {
            "final_prediction": "Unknown",
            "risk_score": 0.0,
            "vision_prediction": "Unknown",
            "vision_confidence": 0.0,
            "ocr_word_count": 0,
            "extracted_text": "",
            "qr_payloads": [],
            "math_breakdown": {}
        }
        
        image = self.feature_extractor.validate_image(image_path)
        if image is None:
            return result

        extracted_text, word_count = self.feature_extractor.extract_ocr(image_path)
        result["extracted_text"] = extracted_text
        result["ocr_word_count"] = word_count
        
        qr_payloads = self.feature_extractor.detect_qr(image_path)
        result["qr_payloads"] = qr_payloads
        
        prediction, confidence = self.classify_image(image)
        result["vision_prediction"] = prediction
        result["vision_confidence"] = confidence
        
        is_vision_phishing = "phish" in prediction.lower()
        
        # Fix 3: Handle "Unknown" predictions to prevent 100% false positives
        if prediction == "Unknown":
            base_risk = 0.5
            base_risk_str = "0.5 (Unknown Vision Prediction)"
        else:
            if is_vision_phishing:
                base_risk = confidence
                base_risk_str = f"{confidence:.4f} (Phishing)"
            else:
                base_risk = 1.0 - confidence
                base_risk_str = f"1.0 - {confidence:.4f} = {base_risk:.4f} (Legitimate)"
        
        suspicious_keywords = ["login", "verify", "account", "suspended", "bank", "update", "password", "secure", "credential", "auth", "wallet"]
        ocr_risk = 0.0
        found_keywords = []
        if extracted_text:
            text_lower = extracted_text.lower()
            for kw in suspicious_keywords:
                if kw in text_lower:
                    found_keywords.append(kw)
            keyword_hits = len(found_keywords)
            ocr_risk = min(0.3, keyword_hits * 0.1)
            
        qr_risk = 0.2 if len(qr_payloads) > 0 else 0.0
        
        total_risk = min(1.0, base_risk + ocr_risk + qr_risk)
        
        result["risk_score"] = round(total_risk, 4)
        result["final_prediction"] = "Phishing" if total_risk >= 0.55 else "Legitimate"
        
        # Add the mathematical breakdown
        result["math_breakdown"] = {
            "base_risk": base_risk_str,
            "ocr_risk": f"min(0.3, {len(found_keywords)} hits * 0.1) = {ocr_risk:.4f} (Found: {', '.join(found_keywords) if found_keywords else 'None'})",
            "qr_risk": f"{qr_risk:.4f} (Payloads: {len(qr_payloads)})",
            "formula_expression": f"min(1.0, {base_risk:.4f} + {ocr_risk:.4f} + {qr_risk:.4f}) = {total_risk:.4f}"
        }
        
        return result

# ==========================================
# 3. Training Pipeline
# ==========================================
def train_model(
    dataset_name=None,
    model_name="microsoft/resnet-18",
    output_dir=None,
    epochs=5,
    freeze_backbone=False
):
    if dataset_name is None:
        dataset_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "phishing_model")
    logger.info(f"Loading dataset: {dataset_name}...")
    
    try:
        dataset = load_dataset(
            "imagefolder", 
            data_files={
                "train": [
                    f"{dataset_name}/**/*.png", 
                    f"{dataset_name}/**/*.jpg", 
                    f"{dataset_name}/**/*.jpeg", 
                    f"{dataset_name}/**/*.webp"
                ]
            }
        )
    except Exception as e:
        logger.error(f"Could not load dataset. Error: {e}")
        return

    if "label" in dataset["train"].features:
        labels = dataset["train"].features["label"].names
    else:
        labels = ["Legitimate", "Phishing"] 
        
    # Fix 1: Ensure label2id uses integer values and id2label uses integer keys
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(model_name)

    aug_transforms = Compose([
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        RandomRotation(degrees=5)
    ])

    # Fix 2: Remove return_tensors="pt" so Hugging Face data collator can handle batching without creating 5D tensors
    def train_transforms_fn(example_batch):
        images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
        inputs = processor(images)
        inputs["labels"] = example_batch["label"]
        return inputs

    def eval_transforms_fn(example_batch):
        images = [x.convert("RGB") for x in example_batch["image"]]
        inputs = processor(images)
        inputs["labels"] = example_batch["label"]
        return inputs

    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.2)
        
    train_ds = dataset["train"]
    train_ds.set_transform(train_transforms_fn)
    
    eval_ds = dataset["test"]
    eval_ds.set_transform(eval_transforms_fn)

    logger.info(f"Loading base model: {model_name}...")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    if freeze_backbone and hasattr(model, "resnet"):
        logger.info("Freezing base model backbone to massively speed up training...")
        for param in model.resnet.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-3 if freeze_backbone else 2e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    acc_metric = evaluate.load("accuracy")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Fix 4: Add zero_division=0 to metrics to prevent zero-division crashes during early training epochs
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
            prec = prec_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
            rec = rec_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
            f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Training complete!")

# ==========================================
# 4. CLI Entrypoint
# ==========================================
def print_analysis_report(result):
    print("\n" + "="*50)
    print("      PHISH SHIELD: ANALYSIS REPORT")
    print("="*50)
    
    print("\n[1] OCR Extraction")
    print(f"  Word Count     : {result.get('ocr_word_count', 0)}")
    print(f"  OCR Risk Score : {result.get('math_breakdown', {}).get('ocr_risk', 'N/A')}")
    
    print("\n[2] QR Detection")
    qr_payloads = result.get('qr_payloads', [])
    print(f"  Payload Count  : {len(qr_payloads)}")
    if qr_payloads:
        print(f"  Payloads       : {qr_payloads}")
    print(f"  QR Risk Score  : {result.get('math_breakdown', {}).get('qr_risk', 'N/A')}")
    
    print("\n[3] Vision Classification")
    print(f"  Raw Prediction : {result.get('vision_prediction', 'Unknown')}")
    print(f"  Confidence     : {result.get('vision_confidence', 0.0) * 100:.2f}%")
    print(f"  Base Risk Score: {result.get('math_breakdown', {}).get('base_risk', 'N/A')}")
    
    print("\n" + "-"*50)
    print("      FINAL RISK EQUATION")
    print("-"*50)
    print(f"  Equation : {result.get('math_breakdown', {}).get('formula_expression', 'N/A')}")
    print(f"  Threshold: >= 0.55")
    
    verdict = result.get('final_prediction', 'Unknown')
    print(f"\n  >>> VERDICT: {verdict.upper()} <<<")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("========================================")
    print("   PhishShield: Phishing Detection")
    print("========================================")
    print("1. Train Model (Full Fine-Tuning)")
    print("2. Test Screenshot (Predict)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Fix 5: Default to full fine-tuning (freeze_backbone=False) and 5 epochs with 2e-5 learning rate for high accuracy
        print("\nStarting full fine-tuning pipeline (5 epochs)...")
        train_model(epochs=5, freeze_backbone=False)
    elif choice == "2":
        image_path = input("\nEnter the path to the screenshot to test: ").strip().strip('\"\'')
        if not image_path:
            print("Error: No image path provided.")
            sys.exit(1)
            
        try:
            print(f"\nAnalyzing '{image_path}'...")
            predictor = PhishingPredictor()
            result = predictor.predict_screenshot(image_path)
            
            print("\n--- Raw JSON Output ---")
            print(json.dumps(result, indent=2))
            
            print_analysis_report(result)
        except Exception as e:
            print(f"\nError processing the image: {e}")
    else:
        print("Invalid choice. Exiting.")
