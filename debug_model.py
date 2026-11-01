#!/usr/bin/env python3
# =============================================================================
# debug_model.py
# Automated Diagnostics & Verification Script for Brain Tumor System
# =============================================================================
"""
Automated Debugging Script for Brain Tumor MRI Classifier.

This script executes 10 core diagnostic steps:
1. Load trained model.
2. Load matching class_names.
3. Randomly sample at least 5 images per class (20+ images).
4. Log and inspect each image.
5. Print Actual, Predicted, Probability vector, Confidence, and Correctness.
6. Save misclassified images into debug/misclassified/.
7. Save Confusion Matrix, Classification Report, Per-Class & Overall Accuracy.
8. Print Top-3 predicted probabilities for every sample.
9. Detect root causes (class mapping, preprocessing mismatch, wrong model, confidence threshold).
10. Automatically generate a debugging report text file (debug/debug_report.txt).
"""

import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.deployment.inference import TumorClassifier
from src.preprocessing.image_utils import load_and_preprocess_image, load_image_for_display
from src.training.trainer import load_best_model
from src.utils.helpers import load_class_mapping
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_automatic_debug(
    samples_per_class: int = 5,
    output_dir: Path | str = PROJECT_ROOT / "debug",
) -> dict:
    """
    Run complete automated diagnostic procedure on the Brain Tumor System.
    """
    output_dir = Path(output_dir)
    misclassified_dir = output_dir / "misclassified"
    output_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  BRAIN TUMOR DETECTION SYSTEM -- AUTOMATIC DEBUGGING SCRIPT")
    print("=" * 80)

    # ── 1 & 2. Load Model & Class Names ─────────────────────────────────────
    model_path = Config.BEST_MODEL_PATH
    print(f"\n[1/10] Loading Model from: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}!")
        sys.exit(1)

    model = load_best_model(model_path)
    mapping = load_class_mapping()
    class_names = mapping.get("class_names") or Config.CLASS_NAMES
    print(f"[2/10] Loaded Class Names ({len(class_names)}): {class_names}")

    classifier = TumorClassifier(model_path=model_path, class_names=class_names)

    # ── 3. Sample Images ────────────────────────────────────────────────────
    print(f"\n[3/10] Sampling at least {samples_per_class} images per class from Testing set...")
    test_dir = Config.TEST_DIR
    sampled_files = []
    
    for cls in class_names:
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            print(f"WARNING: Directory for class {cls} missing in {test_dir}")
            continue
        all_imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        selected = random.sample(all_imgs, min(samples_per_class, len(all_imgs)))
        for img_p in selected:
            sampled_files.append((img_p, cls))

    print(f"Total sampled test images: {len(sampled_files)}")

    # ── 4, 5 & 8. Evaluate Each Sample & Top-3 Probabilities ───────────────
    print("\n[4, 5 & 8/10] Running Inference & Inspecting Predictions:")
    print("-" * 80)

    results = []
    misclassified_count = 0

    for idx, (img_path, actual_class) in enumerate(sampled_files, 1):
        # Load and predict
        res = classifier.predict_from_path(img_path)
        actual_idx = class_names.index(actual_class)
        pred_idx = res.class_idx
        is_correct = (actual_idx == pred_idx)

        # Get sorted top probabilities
        sorted_probs = sorted(res.all_probabilities.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_probs[:3]

        print(f"\nSample #{idx:02d} | File: {img_path.name}")
        print(f"  Actual Label     : {actual_class} (Index {actual_idx})")
        print(f"  Predicted Label  : {res.class_name} (Index {pred_idx})")
        print(f"  Confidence Score : {res.confidence:.2%}")
        print(f"  Result           : {'[CORRECT]' if is_correct else '[MISCLASSIFIED]'}")
        print(f"  Top-3 Predictions:")
        for rank, (c_name, p_val) in enumerate(top3, 1):
            print(f"    {rank}. {c_name:<18} : {p_val:.2%}")

        # ── 6. Save Misclassified Images ────────────────────────────────────
        if not is_correct:
            misclassified_count += 1
            save_name = f"misclassified_{idx:02d}_true_{actual_class}_pred_{res.class_name}_{img_path.name}"
            dest_p = misclassified_dir / save_name
            
            # Save original image to misclassified folder
            display_img = load_image_for_display(img_path)
            plt.imsave(dest_p, display_img)
            print(f"  [SAVED MISCLASSIFIED IMAGE] -> {dest_p}")

        results.append({
            "filename": img_path.name,
            "filepath": str(img_path),
            "actual_class": actual_class,
            "actual_idx": actual_idx,
            "predicted_class": res.class_name,
            "predicted_idx": pred_idx,
            "confidence": res.confidence,
            "is_correct": is_correct,
            "is_confident": res.is_confident,
            "top1_class": top3[0][0], "top1_prob": top3[0][1],
            "top2_class": top3[1][0] if len(top3) > 1 else "", "top2_prob": top3[1][1] if len(top3) > 1 else 0.0,
            "top3_class": top3[2][0] if len(top3) > 2 else "", "top3_prob": top3[2][1] if len(top3) > 2 else 0.0,
        })

    # ── 7. Generate Metrics, Confusion Matrix & Reports ──────────────────────
    print("\n[7/10] Computing Metrics, Confusion Matrix, and Reports...")
    
    y_true = [r["actual_idx"] for r in results]
    y_pred = [r["predicted_idx"] for r in results]
    
    overall_acc = float(accuracy_score(y_true, y_pred))
    
    # Per-class accuracy
    per_class_acc = {}
    for i, c_name in enumerate(class_names):
        c_true = [t for t, p in zip(y_true, y_pred) if t == i]
        c_pred = [p for t, p in zip(y_true, y_pred) if t == i]
        acc_c = accuracy_score(c_true, c_pred) if c_true else 0.0
        per_class_acc[c_name] = float(acc_c)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Diagnostic Confusion Matrix (N={len(results)})\nAccuracy: {overall_acc:.2%}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved Confusion Matrix plot: {cm_path}")

    # Classification report CSV
    clf_rep_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    df_rep = pd.DataFrame(clf_rep_dict).T.round(4)
    rep_csv_path = output_dir / "classification_report.csv"
    df_rep.to_csv(rep_csv_path)
    print(f"  Saved Classification Report CSV: {rep_csv_path}")

    # Accuracy Summary JSON
    accuracy_summary = {
        "overall_accuracy": overall_acc,
        "total_samples": len(results),
        "correct_samples": len(results) - misclassified_count,
        "misclassified_samples": misclassified_count,
        "per_class_accuracy": per_class_acc,
    }
    acc_json_path = output_dir / "accuracy_summary.json"
    with open(acc_json_path, "w") as f:
        json.dump(accuracy_summary, f, indent=4)
    print(f"  Saved Accuracy Summary JSON: {acc_json_path}")

    # ── 9. Detect Root Cause Issues ──────────────────────────────────────────
    print("\n[9/10] Running Automated Root Cause Diagnostic Checks:")
    diagnostics = []

    # Check A: Class Mapping
    detected_on_disk = sorted([d.name for d in Config.TRAIN_DIR.iterdir() if d.is_dir()]) if Config.TRAIN_DIR.exists() else []
    if detected_on_disk != class_names:
        diagnostics.append(
            f"[FAIL] CLASS MAPPING MISMATCH DETECTED:\n"
            f"   Dataset folders: {detected_on_disk}\n"
            f"   Loaded class_names: {class_names}"
        )
    else:
        diagnostics.append("[PASS] Class Mapping Check: PASSED (Folders and model class_names match perfectly).")

    # Check B: Preprocessing Mismatch
    sample_path = sampled_files[0][0]
    arr_path = load_and_preprocess_image(sample_path)
    if arr_path.shape != (1, 224, 224, 3) or arr_path.max() > 1.0 or arr_path.min() < 0.0:
        diagnostics.append(
            f"[FAIL] PREPROCESSING MISMATCH DETECTED:\n"
            f"   Processed shape: {arr_path.shape}, min: {arr_path.min()}, max: {arr_path.max()}"
        )
    else:
        diagnostics.append("[PASS] Preprocessing Check: PASSED (Image tensor shape (1, 224, 224, 3), range [0, 1]).")

    # Check C: Model Loaded
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 4):
        diagnostics.append(
            f"[FAIL] WRONG MODEL ARCHITECTURE DETECTED:\n"
            f"   Input shape: {model.input_shape}, Output shape: {model.output_shape}"
        )
    else:
        diagnostics.append("[PASS] Model Architecture Check: PASSED (Input (None, 224, 224, 3), Output (None, 4)).")

    # Check D: Confidence Thresholding & Low Confidence Warning Cause
    avg_conf = np.mean([r["confidence"] for r in results])
    low_conf_count = sum(1 for r in results if not r["is_confident"])
    if low_conf_count > 0:
        diagnostics.append(
            f"[WARNING] LOW CONFIDENCE PREDICTIONS DETECTED:\n"
            f"   {low_conf_count}/{len(results)} samples fell below required threshold ({Config.CONFIDENCE_THRESHOLD:.0%}).\n"
            f"   Average confidence across samples: {avg_conf:.2%}.\n"
            f"   Root Cause: Model under-training or imbalance penalty causes softmax scores < 75%."
        )
    else:
        diagnostics.append(f"[PASS] Confidence Threshold Check: PASSED (All predictions exceed {Config.CONFIDENCE_THRESHOLD:.0%} confidence).")

    for diag in diagnostics:
        print(f"  {diag}")

    # ── 10. Generate Debug Report File ───────────────────────────────────────
    print("\n[10/10] Writing Diagnostic Report to debug/debug_report.txt...")
    report_file = output_dir / "debug_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  BRAIN TUMOR DETECTION SYSTEM — AUTOMATED DEBUG REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. SUMMARY OF EVALUATION:\n")
        f.write(f"   - Total Sampled Images       : {len(results)}\n")
        f.write(f"   - Overall Accuracy           : {overall_acc:.2%}\n")
        f.write(f"   - Correct Samples             : {len(results) - misclassified_count}\n")
        f.write(f"   - Misclassified Samples       : {misclassified_count}\n")
        f.write(f"   - Average Confidence Score    : {avg_conf:.2%}\n\n")

        f.write("2. PER-CLASS ACCURACY BREAKDOWN:\n")
        for c_name, acc in per_class_acc.items():
            f.write(f"   - {c_name:<20} : {acc:.2%}\n")
        f.write("\n")

        f.write("3. DETECTED SYSTEM DIAGNOSTICS & ROOT CAUSES:\n")
        for diag in diagnostics:
            f.write(f"   {diag}\n")
        f.write("\n")

        f.write("4. DETAILED SAMPLE RESULTS:\n")
        f.write(f"{'Filename':<20} {'Actual Class':<18} {'Predicted Class':<18} {'Conf':<8} {'Status':<12}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            status_str = "CORRECT" if r["is_correct"] else "MISCLASSIFIED"
            f.write(f"{r['filename']:<20} {r['actual_class']:<18} {r['predicted_class']:<18} {r['confidence']:<8.2%} {status_str:<12}\n")

    print(f"  Saved debug report to: {report_file}")
    print("\n" + "=" * 80)
    print("  DEBUGGING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return {
        "overall_accuracy": overall_acc,
        "misclassified_count": misclassified_count,
        "results": results,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    run_automatic_debug(samples_per_class=5)
