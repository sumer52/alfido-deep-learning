"""Evaluate the best checkpoint on CIFAR-10 and persist objective metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from dataset import create_dataloaders
from utils import get_device, load_checkpoint, set_seed, update_managed_results


def evaluate(args: argparse.Namespace) -> dict:
    """Generate predictions, metrics, report, confusion matrix, and documentation summary."""
    set_seed(args.seed); device = get_device(args.device)
    model, metadata = load_checkpoint(args.model_path, device)
    data = create_dataloaders(args.data_dir, args.batch_size, metadata["image_size"], metadata.get("val_fraction", .1),
                              metadata.get("seed", args.seed), args.num_workers, True, None, None, args.max_test_samples)
    actual, predicted, confidences = [], [], []
    with torch.inference_mode():
        for images, labels in data.test_loader:
            probabilities = model(images.to(device)).softmax(1)
            conf, pred = probabilities.max(1)
            actual.extend(labels.tolist()); predicted.extend(pred.cpu().tolist()); confidences.extend(conf.cpu().tolist())
    if not actual: raise RuntimeError("The test loader is empty.")
    accuracy = accuracy_score(actual, predicted)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    metrics = {"run_mode": metadata.get("run_mode", "unknown"), "test_samples": len(actual), "accuracy": accuracy,
               "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f_macro,
               "weighted_precision": p_weighted, "weighted_recall": r_weighted, "weighted_f1": f_weighted}
    output = Path(args.results_dir); output.mkdir(parents=True, exist_ok=True)
    report = classification_report(actual, predicted, labels=list(range(len(metadata["class_names"]))),
                                   target_names=metadata["class_names"], digits=4, zero_division=0)
    summary = "Evaluation metrics\n" + json.dumps(metrics, indent=2) + "\n\nClassification report\n" + report
    (output / "classification_report.txt").write_text(summary, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"actual": [metadata["class_names"][i] for i in actual], "predicted": [metadata["class_names"][i] for i in predicted], "confidence": confidences}).to_csv(output / "predictions.csv", index=False)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(metadata["class_names"]))))
    plt.figure(figsize=(10, 8)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=metadata["class_names"], yticklabels=metadata["class_names"])
    plt.title(f"CIFAR-10 confusion matrix ({metrics['run_mode']} run)"); plt.xlabel("Predicted label"); plt.ylabel("Actual label")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=180); plt.close()
    result_md = (f"**Latest evaluated run:** `{metrics['run_mode']}` ({len(actual):,} test images)  \n"
                 f"Accuracy: **{accuracy:.4f}** · Macro F1: **{f_macro:.4f}** · Weighted F1: **{f_weighted:.4f}**  \n"
                 "See `results/metrics.json` and `results/classification_report.txt`. Smoke-run metrics are pipeline checks, not final performance claims.")
    root = Path(__file__).resolve().parents[1]
    update_managed_results(root / "README.md", result_md); update_managed_results(root / "REPORT.md", result_md)
    print(summary); return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/best_model.pth"); parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results"); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"); parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__": evaluate(parse_args())
