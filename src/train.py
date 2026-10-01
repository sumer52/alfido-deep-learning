"""Train a transfer-learning ResNet18 classifier on CIFAR-10."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import IMAGENET_MEAN, IMAGENET_STD, create_dataloaders
from utils import build_model, get_device, plot_training_curves, save_checkpoint, set_seed, unfreeze_layer4

# Measured per-epoch metrics from the interrupted first leg of the full run
# (logs/full_train.log); used only by --resume to keep the curves complete.
MEASURED_HISTORY = {
    "train_loss": [1.1764, 0.9220, 0.8953, 0.5595, 0.3868, 0.3129],
    "val_loss": [0.9295, 0.8614, 0.8389, 0.3585, 0.3296, 0.2862],
    "train_accuracy": [60.27, 68.35, 69.38, 80.54, 86.54, 88.95],
    "val_accuracy": [70.14, 71.84, 72.08, 87.46, 88.70, 90.34],
}


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train or validation epoch and return mean loss and accuracy."""
    training = optimizer is not None; model.train(training)
    loss_sum = 0.0; correct = 0; total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images); loss = criterion(outputs, labels)
            if training: loss.backward(); optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def train(args: argparse.Namespace) -> dict:
    """Train, checkpoint best validation model, and save history/curves."""
    set_seed(args.seed); device = get_device(args.device)
    data = create_dataloaders(args.data_dir, args.batch_size, args.image_size, args.val_fraction,
                              args.seed, args.num_workers, True, args.max_train_samples,
                              args.max_val_samples, args.max_test_samples)
    model = build_model(len(data.class_names), pretrained=not args.no_pretrained, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}
    best_loss = float("inf"); best_accuracy = 0.0; stale = 0; fine_tuned = False; start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        previous = checkpoint["metadata"]
        start_epoch = previous["best_epoch"] + 1
        best_loss, best_accuracy = previous["best_val_loss"], previous["best_val_accuracy"]
        for key, values in MEASURED_HISTORY.items(): history[key].extend(values[: previous["best_epoch"]])
        if start_epoch > args.fine_tune_epoch:
            unfreeze_layer4(model); fine_tuned = True
        print(f"Resumed from epoch {previous['best_epoch']} (val loss {best_loss:.4f}, accuracy {best_accuracy:.2f}%); continuing at epoch {start_epoch}.")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * (.1 if fine_tuned else 1.0), weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.3, patience=1)
    run_mode = "smoke" if any(x is not None for x in (args.max_train_samples, args.max_val_samples, args.max_test_samples)) else "full"
    for epoch in range(start_epoch, args.epochs + 1):
        if args.fine_tune_epoch and epoch == args.fine_tune_epoch and not fine_tuned:
            unfreeze_layer4(model); fine_tuned = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * .1, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.3, patience=1)
        train_loss, train_acc = run_epoch(model, data.train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, data.val_loader, criterion, device)
        for key, value in (("train_loss", train_loss), ("val_loss", val_loss), ("train_accuracy", train_acc), ("val_accuracy", val_acc)):
            history[key].append(value)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{args.epochs}\n  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%\n  Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")
        if val_loss < best_loss:
            best_loss, best_accuracy, stale = val_loss, val_acc, 0
            metadata = {"architecture": "resnet18", "class_names": data.class_names, "num_classes": len(data.class_names),
                        "image_size": args.image_size, "normalization_mean": IMAGENET_MEAN, "normalization_std": IMAGENET_STD,
                        "seed": args.seed, "val_fraction": args.val_fraction, "run_mode": run_mode,
                        "train_samples": len(data.train_indices), "val_samples": len(data.val_indices), "test_samples": len(data.test_indices),
                        "best_epoch": epoch, "best_val_loss": best_loss, "best_val_accuracy": best_accuracy}
            save_checkpoint(args.model_path, model, metadata)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs."); break
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training_curves(history, results_dir / "training_curves.png")
    print(f"Best validation loss: {best_loss:.4f}; accuracy: {best_accuracy:.2f}%\nSaved: {args.model_path}")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--model-path", default="models/best_model.pth"); parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--results-dir", default="results"); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--patience", type=int, default=3);    parser.add_argument("--fine-tune-epoch", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Continue from the best saved checkpoint instead of starting over.")
    parser.add_argument("--no-pretrained", action="store_true", help="Testing only: do not download ImageNet weights.")
    parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-val-samples", type=int); parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__": train(parse_args())
