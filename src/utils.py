"""Shared model, reproducibility, plotting, checkpoint, and reporting utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/mps to an available device."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return device
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build ImageNet-pretrained ResNet18 with a task-specific classifier."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters(): parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_layer4(model: nn.Module) -> None:
    """Unfreeze the final residual block and classifier for light fine-tuning."""
    for parameter in model.layer4.parameters(): parameter.requires_grad = True
    for parameter in model.fc.parameters(): parameter.requires_grad = True


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    """Save weights and inference metadata together."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    meta_path = path.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved ResNet18 checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    model = build_model(len(metadata["class_names"]), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"]); model.to(device); model.eval()
    return model, metadata


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    """Save readable train/validation loss and accuracy panels."""
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Training")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=180, bbox_inches="tight"); plt.close(fig)


def update_managed_results(file_path: str | Path, text: str) -> None:
    """Replace the auto-results block in README/REPORT without touching other prose."""
    path = Path(file_path)
    if not path.exists(): return
    content = path.read_text(encoding="utf-8")
    start, end = "<!-- AUTO_RESULTS_START -->", "<!-- AUTO_RESULTS_END -->"
    if start in content and end in content:
        before, remainder = content.split(start, 1)
        _, after = remainder.split(end, 1)
        path.write_text(f"{before}{start}\n{text.rstrip()}\n{end}{after}", encoding="utf-8")
