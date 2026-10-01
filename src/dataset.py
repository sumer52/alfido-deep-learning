"""CIFAR-10 data loading, deterministic splitting, and preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DataBundle:
    """Train, validation, and test loaders plus their deterministic indices."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return augmentation-only-for-training and deterministic evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def _limit(indices: Sequence[int], maximum: int | None) -> list[int]:
    values = list(map(int, indices))
    return values if maximum is None else values[: min(maximum, len(values))]


def create_dataloaders(
    data_dir: str | Path = "data/raw",
    batch_size: int = 64,
    image_size: int = 224,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DataBundle:
    """Download CIFAR-10 and create leakage-free deterministic loaders."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    root = Path(data_dir)
    train_tf, eval_tf = build_transforms(image_size)
    train_aug = datasets.CIFAR10(root=root, train=True, download=download, transform=train_tf)
    train_eval = datasets.CIFAR10(root=root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root=root, train=False, download=download, transform=eval_tf)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(train_aug))
    val_size = int(len(train_aug) * val_fraction)
    val_indices = _limit(permutation[:val_size], max_val_samples)
    train_indices = _limit(permutation[val_size:], max_train_samples)
    test_indices = _limit(range(len(test_set)), max_test_samples)

    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(Subset(train_aug, train_indices), shuffle=True, generator=generator, **common)
    val_loader = DataLoader(Subset(train_eval, val_indices), shuffle=False, **common)
    test_loader = DataLoader(Subset(test_set, test_indices), shuffle=False, **common)
    return DataBundle(train_loader, val_loader, test_loader, list(train_aug.classes), train_indices, val_indices, test_indices)
