"""Run top-k inference on one image using the saved best model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from utils import get_device, load_checkpoint


def predict(image_path: str, model_path: str, top_k: int = 3, device_name: str = "auto") -> list[tuple[str, float]]:
    """Return ranked label/confidence pairs, raising clear input errors."""
    path = Path(image_path)
    if not path.is_file(): raise FileNotFoundError(f"Image not found: {path}")
    device = get_device(device_name); model, metadata = load_checkpoint(model_path, device)
    transform = transforms.Compose([transforms.Resize((metadata["image_size"], metadata["image_size"])), transforms.ToTensor(),
                                    transforms.Normalize(metadata["normalization_mean"], metadata["normalization_std"])])
    try:
        with Image.open(path) as image: tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid or unreadable image: {path}") from exc
    with torch.inference_mode(): probabilities = model(tensor).softmax(1)[0]
    values, indices = probabilities.topk(min(top_k, len(metadata["class_names"])))
    return [(metadata["class_names"][i], float(v)) for v, i in zip(values.cpu(), indices.cpu())]


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", default="models/best_model.pth"); parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"); args = parser.parse_args()
    try: results = predict(args.image, args.model_path, args.top_k, args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc: parser.error(str(exc))
    print(f"Predicted class: {results[0][0]}\nConfidence: {results[0][1] * 100:.2f}%\nTop predictions:")
    for rank, (label, confidence) in enumerate(results, 1): print(f"{rank}. {label} — {confidence * 100:.2f}%")


if __name__ == "__main__": main()
