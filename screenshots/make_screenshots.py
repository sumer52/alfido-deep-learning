"""Regenerate the terminal-style submission screenshots from real artifacts only.

Run from the repository root:
    .test-venv/Scripts/python.exe screenshots/make_screenshots.py

Every rendered line of output comes from real sources: logs/full_train.log,
logs/resume_train.log, logs/evaluate.log, results/*, a live inference run,
`git ls-files`, and the downloaded CIFAR-10 data. Nothing is transcribed by hand.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
VENV_PY = str(ROOT / ".test-venv" / "Scripts" / "python.exe")
FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
LINE_H = 22
PAD = 18
FG = (204, 204, 204)
PROMPT = (78, 201, 78)
BG = (12, 12, 12)

PROMPT_PREFIX = "PS C:\\Users\\sumer\\Downloads\\alfido-deep-learning> "


def render_terminal(lines: list[tuple[str, tuple]], out_path: Path) -> None:
    """Render prompt/output pairs as a Windows-Terminal-style PNG."""
    width = int(max(FONT.getlength(text) for text, _ in lines)) + 2 * PAD
    height = len(lines) * LINE_H + 2 * PAD
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    for index, (text, color) in enumerate(lines):
        draw.text((PAD, PAD + index * LINE_H), text, font=FONT, fill=color)
    image.save(out_path)
    print(f"wrote {out_path.relative_to(ROOT)} ({width}x{height})")


def command_line(command: str) -> tuple[str, tuple]:
    return PROMPT_PREFIX + command, PROMPT


def output_lines(text: str) -> list[tuple[str, tuple]]:
    return [(line, FG) for line in text.strip().splitlines()]


def screenshot_01_project_structure() -> None:
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.splitlines()
    tree: dict = {}
    for path in sorted(tracked):
        node = tree
        for part in path.split("/")[:-1]:
            node = node.setdefault(part + "/", {})
        node[path.split("/")[-1]] = None

    lines = ["alfido-deep-learning/", ""]
    entries = sorted(tree.items(), key=lambda kv: (0 if kv[0].endswith("/") else 1, kv[0]))

    def walk(node: dict, prefix: str = "") -> None:
        items = sorted(node.items(), key=lambda kv: (0 if kv[0].endswith("/") else 1, kv[0]))
        for index, (name, child) in enumerate(items):
            last = index == len(items) - 1
            lines.append(prefix + ("└── " if last else "├── ") + name)
            if isinstance(child, dict):
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree)
    rendered = [command_line("git ls-files  # project structure (tracked files)"), *output_lines("\n".join(lines))]
    render_terminal(rendered, OUT / "01_project_structure.png")


def screenshot_02_dataset_exploration() -> None:
    train = datasets.CIFAR10(root=str(ROOT / "data/raw"), train=True, download=False)
    counts = np.bincount(np.asarray(train.targets))
    rng = np.random.default_rng(42)
    picks = rng.choice(len(train), 12, replace=False)

    fig = plt.figure(figsize=(15, 5.4))
    grid = fig.add_gridspec(3, 5, width_ratios=[1.35, 1, 1, 1, 1], hspace=0.35, wspace=0.25)
    bar_ax = fig.add_subplot(grid[:, 0])
    sns.barplot(x=train.classes, y=counts, color="#2783DE", ax=bar_ax)
    bar_ax.set(title="Class distribution — official training images", xlabel="Class", ylabel="Number of images")
    bar_ax.tick_params(axis="x", rotation=45)
    for cell, index in zip([(r, c) for r in range(3) for c in range(1, 5)], picks):
        ax = fig.add_subplot(grid[cell[0], cell[1]])
        image, label = train[int(index)]
        ax.imshow(image)
        ax.set_title(train.classes[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("CIFAR-10 dataset exploration — 50,000 train images split 45,000/5,000; 10,000 test images held out")
    out = OUT / "02_dataset_exploration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def screenshot_03_training_output() -> None:
    keep = re.compile(r"^(Epoch \d|  Train Loss|  Validation Loss|Resumed from|Best validation)")
    first_leg = [line for line in (ROOT / "logs/full_train.log").read_text(errors="replace").splitlines() if keep.match(line)]
    second_leg = [line for line in (ROOT / "logs/resume_train.log").read_text(errors="replace").splitlines() if keep.match(line)]
    lines = [
        command_line("python src/train.py --epochs 10 --batch-size 128 --image-size 128 --num-workers 4 --fine-tune-epoch 4"),
        *output_lines("\n".join(first_leg)),
        command_line("python src/train.py --epochs 10 --batch-size 128 --image-size 128 --num-workers 4 --fine-tune-epoch 4 --resume"),
        *output_lines("\n".join(second_leg)),
    ]
    render_terminal(lines, OUT / "03_training_output.png")


def screenshot_06_classification_report() -> None:
    report = (ROOT / "results/classification_report.txt").read_text(encoding="utf-8")
    lines = [command_line("cat results/classification_report.txt"), *output_lines(report)]
    render_terminal(lines, OUT / "06_classification_report.png")


def screenshot_07_inference() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [VENV_PY, "src/inference.py", "--image", "demo_horse.jpg", "--top-k", "3"],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env,
    )
    lines = [command_line("python src/inference.py --image demo_horse.jpg --top-k 3"), *output_lines(result.stdout)]
    render_terminal(lines, OUT / "07_inference_result.png")


def screenshot_10_running_workflow() -> None:
    evaluation = (ROOT / "logs/evaluate.log").read_text(errors="replace").strip().splitlines()
    head = "\n".join(evaluation[: evaluation.index("Classification report")]).strip()
    lines = [command_line("python src/evaluate.py"), *output_lines(head)]
    render_terminal(lines, OUT / "10_running_workflow.png")


def copy_artifact(source: str, target: str) -> None:
    destination = OUT / target
    destination.write_bytes((ROOT / source).read_bytes())
    print(f"wrote {destination.relative_to(ROOT)} (copied from {source})")


if __name__ == "__main__":
    os.chdir(ROOT)
    screenshot_01_project_structure()
    screenshot_02_dataset_exploration()
    screenshot_03_training_output()
    copy_artifact("results/training_curves.png", "04_training_curves.png")
    copy_artifact("results/confusion_matrix.png", "05_confusion_matrix.png")
    screenshot_06_classification_report()
    screenshot_07_inference()
    screenshot_10_running_workflow()
    print("Screenshots 08 and 09 (GitHub pages) are captured from the live repository page.")
