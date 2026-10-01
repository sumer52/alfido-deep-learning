# CIFAR-10 Image Classification with Transfer Learning

## Overview
A reproducible PyTorch project that adapts an ImageNet-pretrained ResNet18 to classify CIFAR-10 images. It includes a leakage-safe data pipeline, augmentation, training/validation, best-checkpoint saving, evaluation, inference, a runnable notebook, and responsible-AI documentation.

## Objective
Build an internship-level image classifier using correct transfer learning and report performance without fabricated results.

## Dataset
**CIFAR-10**, University of Toronto: 60,000 RGB 32×32 images, 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The official 50,000-image training partition is split deterministically into 45,000 train and 5,000 validation images; 10,000 official test images are preserved. It is small, established, automatically downloadable through torchvision, and diverse enough to demonstrate transfer learning.

## Features
- ImageNet-pretrained ResNet18 with a replaced 10-class head
- Frozen-backbone head training, with optional final-block fine-tuning
- Train-only random crop, horizontal flip, and rotation
- ImageNet normalization and resizing (the reported run uses 128×128 input, chosen for CPU feasibility; CIFAR-10 is natively 32×32 and the value is stored in the checkpoint metadata)
- AdamW, learning-rate reduction, early stopping, best-model checkpointing
- Accuracy, macro/weighted precision, recall, and F1
- Training curves, confusion matrix, classification report, predictions CSV
- Top-3 single-image inference with graceful errors
- Automatic insertion of evaluated metrics into this README and `REPORT.md`

## Technologies
Python 3, PyTorch, torchvision, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pillow, and Jupyter Notebook.

## Project Structure
```text
alfido-deep-learning/
├── data/README.md
├── notebooks/deep_learning_image_classification.ipynb
├── src/{train.py,evaluate.py,inference.py,dataset.py,utils.py}
├── models/.gitkeep
├── results/
├── screenshots/README.md
├── requirements.txt
├── README.md
├── REPORT.md
├── .gitignore
└── LICENSE
```

## Installation
Run commands from the repository root.

### Windows
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter with `jupyter notebook`.

## Training
Full recommended run (the configuration used for the reported results):
```bash
python src/train.py --epochs 10 --batch-size 128 --image-size 128 --num-workers 4 --fine-tune-epoch 4
```
CPU smoke test:
```bash
python src/train.py --epochs 1 --batch-size 16 --image-size 64 --max-train-samples 128 --max-val-samples 64 --max-test-samples 64 --fine-tune-epoch 0
```
Training downloads CIFAR-10 and pretrained weights automatically. Augmentation improves robustness by exposing the model to plausible image variation; it is applied only to the training subset.

## Evaluation
```bash
python src/evaluate.py
```
For the smoke checkpoint, use the matching bounded test: `python src/evaluate.py --max-test-samples 64`.

**Metric interpretation:** accuracy is the overall correct fraction; precision measures prediction correctness; recall measures class recovery; F1 balances precision and recall. Macro averages weight every class equally; weighted averages account for support.

## Inference
```bash
python src/inference.py --image path/to/image.jpg
```
The image is converted to RGB, resized and normalized using checkpoint metadata, then ranked with softmax probabilities.

## Results
<!-- AUTO_RESULTS_START -->
**Latest evaluated run:** `full` (10,000 test images)  
Accuracy: **0.9105** · Macro F1: **0.9103** · Weighted F1: **0.9103**  
See `results/metrics.json` and `results/classification_report.txt`. These are the final full-run test metrics over all 10,000 official test images.
<!-- AUTO_RESULTS_END -->

Artifacts: `models/best_model.pth`, `models/model_metadata.json`, `results/training_curves.png`, `results/confusion_matrix.png`, `results/classification_report.txt`, `results/metrics.json`, and `results/predictions.csv`.

## Model Architecture
Input → train/evaluation preprocessing → ImageNet-pretrained ResNet18 feature extractor → replaced 512-to-10 fully connected layer → logits → softmax only during inference. Cross-entropy consumes logits directly during training.

## Responsible AI
- **Bias:** CIFAR-10 is curated and approximately class-balanced, but its tiny images and source distribution do not represent all real settings.
- **Fairness:** aggregate accuracy can hide class-level gaps; inspect per-class recall/F1 and the confusion matrix.
- **Privacy:** CIFAR-10 is public benchmark data. Real deployments involving people require consent, purpose limitation, retention controls, and privacy review.
- **Reliability:** confidence is not certainty. Distribution shift, blur, cropping, and non-CIFAR imagery can cause errors.
- **Appropriate use:** suitable for education and prototyping; inappropriate for safety-critical, surveillance, identity, or high-stakes decisions.

## Reproducibility
Default seed is 42; splits, shuffling, and framework RNGs are seeded. Package versions are pinned. Exact reproducibility can still vary across hardware and low-level kernels. Checkpoint metadata stores architecture, labels, normalization, image size, split seed, and run mode.

## Limitations
CIFAR-10 resolution is low; resizing does not add detail. The reported run used 128×128 inputs instead of the default 224×224 for CPU-training feasibility — a deliberate speed/quality trade-off recorded in `models/model_metadata.json`. ImageNet-to-CIFAR transfer has domain mismatch. Softmax confidence is not calibrated. A reduced smoke run validates plumbing but not final quality.

## Future Improvements
Stratified split auditing, stronger calibration, systematic hyperparameter tuning, cross-validation, gradual unfreezing, EfficientNet/MobileNet comparison, and deployment tests on out-of-distribution images.

## Model Hosting
`best_model.pth` is ignored by Git. If it is too large, upload it to Google Drive, set sharing to “Anyone with the link,” and replace **MODEL_URL_TO_BE_ADDED_AFTER_UPLOAD** here: `MODEL_URL_TO_BE_ADDED_AFTER_UPLOAD`.

## Author
The King — Alfido Tech Internship Task 2.
