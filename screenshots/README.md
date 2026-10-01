# Submission Screenshot Guide

Capture only real outputs after completing the full run. Recommended screenshots:

All ten captures below were generated from real artifacts. `01`–`03`, `06`, `07`, and `10` render from `logs/`, `results/`, and live command output — regenerate them any time with:

```bash
.test-venv/Scripts/python.exe screenshots/make_screenshots.py
```

`04` and `05` are copies of `results/training_curves.png` and `results/confusion_matrix.png`; `08` and `09` are headless-Chrome captures of the live repository pages.

1. **Project structure** — terminal `tree -a -I 'venv|raw|__pycache__'` output.
2. **Dataset exploration** — notebook sample grid and class-distribution chart.
3. **Training output** — terminal showing multiple epochs and best validation checkpoint.
4. **Training curves** — `results/training_curves.png`.
5. **Confusion matrix** — `results/confusion_matrix.png`.
6. **Classification report** — readable section of `results/classification_report.txt`.
7. **Inference result** — command and top-3 output for a real image.
8. **GitHub repository** — repository landing page with file tree.
9. **README** — rendered overview, commands, and latest-results block.
10. **Running workflow** — notebook or terminal demonstrating a successful command.

Use consistent window size, hide private information, avoid cropped labels, and do not submit smoke-test metrics as final results.
