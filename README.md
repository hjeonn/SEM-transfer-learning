# SEM-Based NCM Cathode Classification via Two-Stage Transfer Learning

This repository contains the code for the paper:

**"Two-Stage Transfer Learning for SEM-Based NCM Cathode State Classification under Electrolyte Additive-Induced Domain Shift"**  
*Journal of Power Sources* (under review)

## Overview

This project implements a two-stage EfficientNet-B0 transfer learning framework for classifying NCM cathode materials (NCM333/622/811 × pristine/formation/100 cycles) from SEM images under domain shift induced by electrolyte additives.

- **Stage 1 (Pretraining):** EfficientNet-B0 trained on additive-free SEM images (E1)
- **Stage 2 (Fine-tuning):** Fine-tuned on additive-containing SEM images (E2)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- numpy, pandas, matplotlib

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib
```

## Data

Image data should be organized in class-specific subfolders:

```
/data/e1/<class_name>/
/data/e2/<class_name>/
```

Classes: `ncm333_pristine`, `ncm333_formation`, `ncm333_100cycles`, `ncm622_pristine`, `ncm622_formation`, `ncm622_100cycles`, `ncm811_pristine`, `ncm811_formation`, `ncm811_100cycles`

## Usage

**Main experiment (5 models, 10 independent runs):**

```bash
python main_experiment.py
```

Select specific models with `--models` flag:

```bash
python main_experiment.py --models 1 3 5
```

| Flag | Model |
|------|-------|
| 1 | Pre+FT(OS) |
| 2 | Pre+FT(WL) |
| 3 | FT(OS) |
| 4 | Pre only |
| 5 | Scratch+FT(OS) |

**Epoch sensitivity analysis:**

```bash
python epoch_comparison.py
```

## Note

This repository supersedes the previous MATLAB implementation presented at AI4Mat-NeurIPS-2025.
