"""
Models
  1. Pre+FT(OS)  : pretrain on E1 → fine-tune on E2 (oversampling)
  2. Pre+FT(WL)  : pretrain on E1 → fine-tune on E2 (weighted loss)
  3. FT(OS)      : train on E2 only               (oversampling)
  4. Pre only    : pretrain on E1 → test on E2     (no fine-tuning)

Output (results/)
  - results_YYYYMMDD_HHMM.mat   : all raw data
  - classwise_results.csv       : mean ± std per class per model
  - confmat_mean_modelN.png     : mean confusion matrix figures
"""

import os
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
SEED        = 42
DATA_E1     = 'data/e1'
DATA_E2     = 'data/e2'
MAX_SAMPLES = 201          # oversampling target (matches MATLAB)
TEST_PER_CLASS = 20        # fixed test size per class
BATCH_SIZE  = 16
LR          = 1e-3
EPOCHS      = 1            # epochs per stage

# ── CLI arguments ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', type=int,
                    default=[1, 2, 3, 4, 5],
                    help='Models to run: 1=Pre+FT(OS) 2=Pre+FT(WL) 3=FT(OS) 4=Pre-only 5=Scratch+FT(OS)')
args = parser.parse_args()
MODELS_TO_RUN = set(args.models)
print(f"Running models: {sorted(MODELS_TO_RUN)}")
NUM_RUNS    = 10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'

# ─────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'confmat_images'), exist_ok=True)

# ─────────────────────────────────────────────
# 2. Dataset helpers
# ─────────────────────────────────────────────
def crop_bottom_tenth(img: Image.Image) -> Image.Image:
    """Remove bottom 1/10 of image (SEM scale bar removal)."""
    w, h = img.size
    crop_h = round(h / 10)
    return img.crop((0, 0, w, h - crop_h))


def collect_files(root: str):
    """Return (file_paths, labels, class_names) from folder structure."""
    class_names = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    files, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                files.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return files, labels, class_names


def oversample(files, labels, class_names, max_samples, rng):
    """Replicate minority-class samples until each class has max_samples."""
    new_files, new_labels = list(files), list(labels)
    for cls in class_names:
        cls_files = [f for f, l in zip(files, labels) if l == cls]
        n = len(cls_files)
        while n < max_samples:
            add_n = min(max_samples - n, len(cls_files))
            chosen = rng.choice(cls_files, size=add_n, replace=False).tolist()
            new_files.extend(chosen)
            new_labels.extend([cls] * add_n)
            n += add_n
    return new_files, new_labels


class SEMDataset(Dataset):
    def __init__(self, files, labels, class_names, transform=None):
        self.files       = files
        self.labels      = labels
        self.class_names = class_names
        self.label_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform   = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = crop_bottom_tenth(img)
        if self.transform:
            img = self.transform(img)
        label = self.label_to_idx[self.labels[idx]]
        return img, label


# ─────────────────────────────────────────────
# 3. Transforms
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 4. Model builder
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with ImageNet weights, replaced classifier head."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def build_model_scratch(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with random initialisation (no ImageNet weights)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# 5. Train / eval
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()

def train_epochs(model, loader, optimizer, criterion, n_epochs):
    for _ in range(n_epochs):
        train_one_epoch(model, loader, optimizer, criterion)


def evaluate(model, loader, class_names):
    """Return confusion matrix (numpy)."""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(lbls.numpy())
    return confusion_matrix(all_trues, all_preds,
                            labels=list(range(len(class_names))))


def compute_metrics(cm):
    """Per-class Recall and F1 from confusion matrix."""
    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    recall = TP / (TP + FN + 1e-9)
    prec   = TP / (TP + FP + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)
    return recall, f1


def get_class_weights(labels, class_names):
    """Median frequency balancing: weight_c = median(counts) / count_c
    Middle-sized classes get weight ~1.0, keeping loss scale
    close to standard CE (avoids gradient shrinkage from sum normalisation).
    """
    counts  = np.array([labels.count(c) for c in class_names], dtype=float)
    weights = np.median(counts) / counts
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ─────────────────────────────────────────────
# 6. Data split (fixed across runs — split once)
# ─────────────────────────────────────────────
rng = np.random.default_rng(SEED)

e1_files, e1_labels, class_names = collect_files(DATA_E1)
e2_files, e2_labels, _           = collect_files(DATA_E2)

num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# --- E2: split test first, then oversample train ---
test_files, test_labels   = [], []
train2_files, train2_labels = [], []

for cls in class_names:
    cls_pool = [(f, l) for f, l in zip(e2_files, e2_labels) if l == cls]
    chosen   = rng.choice(len(cls_pool),
                          size=min(TEST_PER_CLASS, len(cls_pool)),
                          replace=False)
    chosen_set = set(chosen)
    for i, (f, l) in enumerate(cls_pool):
        if i in chosen_set:
            test_files.append(f);  test_labels.append(l)
        else:
            train2_files.append(f); train2_labels.append(l)

# Balance test set to minimum class count
min_test = min(test_labels.count(c) for c in class_names)
balanced_test_f, balanced_test_l = [], []
for cls in class_names:
    pairs = [(f, l) for f, l in zip(test_files, test_labels) if l == cls]
    chosen = rng.choice(len(pairs), size=min_test, replace=False)
    for i in chosen:
        balanced_test_f.append(pairs[i][0])
        balanced_test_l.append(pairs[i][1])

print(f"Test set: {len(balanced_test_f)} images "
      f"({min_test} per class × {num_classes} classes)")

# --- Oversample train sets ---
os_e1_files, os_e1_labels   = oversample(e1_files,    e1_labels,    class_names, MAX_SAMPLES, rng)
os_e2_files, os_e2_labels   = oversample(train2_files, train2_labels, class_names, MAX_SAMPLES, rng)

print(f"E1 train (after OS): {len(os_e1_files)} | "
      f"E2 train (after OS): {len(os_e2_files)}")

# ─────────────────────────────────────────────
# 7. Datasets & loaders
# ─────────────────────────────────────────────
ds_e1_train  = SEMDataset(os_e1_files,     os_e1_labels,     class_names, train_tf)
ds_e2_train  = SEMDataset(os_e2_files,     os_e2_labels,     class_names, train_tf)
ds_test      = SEMDataset(balanced_test_f, balanced_test_l,  class_names, test_tf)

loader_e1    = DataLoader(ds_e1_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_e2_os = DataLoader(ds_e2_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_test  = DataLoader(ds_test,     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Weighted loss needs class weights from the *unaugmented* E2 train set
class_weights_wl = get_class_weights(train2_labels, class_names)

# ─────────────────────────────────────────────
# 8. Main experiment loop
# ─────────────────────────────────────────────
# Storage: shape (num_runs, num_classes)
runs_recall = {m: np.zeros((NUM_RUNS, num_classes)) for m in range(1, 6)}
runs_f1     = {m: np.zeros((NUM_RUNS, num_classes)) for m in range(1, 6)}
all_cm      = {m: np.zeros((num_classes, num_classes, NUM_RUNS)) for m in range(1, 6)}

criterion_ce = nn.CrossEntropyLoss()
criterion_wl = nn.CrossEntropyLoss(weight=class_weights_wl)

for run in range(NUM_RUNS):
    set_seed(SEED + run)          # different seed each run, but reproducible
    print(f"\n===== Run {run+1}/{NUM_RUNS} =====")

    # ── Model 1: Pre+FT(OS) ──────────────────
    if 1 in MODELS_TO_RUN:
        m1 = build_model(num_classes)
        opt = torch.optim.Adam(m1.parameters(), lr=LR)
        train_epochs(m1, loader_e1,    opt, criterion_ce, EPOCHS)
        train_epochs(m1, loader_e2_os, opt, criterion_ce, EPOCHS)
        cm1 = evaluate(m1, loader_test, class_names)
        all_cm[1][:, :, run] = cm1
        runs_recall[1][run], runs_f1[1][run] = compute_metrics(cm1)

    # ── Model 2: Pre+FT(WL) ──────────────────
    if 2 in MODELS_TO_RUN:
        ds_e2_wl    = SEMDataset(train2_files, train2_labels, class_names, train_tf)
        loader_e2_wl = DataLoader(ds_e2_wl, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        m2 = build_model(num_classes)
        opt = torch.optim.Adam(m2.parameters(), lr=LR)
        opt2 = torch.optim.Adam(m2.parameters(), lr=LR)
        train_epochs(m2, loader_e1,    opt,  criterion_ce, EPOCHS)
        opt2 = torch.optim.Adam(m2.parameters(), lr=LR)
        train_epochs(m2, loader_e2_wl, opt2, criterion_wl, EPOCHS)
        cm2 = evaluate(m2, loader_test, class_names)
        all_cm[2][:, :, run] = cm2
        runs_recall[2][run], runs_f1[2][run] = compute_metrics(cm2)

    # ── Model 3: FT(OS) ──────────────────────
    if 3 in MODELS_TO_RUN:
        m3 = build_model(num_classes)
        opt = torch.optim.Adam(m3.parameters(), lr=LR)
        train_epochs(m3, loader_e2_os, opt, criterion_ce, EPOCHS)
        cm3 = evaluate(m3, loader_test, class_names)
        all_cm[3][:, :, run] = cm3
        runs_recall[3][run], runs_f1[3][run] = compute_metrics(cm3)

    # ── Model 4: Pre only ────────────────────
    if 4 in MODELS_TO_RUN:
        m4 = build_model(num_classes)
        opt = torch.optim.Adam(m4.parameters(), lr=LR)
        train_epochs(m4, loader_e1, opt, criterion_ce, EPOCHS)
        cm4 = evaluate(m4, loader_test, class_names)
        all_cm[4][:, :, run] = cm4
        runs_recall[4][run], runs_f1[4][run] = compute_metrics(cm4)

    # ── Model 5: Scratch+FT(OS) ──────────────
    if 5 in MODELS_TO_RUN:
        m5 = build_model_scratch(num_classes)
        opt = torch.optim.Adam(m5.parameters(), lr=LR)
        train_epochs(m5, loader_e2_os, opt, criterion_ce, EPOCHS)
        cm5 = evaluate(m5, loader_test, class_names)
        all_cm[5][:, :, run] = cm5
        runs_recall[5][run], runs_f1[5][run] = compute_metrics(cm5)

    run_log = f"  Run {run+1}"
    for m in sorted(MODELS_TO_RUN):
        run_log += f" | M{m}: {runs_f1[m][run].mean():.3f}"
    print(run_log)

# ─────────────────────────────────────────────
# 9. Aggregate results
# ─────────────────────────────────────────────
mean_recall = {m: runs_recall[m].mean(axis=0) for m in range(1, 6)}
std_recall  = {m: runs_recall[m].std(axis=0)  for m in range(1, 6)}
mean_f1     = {m: runs_f1[m].mean(axis=0)     for m in range(1, 6)}
std_f1      = {m: runs_f1[m].std(axis=0)      for m in range(1, 6)}
mean_cm     = {m: all_cm[m].mean(axis=2)      for m in range(1, 6)}

model_names = {
    1: 'Pre+FT(OS)',
    2: 'Pre+FT(WL)',
    3: 'FT(OS)',
    4: 'Pre only',
    5: 'Scratch+FT(OS)',
}

# ─────────────────────────────────────────────
# 10. Print table
# ─────────────────────────────────────────────
header = f"{'Class':<22}"
for m in range(1, 6):
    header += f"| {model_names[m]:^22} "
print('\n' + '=' * 115)
print(header)
sub = f"{'':22}"
for _ in range(5):
    sub += f"|{'Recall':^12}{'F1':^12}"
print(sub)
print('=' * 115)

for i, cls in enumerate(class_names):
    row = f"{cls:<22}"
    for m in range(1, 6):
        r_str = f"{mean_recall[m][i]:.3f}±{std_recall[m][i]:.3f}"
        f_str = f"{mean_f1[m][i]:.3f}±{std_f1[m][i]:.3f}"
        row += f"|{r_str:^12}{f_str:^12}"
    print(row)

print('=' * 115)
macro_row = f"{'Macro avg':<22}"
for m in range(1, 6):
    r_str = f"{mean_recall[m].mean():.3f}±{std_recall[m].mean():.3f}"
    f_str = f"{mean_f1[m].mean():.3f}±{std_f1[m].mean():.3f}"
    macro_row += f"|{r_str:^12}{f_str:^12}"
print(macro_row)
print('=' * 115)

# ─────────────────────────────────────────────
# 11. Save CSV
# ─────────────────────────────────────────────
rows = []
for i, cls in enumerate(class_names):
    row = {'class': cls}
    for m in range(1, 6):
        row[f'{model_names[m]}_recall'] = f"{mean_recall[m][i]:.4f}±{std_recall[m][i]:.4f}"
        row[f'{model_names[m]}_f1']     = f"{mean_f1[m][i]:.4f}±{std_f1[m][i]:.4f}"
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, 'classwise_results.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV saved → {csv_path}")

# ─────────────────────────────────────────────
# 12. Save .mat
# ─────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
mat_path  = os.path.join(RESULTS_DIR, f'results_{timestamp}.mat')
sio.savemat(mat_path, {
    'class_names':  class_names,
    'model_names':  list(model_names.values()),
    'runs_recall':  np.stack([runs_recall[m] for m in range(1, 6)]),
    'runs_f1':      np.stack([runs_f1[m]     for m in range(1, 6)]),
    'mean_recall':  np.stack([mean_recall[m] for m in range(1, 6)]),
    'std_recall':   np.stack([std_recall[m]  for m in range(1, 6)]),
    'mean_f1':      np.stack([mean_f1[m]     for m in range(1, 6)]),
    'std_f1':       np.stack([std_f1[m]      for m in range(1, 6)]),
    'mean_confmat': np.stack([mean_cm[m]     for m in range(1, 6)]),
    'all_confmat':  np.stack([all_cm[m]      for m in range(1, 6)]),
    'seed':         SEED,
    'num_runs':     NUM_RUNS,
    'epochs':       EPOCHS,
})
print(f".mat saved  → {mat_path}")

# ─────────────────────────────────────────────
# 13. Save mean confusion matrix figures
# ─────────────────────────────────────────────
for m in range(1, 6):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_int = np.round(mean_cm[m]).astype(int)
    sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Mean Confusion Matrix — {model_names[m]} ({NUM_RUNS} runs)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'confmat_mean_model{m}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Confmat saved → {fig_path}")

print("\nDone.")
