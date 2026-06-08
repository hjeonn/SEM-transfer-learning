"""
Epoch Comparison Experiment
Epoch settings: [1, 5, 10, 20]
Metric: macro-averaged Recall and F1 (mean ± std over NUM_RUNS)

Output (results/)
  - epoch_comparison.csv         : macro mean±std per epoch setting
  - epoch_comparison_plot.png    : line plot for the paper
  - epoch_comparison_YYYYMMDD.mat: raw data
"""

import os
import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 0. Config

SEED           = 42
DATA_E1        = 'data/e1'
DATA_E2        = 'data/e2'
MAX_SAMPLES    = 201
TEST_PER_CLASS = 20
BATCH_SIZE     = 16
LR             = 1e-3
EPOCH_LIST     = [1, 5, 10, 20]   # epoch settings to compare
NUM_RUNS       = 5                 # 5 runs 
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR    = 'results'

print(f"Device: {DEVICE}")


# 1. Reproducibility

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


# 2. Dataset helpers 

def crop_bottom_tenth(img):
    w, h = img.size
    return img.crop((0, 0, w, h - round(h / 10)))

def collect_files(root):
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

class SEMDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, class_names, transform=None):
        self.files        = files
        self.labels       = labels
        self.class_names  = class_names
        self.label_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = crop_bottom_tenth(img)
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_idx[self.labels[idx]]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# 3. Model / train / eval helpers

def build_model(num_classes):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m.to(DEVICE)

def train_epochs(model, loader, optimizer, criterion, n_epochs):
    model.train()
    for _ in range(n_epochs):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()

def macro_f1(model, loader, num_classes):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())
    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
    TP = np.diag(cm).astype(float)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(f1.mean()), float(rec.mean())

def get_class_weights(labels, class_names):
    counts = np.array([labels.count(c) for c in class_names], dtype=float)
    w = 1.0 / counts
    w /= w.sum()
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


# 4. Data split (same as main_experiment.py)

rng = np.random.default_rng(SEED)

e1_files, e1_labels, class_names = collect_files(DATA_E1)
e2_files, e2_labels, _           = collect_files(DATA_E2)
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# E2: test split first
test_files, test_labels     = [], []
train2_files, train2_labels = [], []
for cls in class_names:
    pool   = [(f, l) for f, l in zip(e2_files, e2_labels) if l == cls]
    chosen = set(rng.choice(len(pool), size=min(TEST_PER_CLASS, len(pool)), replace=False))
    for i, (f, l) in enumerate(pool):
        (test_files if i in chosen else train2_files).append(f)
        (test_labels if i in chosen else train2_labels).append(l)

min_test = min(test_labels.count(c) for c in class_names)
bal_test_f, bal_test_l = [], []
for cls in class_names:
    pairs  = [(f, l) for f, l in zip(test_files, test_labels) if l == cls]
    chosen = rng.choice(len(pairs), size=min_test, replace=False)
    for i in chosen:
        bal_test_f.append(pairs[i][0])
        bal_test_l.append(pairs[i][1])

# Oversample
os_e1_f, os_e1_l   = oversample(e1_files,    e1_labels,     class_names, MAX_SAMPLES, rng)
os_e2_f, os_e2_l   = oversample(train2_files, train2_labels, class_names, MAX_SAMPLES, rng)

ds_e1   = SEMDataset(os_e1_f,   os_e1_l,   class_names, train_tf)
ds_e2os = SEMDataset(os_e2_f,   os_e2_l,   class_names, train_tf)
ds_e2wl = SEMDataset(train2_files, train2_labels, class_names, train_tf)
ds_test = SEMDataset(bal_test_f, bal_test_l, class_names, test_tf)

loader_e1   = DataLoader(ds_e1,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_e2os = DataLoader(ds_e2os, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_e2wl = DataLoader(ds_e2wl, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_weights_wl = get_class_weights(train2_labels, class_names)
criterion_ce = nn.CrossEntropyLoss()
criterion_wl = nn.CrossEntropyLoss(weight=class_weights_wl)


# 5. Experiment loop


results_f1  = {'Pre+FT(OS)': {e: [] for e in EPOCH_LIST},
               'Pre+FT(WL)': {e: [] for e in EPOCH_LIST}}
results_rec = {'Pre+FT(OS)': {e: [] for e in EPOCH_LIST},
               'Pre+FT(WL)': {e: [] for e in EPOCH_LIST}}

total_jobs = len(EPOCH_LIST) * NUM_RUNS
job = 0

for epochs in EPOCH_LIST:
    print(f"\n{'='*50}")
    print(f"  Epochs per stage: {epochs}")
    print(f"{'='*50}")

    for run in range(NUM_RUNS):
        set_seed(SEED + run)
        job += 1
        print(f"  [{job}/{total_jobs}] epochs={epochs}, run={run+1}/{NUM_RUNS}", end=' ... ')

        # Pre+FT(OS)
        m_os = build_model(num_classes)
        opt  = torch.optim.Adam(m_os.parameters(), lr=LR)
        train_epochs(m_os, loader_e1,   opt, criterion_ce, epochs)  # stage 1
        train_epochs(m_os, loader_e2os, opt, criterion_ce, epochs)  # stage 2
        f1_os, rec_os = macro_f1(m_os, loader_test, num_classes)
        results_f1['Pre+FT(OS)'][epochs].append(f1_os)
        results_rec['Pre+FT(OS)'][epochs].append(rec_os)

        # Pre+FT(WL)
        m_wl = build_model(num_classes)
        opt  = torch.optim.Adam(m_wl.parameters(), lr=LR)
        train_epochs(m_wl, loader_e1,   opt, criterion_ce, epochs)  # stage 1
        train_epochs(m_wl, loader_e2wl, opt, criterion_wl, epochs)  # stage 2
        f1_wl, rec_wl = macro_f1(m_wl, loader_test, num_classes)
        results_f1['Pre+FT(WL)'][epochs].append(f1_wl)
        results_rec['Pre+FT(WL)'][epochs].append(rec_wl)

        print(f"OS F1={f1_os:.3f}  WL F1={f1_wl:.3f}")


# 6. Aggregate & print

print(f"\n{'='*60}")
print(f"{'Epochs':>8} | {'Pre+FT(OS) F1':^20} | {'Pre+FT(WL) F1':^20}")
print(f"{'='*60}")
for e in EPOCH_LIST:
    os_vals = results_f1['Pre+FT(OS)'][e]
    wl_vals = results_f1['Pre+FT(WL)'][e]
    print(f"{e:>8} | {np.mean(os_vals):.3f} ± {np.std(os_vals):.3f}          "
          f"| {np.mean(wl_vals):.3f} ± {np.std(wl_vals):.3f}")
print(f"{'='*60}")


# 7. Save CSV

rows = []
for e in EPOCH_LIST:
    rows.append({
        'epochs':           e,
        'OS_F1_mean':       round(np.mean(results_f1['Pre+FT(OS)'][e]), 4),
        'OS_F1_std':        round(np.std(results_f1['Pre+FT(OS)'][e]),  4),
        'OS_Recall_mean':   round(np.mean(results_rec['Pre+FT(OS)'][e]), 4),
        'OS_Recall_std':    round(np.std(results_rec['Pre+FT(OS)'][e]),  4),
        'WL_F1_mean':       round(np.mean(results_f1['Pre+FT(WL)'][e]), 4),
        'WL_F1_std':        round(np.std(results_f1['Pre+FT(WL)'][e]),  4),
        'WL_Recall_mean':   round(np.mean(results_rec['Pre+FT(WL)'][e]), 4),
        'WL_Recall_std':    round(np.std(results_rec['Pre+FT(WL)'][e]),  4),
    })
df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, 'epoch_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV saved → {csv_path}")


# 8. Save .mat

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
mat_path  = os.path.join(RESULTS_DIR, f'epoch_comparison_{timestamp}.mat')
sio.savemat(mat_path, {
    'epoch_list':   EPOCH_LIST,
    'num_runs':     NUM_RUNS,
    'seed':         SEED,
    'OS_F1':        np.array([results_f1['Pre+FT(OS)'][e] for e in EPOCH_LIST]),
    'WL_F1':        np.array([results_f1['Pre+FT(WL)'][e] for e in EPOCH_LIST]),
    'OS_Recall':    np.array([results_rec['Pre+FT(OS)'][e] for e in EPOCH_LIST]),
    'WL_Recall':    np.array([results_rec['Pre+FT(WL)'][e] for e in EPOCH_LIST]),
})
print(f".mat saved  → {mat_path}")


# 9. Plot

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, metric, r_dict in zip(
        axes,
        ['Macro F1', 'Macro Recall'],
        [results_f1, results_rec]):

    for model, color, marker in [('Pre+FT(OS)', '#1a6faf', 'o'),
                                  ('Pre+FT(WL)', '#e05c2a', 's')]:
        means = [np.mean(r_dict[model][e]) for e in EPOCH_LIST]
        stds  = [np.std(r_dict[model][e])  for e in EPOCH_LIST]
        ax.errorbar(EPOCH_LIST, means, yerr=stds,
                    label=model, color=color, marker=marker,
                    linewidth=1.8, markersize=6, capsize=4)

    ax.set_xlabel('Epochs per stage', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'{metric} vs. Epochs', fontsize=12)
    ax.set_xticks(EPOCH_LIST)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.suptitle('Effect of Training Epochs on Pre+FT Performance', fontsize=13)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'epoch_comparison_plot.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved  → {plot_path}")
print("\nDone.")
