#!/usr/bin/env python3
import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# Dataset
# =========================

class EEGGCDataset(Dataset):
    def __init__(self, file_paths, subject_ids, label_map,
                 log_transform=True, eps=1e-6, diag_zero=True):
        """
        file_paths: list of .npy paths
        subject_ids: list of subject_id strings (same length as file_paths)
        label_map: dict {subject_id: 0 or 1}  (0=healthy, 1=MDD)
        """
        assert len(file_paths) == len(subject_ids)
        self.file_paths = file_paths
        self.subject_ids = subject_ids
        self.label_map = label_map
        self.log_transform = log_transform
        self.eps = eps
        self.diag_zero = diag_zero

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        subj = self.subject_ids[idx]
        y = self.label_map[subj]  # 0 or 1

        gc = np.load(path).astype(np.float32)  # (19, 19) or (C, H, W) if you change later

        # Assume gc is (19, 19) of raw p-values
        if self.log_transform:
            gc = -np.log10(gc + self.eps)
        if self.diag_zero:
            np.fill_diagonal(gc, 0.0)

        # Add channel dimension -> (1, H, W)
        gc = np.expand_dims(gc, axis=0)

        x = torch.from_numpy(gc)                  # float32
        y = torch.tensor(float(y), dtype=torch.float32)

        return x, y, subj


# =========================
# Model
# =========================

class GCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 19 -> 9

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 9 -> 4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Make everything fixed-size
            nn.AdaptiveAvgPool2d((2, 2))           # -> (128, 2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # 128 * 2 * 2 = 512
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)                      # binary logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)                       # (B,)


# =========================
# Utility: file parsing & splitting
# =========================

def infer_label_from_filename(fname):
    """Return 0 for healthy, 1 for MDD based on filename prefix."""
    base = os.path.basename(fname)
    if base.startswith("H_"):
        return 0
    elif base.startswith("MDD_"):
        return 1
    else:
        raise ValueError(f"Cannot infer label from filename: {base}")


def parse_subject_id_from_filename(fname):
    """
    Heuristic: use the first two whitespace-separated tokens as subject ID.
    Example:
      'H_6921143_H S15 EO_seg_21_GCmatrix.npy' -> 'H_6921143_H S15'
      'MDD_MDD S17 EC_seg_18_GCmatrix.npy' -> 'MDD_MDD S17'
    Adjust this function if your naming scheme differs.
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    tokens = name.split()
    if len(tokens) == 1:
        # fallback: use the whole thing
        return tokens[0]
    return " ".join(tokens[:2])


def build_subject_index(file_paths):
    """
    Group files by subject and build label map.
    Returns:
      subject_to_files: dict {subject_id: [path1, path2, ...]}
      subject_to_label: dict {subject_id: 0 or 1}
    """
    subject_to_files = defaultdict(list)
    subject_to_label = {}

    for path in file_paths:
        subj = parse_subject_id_from_filename(path)
        label = infer_label_from_filename(path)
        subject_to_files[subj].append(path)

        if subj in subject_to_label:
            # sanity check: same label within subject
            if subject_to_label[subj] != label:
                raise ValueError(f"Conflicting labels for subject {subj}")
        else:
            subject_to_label[subj] = label

    return subject_to_files, subject_to_label


def subject_wise_split(subject_ids, val_frac=0.2, test_frac=0.2, seed=42):
    """
    Split subject IDs into train/val/test lists.
    """
    rng = np.random.RandomState(seed)
    subject_ids = np.array(subject_ids)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_ids = subject_ids[:n_test]
    val_ids = subject_ids[n_test:n_test + n_val]
    train_ids = subject_ids[n_test + n_val:]

    return list(train_ids), list(val_ids), list(test_ids)


def make_datasets(file_paths, val_frac, test_frac, seed):
    subject_to_files, subject_to_label = build_subject_index(file_paths)
    all_subjects = list(subject_to_files.keys())

    train_subj, val_subj, test_subj = subject_wise_split(
        all_subjects, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    def expand(subj_list):
        paths = []
        subj_ids = []
        for s in subj_list:
            s_files = subject_to_files[s]
            paths.extend(s_files)
            subj_ids.extend([s] * len(s_files))
        return paths, subj_ids

    train_paths, train_subj_ids = expand(train_subj)
    val_paths, val_subj_ids = expand(val_subj)
    test_paths, test_subj_ids = expand(test_subj)

    train_dataset = EEGGCDataset(train_paths, train_subj_ids, subject_to_label)
    val_dataset = EEGGCDataset(val_paths, val_subj_ids, subject_to_label)
    test_dataset = EEGGCDataset(test_paths, test_subj_ids, subject_to_label)

    return train_dataset, val_dataset, test_dataset


# =========================
# Training / Evaluation
# =========================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)                 # (B,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_segment_level(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")

    return acc, auc


@torch.no_grad()
def evaluate_subject_level(model, loader, device):
    model.eval()
    subj_logits = defaultdict(list)
    subj_labels = {}

    for x, y, subj in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).cpu().numpy()
        y = y.cpu().numpy()

        for l, label, s in zip(logits, y, subj):
            subj_logits[s].append(l)
            subj_labels[s] = int(label)

    all_probs, all_y = [], []
    for s, logits_list in subj_logits.items():
        mean_logit = np.mean(logits_list)
        prob = 1.0 / (1.0 + np.exp(-mean_logit))
        all_probs.append(prob)
        all_y.append(subj_labels[s])

    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_y, preds)
    auc = roc_auc_score(all_y, all_probs)

    return acc, auc


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="EEG GC CNN training (MDD vs Healthy)")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .npy GC matrices.")
    parser.add_argument("--pattern", type=str, default="*.npy",
                        help="Glob pattern for files inside data_dir.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', or 'cuda'")

    parser.add_argument("--log_interval", type=int, default=1,
                        help="How often (epochs) to print validation metrics.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Reproducibility-ish
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Collect files
    pattern = os.path.join(args.data_dir, args.pattern)
    file_paths = sorted(glob.glob(pattern))
    if len(file_paths) == 0:
        raise RuntimeError(f"No files matched {pattern}")

    print(f"Found {len(file_paths)} files.")

    # Create datasets
    train_dataset, val_dataset, test_dataset = make_datasets(
        file_paths, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    print(f"Train segments: {len(train_dataset)}, "
          f"Val segments: {len(val_dataset)}, "
          f"Test segments: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model, loss, optimizer, scheduler
    model = GCNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_val_subj_auc = -np.inf
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_seg_acc, val_seg_auc = evaluate_segment_level(model, val_loader, device)
        val_subj_acc, val_subj_auc = evaluate_subject_level(model, val_loader, device)

        scheduler.step(val_subj_auc)

        if val_subj_auc > best_val_subj_auc:
            best_val_subj_auc = val_subj_auc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val seg Acc/AUC: {val_seg_acc:.3f}/{val_seg_auc:.3f} | "
                f"Val subj Acc/AUC: {val_subj_acc:.3f}/{val_subj_auc:.3f}"
            )

    # Load best model (by subject-level validation AUC)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    # Final test evaluation
    test_seg_acc, test_seg_auc = evaluate_segment_level(model, test_loader, device)
    test_subj_acc, test_subj_auc = evaluate_subject_level(model, test_loader, device)

    print("=== Final Test Metrics ===")
    print(f"Segment-level  Acc: {test_seg_acc:.3f}  AUC: {test_seg_auc:.3f}")
    print(f"Subject-level  Acc: {test_subj_acc:.3f}  AUC: {test_subj_auc:.3f}")


if __name__ == "__main__":
    main()
