"""멀티모달 모델 학습 및 Confusion Matrix 생성."""

import os
import sys
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# 모델 파일 import (파일명에 공백 포함)
spec = importlib.util.spec_from_file_location(
    "multimodal_arch",
    "Multimodal model architecture.py",
)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
MultimodalModel = arch.MultimodalModel

DATA_DIR = "data/processed"
RESULT_DIR = "results"
BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SleepDataset(Dataset):
    """mmap 배열에서 샘플 단위로 로드 (메모리 절약)."""

    def __init__(self, ecg, emg, labels, indices):
        self.ecg = ecg
        self.emg = emg
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        ecg = torch.tensor(self.ecg[i], dtype=torch.float32).unsqueeze(0)
        emg = torch.tensor(self.emg[i], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor([float(self.labels[i])], dtype=torch.float32)
        return ecg, emg, label


def load_data():
    ecg = np.load(f"{DATA_DIR}/ECG_10s.npy", mmap_mode="r")
    emg = np.load(f"{DATA_DIR}/EMG_10s.npy", mmap_mode="r")
    y = np.load(f"{DATA_DIR}/target_10s.npy")
    return ecg, emg, y


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for ecg, emg, labels in loader:
        ecg, emg, labels = ecg.to(DEVICE), emg.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(ecg, emg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for ecg, emg, labels in loader:
        ecg, emg, labels = ecg.to(DEVICE), emg.to(DEVICE), labels.to(DEVICE)
        outputs = model(ecg, emg)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["HC (0)", "PD (1)"],
        yticklabels=["HC (0)", "PD (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Parkinson's Disease Detection")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return cm


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    ecg, emg, y = load_data()
    print(f"Dataset: ECG{ecg.shape} EMG{emg.shape} labels={np.bincount(y.astype(int))}")

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=42, stratify=y[train_idx]
    )

    train_ds = SleepDataset(ecg, emg, y, train_idx)
    val_ds = SleepDataset(ecg, emg, y, val_idx)
    test_ds = SleepDataset(ecg, emg, y, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultimodalModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}",
            flush=True,
        )

    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["HC", "PD"]))

    cm = plot_confusion_matrix(
        y_true, y_pred, f"{RESULT_DIR}/confusion_matrix.png"
    )
    print(f"Confusion Matrix:\n{cm}")

    # 학습 곡선 저장
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{RESULT_DIR}/training_curves.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), f"{RESULT_DIR}/model_weights.pt")
    print(f"\nResults saved to {RESULT_DIR}/")


if __name__ == "__main__":
    main()
