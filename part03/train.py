import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import ASSETS
from common.torch_utils import get_device, set_seed
from common.utils import print_title, pretty_print
from part02.data import get_dataloaders
from part03.model import BaselineCNN

# Config constants.
BASELINE_EPOCHS = 30
BASELINE_LR = 1e-3
BASELINE_BATCH = 32

MODELS_DIR = os.path.join(ASSETS, "models")
HISTORY_DIR = os.path.join(ASSETS, "part03")
FEATURED_DIR = os.path.join(ASSETS, "featured")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train if optimizer given, else evaluate. Returns (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay=0.0, verbose=True):
    """Train a model and return per-epoch metrics."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f}", flush=True)

    return history


def plot_curves(history, out_path, title="Training Curves"):
    """Save a two-panel loss/accuracy curve figure."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title(f"{title} - Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title(f"{title} - Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main():
    set_seed()
    device = get_device()

    print_title("HW3 Part 3 - Baseline CNN Training")
    pretty_print("Device", device)
    pretty_print("Epochs", BASELINE_EPOCHS)
    pretty_print("Learning rate", BASELINE_LR)
    pretty_print("Batch size", BASELINE_BATCH)
    print()

    train_loader, val_loader, _, classes = get_dataloaders(batch_size=BASELINE_BATCH)
    model = BaselineCNN(num_classes=len(classes), dropout=0.0)

    history = train_model(
        model, train_loader, val_loader, device,
        epochs=BASELINE_EPOCHS, lr=BASELINE_LR,
    )

    weights_path = os.path.join(MODELS_DIR, "baseline.pt")
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), weights_path)

    save_history(history, os.path.join(HISTORY_DIR, "baseline_history.json"))
    plot_curves(
        history,
        os.path.join(FEATURED_DIR, "baseline_curves.png"),
        title="Baseline CNN",
    )

    best_epoch = max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i])

    print()
    pretty_print("Saved weights", os.path.relpath(weights_path))
    pretty_print("Final val loss", f"{history['val_loss'][-1]:.4f}")
    pretty_print("Final val acc", f"{history['val_acc'][-1]:.3f}")
    pretty_print("Best val acc", f"{history['val_acc'][best_epoch]:.3f} (epoch {best_epoch + 1})")


if __name__ == "__main__":
    main()
