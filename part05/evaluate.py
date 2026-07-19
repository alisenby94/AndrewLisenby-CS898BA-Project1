import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)

from common import ASSETS
from common.torch_utils import get_device, set_seed
from common.utils import print_title
from part02.data import get_dataloaders
from part03.model import BaselineCNN

MODELS_DIR = os.path.join(ASSETS, "models")
FEATURED_DIR = os.path.join(ASSETS, "featured")


def load_model(weights_name, num_classes, device):
    model = BaselineCNN(num_classes=num_classes)
    state = torch.load(os.path.join(MODELS_DIR, weights_name), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def build_grid(cm, classes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(14, 10))

    # Row 1: the saved curve figures.
    for col, name, title in [
        (1, "baseline_curves.png", "Baseline curves"),
        (2, "optimized_curves.png", "Optimized curves"),
    ]:
        ax = fig.add_subplot(2, 2, col)
        path = os.path.join(FEATURED_DIR, name)
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path))
        ax.set_title(title)
        ax.axis("off")

    # Row 2: confusion matrix of the optimized model.
    ax_cm = fig.add_subplot(2, 1, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Optimized model - Confusion matrix (test set)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_report(y_true, y_pred, classes):
    """Print per-class Accuracy, Precision, Recall, F1 plus overall accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print(f"{'Class':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for i, name in enumerate(classes):
        print(f"{name:<12}{per_class_acc[i]:>10.3f}{precision[i]:>11.3f}"
              f"{recall[i]:>9.3f}{f1[i]:>8.3f}")
    print(f"\nOverall accuracy: {(y_true == y_pred).mean():.3f}")


def main():
    set_seed()
    device = get_device()

    # Test loader is deterministic, so it matches the trained split (same seed).
    _, _, test_loader, classes = get_dataloaders(batch_size=32)

    baseline = load_model("baseline.pt", len(classes), device)
    optimized = load_model("optimized.pt", len(classes), device)

    y_true_b, y_pred_b = collect_predictions(baseline, test_loader, device)
    y_true_o, y_pred_o = collect_predictions(optimized, test_loader, device)

    print_title("Baseline - Classification report (test set)")
    print_report(y_true_b, y_pred_b, classes)

    print_title("Optimized - Classification report (test set)")
    print_report(y_true_o, y_pred_o, classes)

    cm = confusion_matrix(y_true_o, y_pred_o)
    grid_path = os.path.join(FEATURED_DIR, "comparison_grid.png")
    build_grid(cm, classes, grid_path)

    print_title("Artifacts")
    print(f"Comparison grid saved to {os.path.relpath(grid_path)}")


if __name__ == "__main__":
    main()
