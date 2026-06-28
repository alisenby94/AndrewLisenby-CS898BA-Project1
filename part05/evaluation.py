"""
Author: Andrew Lisenby
Date: 20260627

HW2 Part 5 - Evaluation and analysis.
"""

import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import ASSETS, SOURCE_IMAGE
from common.image_io import load_image
from common.utils import print_title, pretty_print, iou, dice

NORMALIZED = "part02/normalized.png"
REFERENCE = "part05/reference_mask.png"
PLOT = "featured/segmentation_comparison.png"
ERROR_PLOT = "featured/segmentation_error_maps.png"
KMEANS_ERROR_PLOT = "featured/kmeans_error_maps.png"

K_VALUES = (3, 4, 5)


def load_reference():
    return load_image(REFERENCE, cv2.IMREAD_GRAYSCALE)

def error_map(mask, reference):
    a = mask > 127
    b = reference > 127
    rgb = np.zeros((*a.shape, 3), np.uint8)
    rgb[..., 0] = np.where(a & ~b, 255, 0)  # false positive
    rgb[..., 1] = np.where(a & b, 255, 0)   # true positive
    rgb[..., 2] = np.where(~a & b, 255, 0)  # false negative
    return rgb

def main():
    print_title("HW2 Part 5 - Evaluation and Analysis")

    original = load_image(SOURCE_IMAGE, cv2.IMREAD_COLOR)
    normalized = load_image(NORMALIZED, cv2.IMREAD_COLOR)
    reference = load_reference()

    masks = {
        "Otsu": load_image("part03/otsu_mask.png", cv2.IMREAD_GRAYSCALE),
        "Adaptive": load_image("part03/adaptive_mask.png", cv2.IMREAD_GRAYSCALE),
        "K-Means": load_image("part04/figure_mask.png", cv2.IMREAD_GRAYSCALE),
    }

    pretty_print("Method", "IoU / Dice")
    for name, mask in masks.items():
        pretty_print(name, f"{iou(mask, reference):.3f} / {dice(mask, reference):.3f}")

    panels = [
        ("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)),
        ("Normalized", cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)),
        ("Reference", reference),
        ("Otsu", masks["Otsu"]),
        ("Adaptive", masks["Adaptive"]),
        ("K-Means", masks["K-Means"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    out = os.path.join(ASSETS, PLOT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    pretty_print("Saved plot", f"assets/{PLOT}")

    fig, axes = plt.subplots(1, len(masks), figsize=(16, 5))
    for ax, (name, mask) in zip(axes.ravel(), masks.items()):
        ax.imshow(error_map(mask, reference))
        ax.set_title(f"{name}  (IoU {iou(mask, reference):.3f})")
        ax.axis("off")
    fig.suptitle("Error Maps  -  Green: True Positive   Red: False Positive   Blue: False Negative")
    fig.tight_layout()

    err = os.path.join(ASSETS, ERROR_PLOT)
    fig.savefig(err, dpi=110, bbox_inches="tight")
    plt.close(fig)
    pretty_print("Saved error maps", f"assets/{ERROR_PLOT}")

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(16, 5))
    for ax, k in zip(axes.ravel(), K_VALUES):
        km = load_image(f"part04/figure_mask_k{k}.png", cv2.IMREAD_GRAYSCALE)
        ax.imshow(error_map(km, reference))
        ax.set_title(f"K={k}  (IoU {iou(km, reference):.3f})")
        ax.axis("off")
    fig.suptitle("K-Means Error Maps  -  Green: True Positive   Red: False Positive   Blue: False Negative")
    fig.tight_layout()

    kerr = os.path.join(ASSETS, KMEANS_ERROR_PLOT)
    fig.savefig(kerr, dpi=110, bbox_inches="tight")
    plt.close(fig)
    pretty_print("Saved K-Means maps", f"assets/{KMEANS_ERROR_PLOT}")

if __name__ == "__main__":
    main()