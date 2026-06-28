"""
Author: Andrew Lisenby
Date: 20260627

HW2 Part 4 - Color-space clustering with K-Means.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.image_io import load_image, save_image
from common.utils import print_title, pretty_print, iou

NORMALIZED = "part02/normalized.png"
REFERENCE = "part05/reference_mask.png"

SEED = 898  # Fixed seed for reproducibility.

K_VALUES = (3, 4, 5)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ATTEMPTS = 10


def cluster(hsv, k):
    z = hsv.reshape(-1, 3).astype(np.float32)

    compactness, labels, centers = cv2.kmeans(z, k, None, CRITERIA, ATTEMPTS, cv2.KMEANS_PP_CENTERS)  # type: ignore
    return labels.reshape(hsv.shape[:2]), centers, compactness


def colorize(labels, centers):
    seg_hsv = centers.astype(np.uint8)[labels]
    return cv2.cvtColor(seg_hsv, cv2.COLOR_HSV2BGR)


def best_cluster(labels, k, reference):
    best_idx, best_iou = 0, -1.0
    for c in range(k):
        mask = np.where(labels == c, 255, 0).astype(np.uint8)
        score = iou(mask, reference)
        if score > best_iou:
            best_idx, best_iou = c, score
    return best_idx, best_iou


def main():
    print_title("HW2 Part 4 - K-Means Color Clustering")
    cv2.setRNGSeed(SEED)

    normalized = load_image(NORMALIZED, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    reference = load_image(REFERENCE, cv2.IMREAD_GRAYSCALE)

    candidates = []
    for k in K_VALUES:
        labels, centers, _ = cluster(hsv, k)
        save_image(colorize(labels, centers), f"part04/kmeans_k{k}.png")
        idx, score = best_cluster(labels, k, reference)

        save_image(np.where(labels == idx, 255, 0).astype(np.uint8),
                   f"part04/figure_mask_k{k}.png")
        pretty_print(f"K={k} best cluster", f"#{idx}  IoU {score:.3f}")
        candidates.append((k, labels, idx, score))

    k, labels, idx, score = max(candidates, key=lambda c: c[3])
    mask = np.where(labels == idx, 255, 0).astype(np.uint8)
    foreground = cv2.bitwise_and(normalized, normalized, mask=mask)

    save_image(mask, "part04/figure_mask.png")
    save_image(foreground, "part04/figure_foreground.png")
    pretty_print("Chosen K", f"{k} (cluster #{idx}, IoU {score:.3f})")
    pretty_print("Saved", "part04/figure_mask.png (+ foreground)")


if __name__ == "__main__":
    main()
