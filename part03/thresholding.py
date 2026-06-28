"""
Author: Andrew Lisenby
Date: 20260627

HW2 Part 3 - Threshold-based segmentation.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.image_io import load_image, save_image
from common.utils import print_title, pretty_print

NORMALIZED = "part02/normalized.png"

# Picked large block to capture a more representative portion of the figure.
BLOCK_SIZE = 51
C = 5


def main():
    print_title("HW2 Part 3 - Threshold-Based Segmentation")

    normalized = load_image(NORMALIZED, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pretty_print("Otsu threshold", f"{t:.1f}")

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, BLOCK_SIZE, C,
    )

    for name, mask in (("otsu", otsu), ("adaptive", adaptive)):
        foreground = cv2.bitwise_and(normalized, normalized, mask=mask)
        save_image(mask, f"part03/{name}_mask.png")
        save_image(foreground, f"part03/{name}_foreground.png")
        pretty_print(f"Saved {name}", f"part03/{name}_mask.png (+ foreground)")


if __name__ == "__main__":
    main()
