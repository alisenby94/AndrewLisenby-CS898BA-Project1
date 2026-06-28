"""
Author: Andrew Lisenby
Date: 20260627

HW2 Part 2 - Multi-channel normalization.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import SOURCE_IMAGE
from common.image_io import load_image, save_image
from common.utils import print_title, pretty_print

OUTPUT = "part02/normalized.png"


def normalize(img):
    # Equalize each BGR channel independently, then merge.
    channels = cv2.split(img)
    equalized = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(equalized)


def main():
    print_title("HW2 Part 2 - Multi-Channel Normalization")

    img = load_image(SOURCE_IMAGE, cv2.IMREAD_COLOR)
    normalized = normalize(img)
    save_image(normalized, OUTPUT)

    for label, before, after in zip("BGR", cv2.split(img), cv2.split(normalized)):
        pretty_print(f"{label} mean (orig -> norm)",
                     f"{before.mean():.1f} -> {after.mean():.1f}")

    pretty_print("Saved", OUTPUT)


if __name__ == "__main__":
    main()
