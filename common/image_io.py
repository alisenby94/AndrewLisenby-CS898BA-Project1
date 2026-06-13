"""
Author: Andrew Lisenby
Date: 20260613

Reusable image I/O helpers.
"""

import os
import cv2

from common import ASSETS

def load_image(path, flag=cv2.IMREAD_UNCHANGED):
    img = cv2.imread(os.path.join(ASSETS, path), flag)

    # To keep pylance happy
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    
    return img


def save_image(img, path):
    full = os.path.join(ASSETS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    cv2.imwrite(full, img)


def list_images(folder):
    """Load every PNG in an assets folder, sorted by name. Returns (name, img) pairs."""
    full = os.path.join(ASSETS, folder)
    names = sorted(f[:-4] for f in os.listdir(full) if f.endswith(".png"))
    return [(name, load_image(f"{folder}/{name}.png")) for name in names]
