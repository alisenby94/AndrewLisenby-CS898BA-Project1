"""
Author: Andrew Lisenby, B.S. Wichita State University.
Date: 20260613

REFS:

[1] CS898BA-HWOne
Cody Farlow, M.S. Wichita State University.
https://github.com/codyfarlow1/CS898BA-HWOne

[2] opencv-gettingstarted
Hannah Dee, PhD. Aberystwyth University.
https://github.com/handee/opencv-gettingstarted

Description:
Part 3 - Edge detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display, just write files
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.utils import print_title, pretty_print
from common.image_io import load_image, save_image
from common.metadata import load_metadata
from common import ASSETS
from part03 import (
    EDGE_TECHNIQUES,
    NUM_SUBSETS,
    CHOSEN_SUBSET,
    RANDOM_SEED,
    NUM_README_PLOTS,
)

SOURCE_FOLDERS = ["part02/colorspaces", "part02/affine", "part02/blurred"]


def load_pool():
    pool = []
    for folder in SOURCE_FOLDERS:
        full = os.path.join(ASSETS, folder)
        names = sorted(f[:-4] for f in os.listdir(full) if f.endswith(".png"))
        for name in names:
            pool.append((name, load_image(f"{folder}/{name}.png")))
    return pool


def split_subsets(images, n, seed):
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    size = len(shuffled) // n
    return [shuffled[i * size:(i + 1) * size] for i in range(n)]


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _normalize(mag):
    mag = mag - mag.min()
    peak = mag.max()
    if peak > 0:
        mag = mag / peak * 255
    return mag.astype(np.uint8)


def sobel_edges(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return _normalize(np.sqrt(gx ** 2 + gy ** 2))


def laplacian_edges(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return _normalize(np.abs(lap))


def canny_edges(gray):
    return cv2.Canny(gray, 100, 200)


def prewitt_edges(gray):
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)
    g = gray.astype(np.float64)
    gx = cv2.filter2D(g, cv2.CV_64F, kx)
    gy = cv2.filter2D(g, cv2.CV_64F, ky)
    return _normalize(np.sqrt(gx ** 2 + gy ** 2))


EDGE_FUNCS = {
    "sobel": sobel_edges,
    "laplacian": laplacian_edges,
    "canny": canny_edges,
    "prewitt": prewitt_edges,
}


def detect_edges(gray):
    """Run every technique on a grayscale image."""
    return {name: EDGE_FUNCS[name](gray) for name in EDGE_TECHNIQUES}


def _for_display(img):
    """Convert to RGB for matplotlib display."""
    if img.ndim == 2:
        return img, "gray"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None


def _edge_display_args(edge):
    vmax = max(1.0, float(np.percentile(edge, 99)))
    return {"cmap": "gray", "vmin": 0, "vmax": vmax}


# darkened background (dark images on white distorted details).
BG_COLOR = "#272727"
TEXT_COLOR = "#f0f0f0"


def save_plot(sample_num, steps, original, edges, path):
    """Build the plot in 't' shape as described in assignment parameters."""
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(3, 3)

    # technique label -> grid cell, with the input in the center.
    layout = {
        "sobel":     gs[0, 1],
        "laplacian": gs[1, 0],
        "_input":    gs[1, 1],
        "canny":     gs[1, 2],
        "prewitt":   gs[2, 1],
    }

    for key, cell in layout.items():
        ax = fig.add_subplot(cell)
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        if key == "_input":
            disp, cmap = _for_display(original)
            ax.imshow(disp, cmap=cmap)
            ax.set_title("Input", color=TEXT_COLOR, fontsize=14)
        else:
            ax.imshow(edges[key], **_edge_display_args(edges[key]))
            ax.set_title(key.capitalize(), color=TEXT_COLOR, fontsize=14)

    title = (f"Sample {sample_num} Pipeline Trajectory:\n"
             + steps[0]
             + "".join(f"\n\u2192 {s}" for s in steps[1:]))
    fig.suptitle(title, color=TEXT_COLOR, fontsize=15, x=0.5, ha="center")

    full = os.path.join(ASSETS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    fig.savefig(full, dpi=80, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)


def main():
    print_title("Edge Detection")

    pool = load_pool()
    metadata = load_metadata()
    subsets = split_subsets(pool, NUM_SUBSETS, RANDOM_SEED)
    subset = subsets[CHOSEN_SUBSET]

    pretty_print("Image pool", len(pool))
    pretty_print("Subsets", f"{NUM_SUBSETS} x {len(subset)}")
    pretty_print("Chosen subset", CHOSEN_SUBSET)
    print("-" * 70)

    saved = 0
    plot_names = []
    for i, (name, img) in enumerate(tqdm(subset, desc="Edge detection", unit="img", ncols=70), start=1):
        # Save the original.
        save_image(img, f"part03/original/{name}.png")
        saved += 1

        gray = to_gray(img)
        edges = detect_edges(gray)
        for tech, edge_img in edges.items():
            save_image(edge_img, f"part03/{tech}/{name}.png")
            saved += 1

        steps = metadata.get(name, {}).get("steps", [name])
        save_plot(i, steps, img, edges, f"part03/plots/{name}.png")
        plot_names.append(name)

    # Pick random plots for README.
    featured = random.Random(RANDOM_SEED).sample(plot_names, NUM_README_PLOTS)

    print()
    pretty_print("Subset images", len(subset))
    pretty_print("Techniques", len(EDGE_TECHNIQUES))
    pretty_print("Images saved", saved)
    pretty_print("Comparison plots", len(plot_names))
    print_title("Featured plots for README:", width=70)
    for fname in featured:
        pretty_print(f"part03/plots/{fname}.png", "FEATURED", width=62, value_width=8)
    print()


if __name__ == "__main__":
    main()
