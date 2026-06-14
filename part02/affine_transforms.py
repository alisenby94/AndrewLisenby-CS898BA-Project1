"""
Author: Andrew Lisenby
Date: 20260613

REFS:

[1] CS898BA-HWOne
Cody Farlow, M.S. Wichita State University.
https://github.com/codyfarlow1/CS898BA-HWOne

[2] opencv-gettingstarted
Hannah Dee PhD. Aberystwyth University.
https://github.com/handee/opencv-gettingstarted

[3] Transformation Matrix
https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations

Description:
Part 2 - Step 5: Affine transformations.

NOTE: Run color_spaces.py first to generate the input images.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from common.utils import pretty_print, print_title
from common.image_io import load_image, save_image
from common.metadata import load_metadata, update_metadata

from part02 import COLORSPACE_NAMES


def compose_affine(cx, cy, tx=0.0, ty=0.0, theta_deg=0.0, scale=1.0, shx=0.0, shy=0.0):
    """
    [3] I wanted to do this the hard way to practice linear algebra.
    Composing a single affine matrix from parameterized transforms.
    """
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # T_neg: translate the centroid to the origin so transforms pivot about it
    T_neg = np.array(
        [[1, 0, -cx],
         [0, 1, -cy],
         [0, 0,   1]],
        dtype=np.float64
    )

    # H: shear.  x' = x + shx*y,  y' = y + shy*x
    H = np.array(
        [[  1, shx, 0],
         [shy,   1, 0],
         [  0,   0, 1]],
        dtype=np.float64
    )

    # R: rotation.  [[cos, -sin], [sin, cos]]
    R = np.array(
        [[cos_t, -sin_t, 0],
         [sin_t,  cos_t, 0],
         [    0,      0, 1]],
        dtype=np.float64
    )

    # S: uniform scale.  x' = s*x,  y' = s*y
    S = np.array(
        [[scale,     0, 0],
         [    0, scale, 0],
         [    0,     0, 1]],
        dtype=np.float64
    )

    # T_pos: translate back from origin to centroid
    T_pos = np.array(
        [[1, 0, cx],
         [0, 1, cy],
         [0, 0,  1]],
        dtype=np.float64
    )

    # T: pixel-space translation offset, applied last
    T = np.array(
        [[1, 0, tx],
         [0, 1, ty],
         [0, 0,  1]],
        dtype=np.float64
    )

    # Mathematical representation:
    # M = T_pos(TSRH)T_neg
    # Center data on coordinate system origin
    # -> transform/shear/rotate/scale about origin
    # -> restore original position plus translation offset
    M = T_pos @ T @ S @ R @ H @ T_neg
    return M[:2].astype(np.float32)  # drop the [0,0,1] row for warpAffine


def build_affine_transforms(h, w):
    """
    Generate the transforms to apply to each image.
    """

    print_title("Building Affine Transformation Matrices")

    # calculate centroid of image for rotation/scale
    cx, cy = w / 2, h / 2

    # Each transform carries a hand-written description so downstream steps
    # never have to parse it back out of the file name.
    affine_transforms = {
        # original
        "translate_100_60":       (compose_affine(cx, cy, tx=100, ty=60),            "Affine(Trans:[100,60])"),
        "rotate_15":              (compose_affine(cx, cy, theta_deg=15),             "Affine(Rot:15\u00b0)"),
        # greyscale
        "rotate_neg30":           (compose_affine(cx, cy, theta_deg=-30),            "Affine(Rot:-30\u00b0)"),
        "scale_1.2":              (compose_affine(cx, cy, scale=1.2),                "Affine(Scale:1.2)"),
        # binary
        "translate_neg80_120":    (compose_affine(cx, cy, tx=-80, ty=120),           "Affine(Trans:[-80,120])"),
        "rotate_90":              (compose_affine(cx, cy, theta_deg=90),             "Affine(Rot:90\u00b0)"),
        # hsv
        "scale_0.8":              (compose_affine(cx, cy, scale=0.8),                "Affine(Scale:0.8)"),
        "shear_x_0.10":           (compose_affine(cx, cy, shx=0.10),                 "Affine(Shear X:0.10)"),
        # lab
        "rotate_180":             (compose_affine(cx, cy, theta_deg=180),            "Affine(Rot:180\u00b0)"),
        "translate_150_neg100":   (compose_affine(cx, cy, tx=150, ty=-100),          "Affine(Trans:[150,-100])"),
        # hls
        "shear_y_0.12":           (compose_affine(cx, cy, shy=0.12),                 "Affine(Shear Y:0.12)"),
        "rotate_neg45":           (compose_affine(cx, cy, theta_deg=-45),            "Affine(Rot:-45\u00b0)"),
        # normalized
        "translate_neg100_neg80": (compose_affine(cx, cy, tx=-100, ty=-80),          "Affine(Trans:[-100,-80])"),
        "rotate_45_scale_1.1":    (compose_affine(cx, cy, theta_deg=45, scale=1.1),  "Affine(Rot:45\u00b0, Scale:1.1)"),
    }
    for name in affine_transforms.keys():
        pretty_print(f"Transform - {name}:", "DONE", width=56, value_width=4)
    print('-' * 70)
    pretty_print("Transforms/Image", "2")
    pretty_print("Total transforms", len(affine_transforms))
    print()

    return affine_transforms


def apply_affine_transforms(images, transforms):
    """Apply 2 transforms per image. Returns (name, img, parent, desc) tuples."""
    results = []
    keys = list(transforms.keys())

    print_title("Applying Affine Transforms")

    for (img_name, img) in images:
        h, w = img.shape[:2]
        for i in range(2):
            t_label = keys.pop(0)
            M, desc = transforms[t_label]
            warped = cv2.warpAffine(img, M, (w, h))
            results.append((f"{img_name}_{t_label}", warped, img_name, desc))
            pretty_print(f"{img_name} - {t_label}", "DONE", width=66, value_width=4)
    print()

    return results


def main():
    print_title("Affine Transformations")

    images = [(name, load_image(f"part02/colorspaces/{name}.png")) for name in COLORSPACE_NAMES]
    
    h, w = images[0][1].shape[:2]
    transforms = build_affine_transforms(h, w)
    transformed = apply_affine_transforms(images, transforms)
    
    print_title("Saving Affine Transformed Images")
    pretty_print("Saving images to:", "assets/part02/affine/")
    print('-' * 70)

    meta = load_metadata()
    entries = {}
    for name, t_img, parent, desc in transformed:
        save_image(t_img, f"part02/affine/{name}.png")
        parent_steps = meta.get(parent, {}).get("steps", ["Original"])
        entries[name] = {"steps": parent_steps + [desc]}
        pretty_print(f"Saving {name}.png", "DONE", width=66, value_width=4)
    update_metadata(entries)

    print('-' * 70)
    pretty_print("New images", len(transformed))
    pretty_print("Total images", len(images) + len(transformed))
    print()


if __name__ == "__main__":
    main()
