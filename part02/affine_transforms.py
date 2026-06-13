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

Part 2 - Step 5: Affine transformations.
Reads the 7 color space images and applies 2 unique affine transforms to each
(14 total). Each transform is unique in either type or value. Produces 14 images
saved under assets/part02/affine/ (21 images total with the color space set).

Run color_spaces.py first to generate the input images.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from common.utils import pretty_print, print_title
from common.image_io import load_image, save_image

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

    affine_transforms = {
        # original
        "translate_100_60":       compose_affine(cx, cy, tx=100, ty=60),
        "rotate_15":              compose_affine(cx, cy, theta_deg=15),
        # greyscale
        "rotate_neg30":           compose_affine(cx, cy, theta_deg=-30),
        "scale_1.2":              compose_affine(cx, cy, scale=1.2),
        # binary
        "translate_neg80_120":    compose_affine(cx, cy, tx=-80, ty=120),
        "rotate_90":              compose_affine(cx, cy, theta_deg=90),
        # hsv
        "scale_0.8":              compose_affine(cx, cy, scale=0.8),
        "shear_x_0.10":           compose_affine(cx, cy, shx=0.10),
        # lab
        "rotate_180":             compose_affine(cx, cy, theta_deg=180),
        "translate_150_neg100":   compose_affine(cx, cy, tx=150, ty=-100),
        # hls
        "shear_y_0.12":           compose_affine(cx, cy, shy=0.12),
        "rotate_neg45":           compose_affine(cx, cy, theta_deg=-45),
        # normalized
        "translate_neg100_neg80": compose_affine(cx, cy, tx=-100, ty=-80),
        "rotate_45_scale_1.1":    compose_affine(cx, cy, theta_deg=45, scale=1.1),
    }
    for name in affine_transforms.keys():
        pretty_print(f"Transform - {name}:", "DONE", width=56, value_width=4)
    print('-' * 70)
    pretty_print("Transforms/Image", "2")
    pretty_print("Total transforms", len(affine_transforms))
    print()

    return affine_transforms


def apply_affine_transforms(images, transforms):
    """Apply 2 transforms per image. Returns list of (label, img) pairs."""
    results = []
    keys = list(transforms.keys())

    print_title("Applying Affine Transforms")

    for (img_name, img) in images:
        h, w = img.shape[:2]
        for i in range(2):
            t_label = keys.pop(0)
            M = transforms[t_label]
            results.append((f"{img_name}_{t_label}", cv2.warpAffine(img, M, (w, h))))
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

    for name, t_img in transformed:
        save_image(t_img, f"part02/affine/{name}.png")
        pretty_print(f"Saving {name}.png", "DONE", width=66, value_width=4)
    print('-' * 70)
    pretty_print("New images", len(transformed))
    pretty_print("Total images", len(images) + len(transformed))
    print()


if __name__ == "__main__":
    main()
