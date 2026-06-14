"""
Author: Andrew Lisenby, B.S. Wichita State University.
Date: 20260613

REFS:

[1] CS898BA-HWOne
Cody Farlow, M.S. Wichita State University.
https://github.com/codyfarlow1/CS898BA-HWOne

[2] opencv-gettingstarted
Hannah Dee PhD. Aberystwyth University.
https://github.com/handee/opencv-gettingstarted

[3] Gaussian Blur: Separable Convolution vs Full 2D Convolution
Raymond Tay.
https://medium.com/@RaymondTayBL/gaussian-blur-separable-convolution-vs-full-2d-convolution-d5e9b64bf84f

Description:
Part 2 - Step 6: Gaussian blur.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.ndimage import convolve1d
from tqdm import tqdm

from common.utils import pretty_format, print_title, pretty_print
from common.image_io import load_image, save_image, list_images
from common.metadata import load_metadata, update_metadata
from part02 import COLORSPACE_NAMES, SIGMAS

def gaussian_kernel_1d(sigma, kernel_size=5):
    """
    [3] 1D Gaussian kernel for a given sigma.
    Separable convolution costs O(2k) vs O(k^2).
    G(x, y) = G_1(x) × G_1(y)
    """
    x = np.arange(kernel_size) - kernel_size // 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(img, sigma):
    """[3] Apply Gaussian blur to an image in 2 convolutions."""
    k = gaussian_kernel_1d(sigma)
    out = img.astype(np.float64)
    # vertical
    out = convolve1d(out, k, axis=0)
    # horizontal
    out = convolve1d(out, k, axis=1)
    return np.clip(out, 0, 255).astype(img.dtype)


def blur_and_save(all_images, sigmas):
    """Blur each image at every sigma and save immediately to improve memory usage."""
    count = 0
    total = len(all_images) * len(sigmas)
    meta = load_metadata()
    entries = {}
    pretty_print("Saving blurred images to:", "assets/part02/blurred/", width=35, value_width=35)
    print('-' * 70)
    with tqdm(total=total, desc="Blurring", unit="img", ncols=70) as bar:
        for img_name, img in all_images:
            parent_steps = meta.get(img_name, {}).get("steps", ["Original"])
            for sigma in sigmas:
                name = f"{img_name}_blur_s{sigma}"
                save_image(gaussian_blur(img, sigma), f"part02/blurred/{name}.png")
                entries[name] = {"steps": parent_steps + [f"Gaussian Blur(\u03c3:{sigma})"]}
                count += 1
                tqdm.write(pretty_format(f"Saving blurred/{name}.png", "DONE", width=66, value_width=4))
                bar.update(1)
    update_metadata(entries)
    return count


def main():
    # My GPU is occupied with other research. KISS implementation.
    print_title("Gaussian Blur (single threaded)", width=70)

    colorspaces = [(name, load_image(f"part02/colorspaces/{name}.png")) for name in COLORSPACE_NAMES]
    affine = list_images("part02/affine")
    all_images = colorspaces + affine

    blurred = blur_and_save(all_images, SIGMAS)
    print('-' * 70)
    pretty_print("Sigma levels per image", len(SIGMAS))
    pretty_print("Source images", len(all_images))
    pretty_print("Blurred images", blurred)
    pretty_print("Total images", len(all_images) + blurred)
    print()


if __name__ == "__main__":
    main()
