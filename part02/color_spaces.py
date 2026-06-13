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
Part 2 - Steps 2-5: Color space conversions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2

from common.utils import print_title, pretty_print
from common.image_io import load_image, save_image
from common import SOURCE_IMAGE

def convert_color_spaces(img):
    """3 Image stats and image processing.ipynb [2]"""
    # Greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary
    _, binary = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # create copy of HSV, equalize histogram of V channel
    hsv_eq = hsv.copy()
    hsv_eq[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Instructions called for RGB conversion, but openCV uses BGR.
    normalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return {
        "original":   img,
        "greyscale":  grey,
        "binary":     binary,
        "hsv":        hsv,
        "lab":        lab,
        "hls":        hls,
        "normalized": normalized,
    }


def main():
    print_title("Color Space Conversions")

    img = load_image(SOURCE_IMAGE, cv2.IMREAD_COLOR)

    # Convert to all color spaces (including normalized HSV)
    images = convert_color_spaces(img)
    for colorspace in images.keys():
        pretty_print(f"{colorspace} conversion", "DONE")

    print()
    print_title("Saving Color Space Images")

    # Save the colorspace images
    pretty_print("Saving images to", "assets/part02/colorspaces/")  
    print('-' * 70)
    for name, converted in images.items():
        save_image(converted, f"part02/colorspaces/{name}.png")
        pretty_print(f"Saving {name}.png", "DONE")

    print('-' * 70)
    pretty_print("Total images", f"{len(images)}")
    print()


if __name__ == "__main__":
    main()
