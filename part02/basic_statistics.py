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
Part 2 - Step 1: Basic statistics for image analysis.

NOTE: SEE stats/__init__.py for the actual implementations of the statistics functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2

from common.utils import print_title, pretty_print
from common.image_io import load_image
from stats import *

SOURCE_IMAGE = "HW1_IMG_CS898BA.png"


def print_image_metadata(img):
    print_title("Image Metadata")

    stats = get_image_stats(img)

    pretty_print("Image Shape", stats["image_shape"])
    pretty_print("Data Type", stats["image_dtype"])
    pretty_print("Number of Channels", stats["image_channels"])
    pretty_print("Total Pixels", stats["image_pixels"])
    print()


def print_channel_stats(channel, name):
    print_title(f"{name} Channel Statistics")

    stats = get_channel_stats(channel)

    pretty_print("Min", stats["channel_min"])
    pretty_print("Max", stats["channel_max"])
    pretty_print("Mean", f"{stats['channel_mean']:.2f}")
    pretty_print("Median", f"{stats['channel_median']:.2f}")
    pretty_print("Mode", stats["channel_mode"])
    pretty_print("Range", stats["channel_range"])
    pretty_print("Std Dev", f"{stats['channel_std']:.2f}")
    pretty_print("Variance", f"{stats['channel_variance']:.2f}")
    pretty_print("Skew", f"{stats['channel_skew']:.2f}")
    print()


def main():
    print_title("Basic Statistics for Image Analysis")
    print()

    img = load_image(SOURCE_IMAGE, cv2.IMREAD_COLOR)

    # Metadata
    print_image_metadata(img)

    # Per-channel statistics (BGR)
    b, g, r = cv2.split(img)
    print_channel_stats(b, "Blue")
    print_channel_stats(g, "Green")
    print_channel_stats(r, "Red")


if __name__ == "__main__":
    main()
