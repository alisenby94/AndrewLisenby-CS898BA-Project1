import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

__all__ = [
    "get_channel_min",
    "get_channel_max",
    "get_channel_mean",
    "get_channel_median",
    "get_channel_mode",
    "get_channel_range",
    "get_channel_std",
    "get_channel_variance",
    "get_channel_skew",
    "get_image_shape",
    "get_image_dtype",
    "get_image_channels",
    "get_image_pixels",
    "get_channel_stats",
    "get_image_stats",
]

def get_channel_min(channel):
    return np.min(channel)

def get_channel_max(channel):
    return np.max(channel)

def get_channel_mean(channel):
    return np.mean(channel)

def get_channel_median(channel):
    return np.median(channel)

def get_channel_mode(channel):
    return int(np.bincount(channel.flatten()).argmax())

def get_channel_range(channel):
    return int(np.max(channel) - np.min(channel))

def get_channel_std(channel):
    return np.std(channel)

def get_channel_variance(channel):
    return np.var(channel)

def get_channel_skew(channel):
    from scipy import stats
    return stats.skew(channel.flatten())

def get_image_shape(img):
    return img.shape

def get_image_dtype(img):
    return img.dtype

def get_image_channels(img):
    """Get number of channels (XYC for openCV vs CXY for pytorch)"""
    return img.shape[2] if len(img.shape) == 3 else 1

def get_image_pixels(img):
    return img.size

__channel_callback__ = {
    "channel_min": get_channel_min,
    "channel_max": get_channel_max,
    "channel_mean": get_channel_mean,
    "channel_median": get_channel_median,
    "channel_mode": get_channel_mode,
    "channel_range": get_channel_range,
    "channel_std": get_channel_std,
    "channel_variance": get_channel_variance,
    "channel_skew": get_channel_skew,
}

__image_callback__ = {
    "image_shape": get_image_shape,
    "image_dtype": get_image_dtype,
    "image_channels": get_image_channels,
    "image_pixels": get_image_pixels,
}

def get_channel_stats(channel, stats_list=__channel_callback__.keys()):
    """Returns dict with the requested channel statistics:
        channel_min,
        channel_max,
        channel_mean,
        channel_median,
        channel_mode,
        channel_range,
        channel_std,
        channel_variance,
        channel_skew
    """
    stat_ret = {}
    for stat in stats_list:
        stat_ret[stat] = __channel_callback__[stat](channel)
    return stat_ret

def get_image_stats(img, stats_list=__image_callback__.keys()):
    """Returns dict with the requested image statistics:
        image_shape,
        image_dtype,
        image_channels,
        image_pixels
    """
    stat_ret = {}
    for stat in stats_list:
        stat_ret[stat] = __image_callback__[stat](img)
    return stat_ret