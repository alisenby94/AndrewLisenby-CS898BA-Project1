"""
Author: Andrew Lisenby
Date: 20260613

Stores and updates metadata about the transformations applied to each image. Used to generate the plots in part3.
"""

import os
import json

from common import ASSETS

METADATA_PATH = "part02/metadata.json"


def _full_path():
    return os.path.join(ASSETS, METADATA_PATH)


def load_metadata():
    """Return the metadata dict (name -> record), or empty if none exists yet."""
    path = _full_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_metadata(meta):
    """Write the full metadata dict to disk."""
    path = _full_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def update_metadata(entries):
    """Merge new entries into the metadata file and return the result."""
    meta = load_metadata()
    meta.update(entries)
    save_metadata(meta)
    return meta
