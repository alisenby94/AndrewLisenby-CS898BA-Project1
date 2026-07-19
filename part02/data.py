import os
import sys

# Allow "python part02/data.py" from the project root to resolve common/part02.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common import ASSETS
from common.torch_utils import SEED
from common.utils import print_title, pretty_print

# Config constants.
DATA_DIR = "Fish"           # assets/Fish/<ClassName>/<image>.jpg
IMG_SIZE = 128              # 128x128 target size for all images. May try 224x224 ablation later.
TEST_FRACTION = 0.15        # Reserve 15% of data for testing.
VAL_FRACTION = 0.15         # Reserve 15% of training data for validation.


def _data_root():
    """Return the path to the root of the Fish dataset."""
    return os.path.join(ASSETS, DATA_DIR)


def build_transforms(train):
    """Train pipeline augments. Skip augments for val/test."""
    if train:
        # Return an augmentation pipeline. Resize, randomly flip, rotate, and jitter brightness.
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ])
    # Don't augment val/test; just resize and convert to tensor.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


class TransformedSubset(Dataset):
    """Iterator for each split. Loads and transforms images."""
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_splits():
    """Build stratified train/val/test datasets.
    Configuration constants at top of this file."""

    # Get the class names.
    base = datasets.ImageFolder(_data_root())
    targets = np.array(base.targets)
    indices = np.arange(len(targets))

    # Split for train/test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_FRACTION,
        stratify=targets,
        random_state=SEED,
    )

    # Reserve configured fraction of training data for validation.
    val_relative = VAL_FRACTION / (1.0 - TEST_FRACTION) # scale to split size
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        stratify=targets[train_val_idx],
        random_state=SEED,
    )

    # Initialize the iterators for each split.
    train_ds = TransformedSubset(base, train_idx, build_transforms(train=True))
    val_ds = TransformedSubset(base, val_idx, build_transforms(train=False))
    test_ds = TransformedSubset(base, test_idx, build_transforms(train=False))
    return train_ds, val_ds, test_ds, base.classes


def get_dataloaders(batch_size=32, num_workers=2):
    """Wrap each split in a DataLoader. Only the train loader shuffles."""
    train_ds, val_ds, test_ds, classes = make_splits()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, classes


def main():
    """Test data pipeline and print split statistics."""
    print_title("HW3 Part 2 - Data Pipeline Summary")
    train_ds, val_ds, test_ds, classes = make_splits()

    pretty_print("Image size", f"{IMG_SIZE}x{IMG_SIZE}", width=20, value_width=50)
    pretty_print("Classes", ", ".join(classes), width=20, value_width=50)
    pretty_print("Train images", len(train_ds), width=20, value_width=50)
    pretty_print("Val images", len(val_ds), width=20, value_width=50)
    pretty_print("Test images", len(test_ds), width=20, value_width=50)
    pretty_print("Total", len(train_ds) + len(val_ds) + len(test_ds), width=20, value_width=50)

    print()

    print_title("Per-class counts per split")
    base = train_ds.base
    pretty_print("Class", f"{'Train':>8}{'Val':>8}{'Test':>8}", width=20, value_width=50)
    for cls_idx, cls_name in enumerate(classes):
        tr = sum(base.targets[i] == cls_idx for i in train_ds.indices)
        va = sum(base.targets[i] == cls_idx for i in val_ds.indices)
        te = sum(base.targets[i] == cls_idx for i in test_ds.indices)
        pretty_print(cls_name, f"{tr:>8}{va:>8}{te:>8}", width=20, value_width=50)


if __name__ == "__main__":
    main()
