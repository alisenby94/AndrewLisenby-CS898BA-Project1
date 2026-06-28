import numpy as np


def print_title(title, width=70):
    print("-"*width)
    print(f"{'-'*5}{title:^{width-10}}{'-'*5}")
    print("-"*width)

def pretty_format(label, value, width=35, value_width=35):
    return f"{str(label + ':'):<{width}}{str(value):>{value_width}}"

def pretty_print(label, value, width=35, value_width=35):
    print(pretty_format(label, value, width, value_width))


# Evaluation metrics for HW2
def iou(mask, reference):
    a = mask > 127
    b = reference > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0

def dice(mask, reference):
    a = mask > 127
    b = reference > 127
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return 2 * inter / total if total else 0.0