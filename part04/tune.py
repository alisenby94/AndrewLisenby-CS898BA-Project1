import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import itertools
import json
import time

import torch

from common import ASSETS
from common.torch_utils import get_device, set_seed
from common.utils import print_title, pretty_print
from part02.data import get_dataloaders
from part03.model import BaselineCNN
from part03.train import plot_curves, save_history, train_model

# Config constants.
LEARNING_RATES = [1e-2, 1e-3, 1e-4] # test LRs
BATCH_SIZES = [32, 64]              # test batch sizes
DROPOUTS = [0.3, 0.5]               # test dropout rates

MODELS_DIR = os.path.join(ASSETS, "models")
RESULTS_DIR = os.path.join(ASSETS, "part04")
FEATURED_DIR = os.path.join(ASSETS, "featured")


def main():
    device = get_device()
    print_title("HW3 Part 4 - Hyperparameter Grid Search")
    pretty_print("Device", device)
    pretty_print("Configurations", len(LEARNING_RATES) * len(BATCH_SIZES) * len(DROPOUTS))
    pretty_print("Epochs per config", 30)
    print()

    results = []
    best = None  # (val_loss, config, history, state_dict)
    all_configs = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROPOUTS))
    total_configs = len(all_configs)
    run_start = time.time()

    for idx, (lr, batch, dropout) in enumerate(all_configs, start=1):
        # Reseed before each run so configs are compared on equal footing.
        set_seed()
        config = {"lr": lr, "batch_size": batch, "dropout": dropout}
        config_start = time.time()
        print_title(f"Config {idx}/{total_configs} | lr={lr}  batch={batch}  dropout={dropout}")
        print("Starting training...", flush=True)

        train_loader, val_loader, _, classes = get_dataloaders(batch_size=batch)
        model = BaselineCNN(num_classes=len(classes), dropout=dropout)
        history = train_model(
            model, train_loader, val_loader, device,
            epochs=30, lr=lr, verbose=True,
        )

        val_loss = history["val_loss"][-1]
        val_acc = history["val_acc"][-1]
        best_acc = max(history["val_acc"])
        results.append({**config, "val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_acc})
        elapsed = time.time() - config_start
        pretty_print("Final val loss", f"{val_loss:.4f}")
        pretty_print("Final val acc", f"{val_acc:.3f}")
        pretty_print("Best val acc", f"{best_acc:.3f}")
        pretty_print("Config time (sec)", f"{elapsed:.1f}")
        print()

        if best is None or val_loss < best[0]:
            best = (val_loss, config, history, {k: v.cpu() for k, v in model.state_dict().items()})

    # Persist the winning config and metrics
    best_loss, best_config, best_history, best_state = best

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(MODELS_DIR, "optimized.pt"))
    save_history(best_history, os.path.join(RESULTS_DIR, "optimized_history.json"))
    plot_curves(
        best_history,
        os.path.join(FEATURED_DIR, "optimized_curves.png"),
        title="Optimized CNN",
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tuning_results.json"), "w") as f:
        json.dump({"results": results, "best": best_config, "best_val_loss": best_loss}, f, indent=2)

    print_title("Best configuration (lowest val loss)")
    pretty_print("Learning rate", best_config["lr"])
    pretty_print("Batch size", best_config["batch_size"])
    pretty_print("Dropout", best_config["dropout"])
    pretty_print("Val loss", f"{best_loss:.4f}")
    pretty_print("Best val acc", f"{max(best_history['val_acc']):.3f}")
    pretty_print("Total tuning time (min)", f"{(time.time() - run_start) / 60.0:.2f}")


if __name__ == "__main__":
    main()
