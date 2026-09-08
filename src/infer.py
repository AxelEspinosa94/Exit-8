"""
Runs inference on a single image using a trained autoencoder checkpoint.

This mirrors what a real inspection system would do at runtime: load one
image, get a reconstruction error, compare it against a threshold, and
report good/anomalous. No test-set labels involved -- this script doesn't
know (or care) whether the image is actually defective.

Since the checkpoint doesn't store a threshold, this script recomputes one
the same way evaluate.py does: from the reconstruction error distribution
on the "good" training images. If you already ran evaluate.py and want to
match its result exactly, pass the same --threshold_percentile it used.

Usage:
    python -m src.infer --image path/to/image.png --checkpoint models/bottle_best.pt
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.dataset import MVTecADDataset, default_transform
from src.model import ConvAutoencoder
from utils.helper_funcs import load_config


def load_model(category: str, device):
    """Loads the checkpoint and returns (model, cfg)."""
    cfg = load_config(category)
    threshold_percentile = cfg["threshold_percentile"]
    checkpoint_path = f"{cfg['checkpoint_dir']}/{category}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ConvAutoencoder(
        img_size=cfg["img_size"],
        latent_channels=cfg["latent_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, threshold_percentile, cfg


@torch.no_grad()
def score_image(model, image_path: str, img_size: int, device) -> float:
    """Returns the reconstruction error (MSE) for a single image."""
    image = Image.open(image_path).convert("RGB")
    tensor = default_transform(img_size)(image).unsqueeze(0).to(device)  # add batch dim

    reconstructed = model(tensor)
    error = torch.mean((reconstructed - tensor) ** 2).item()
    return error


@torch.no_grad()
def calibrate_threshold(model, cfg: dict, device, percentile: float) -> float:
    """Recomputes the threshold from the good-training-image error distribution,
    the same way evaluate.py does. Kept separate from evaluate.py's own
    calibration so this script has no dependency on it beyond the checkpoint.
    """
    train_good_ds = MVTecADDataset(
        root_dir=cfg["data_root"], category=cfg["category"],
        split="train", img_size=cfg["img_size"],
    )
    loader = DataLoader(train_good_ds, batch_size=cfg["batch_size"], shuffle=False)

    scores = []
    for images, _labels in loader:
        images = images.to(device)
        reconstructed = model(images)
        per_image_error = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))
        scores.append(per_image_error.cpu().numpy())

    scores = np.concatenate(scores)
    return float(np.percentile(scores, percentile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a single image")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Use a fixed threshold instead of recalibrating it")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, threshold_percentile, cfg = load_model(args.category, device)

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = calibrate_threshold(model, cfg, device, threshold_percentile)

    score = score_image(model, args.image, cfg["img_size"], device)
    verdict = "ANOMALOUS" if score > threshold else "GOOD"

    print(f"Image             : {args.image}")
    print(f"Reconstruction err: {score:.6f}")
    print(f"Threshold         : {threshold:.6f}")
    print(f"Verdict           : {verdict}")


if __name__ == "__main__":
    main()