"""
Trains the convolutional autoencoder on defect-free ("good") images only.

Usage:
    python -m src.train --config configs/default.yaml
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from src.dataset import MVTecADDataset
from src.model import ConvAutoencoder


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    full_train_ds = MVTecADDataset(
        root_dir=cfg["data_root"],
        category=cfg["category"],
        split="train",
        img_size=cfg["img_size"],
    )

    val_size = int(len(full_train_ds) * cfg["val_split"])
    train_size = len(full_train_ds) - val_size
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. If `optimizer` is given, trains; otherwise, evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, _labels in loader:
            images = images.to(device)

            if is_train:
                optimizer.zero_grad()

            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def main():
    ## Step 0. Configuration and device setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Step 1. Build dataloaders.
    ### Train loader is built out of 188 images and validation loader built out of the rest of the bottle one
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    ## Step 2. Build model, loss function (criterion), and optimizer.
    model = ConvAutoencoder(
        img_size=cfg["img_size"],
        latent_channels=cfg["latent_channels"],
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(
        cfg["checkpoint_dir"], f"{cfg['category']}_best.pt"
    )

    ## Step 3. Training loop. Each epoch, the model compresses and reconstructs
    ## every training image, and the loss is the mean squared error between the
    ## original and reconstructed images. Validation images go through the same
    ## compress/reconstruct process but are never used to update the weights --
    ## they're only used to check the loss on images the model didn't train on.
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(f"Epoch {epoch:03d}/{cfg['epochs']} "
              f"- train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        ## Step 4. Save a checkpoint only when validation loss improves --
        ## this protects against keeping an over-fitted later epoch by mistake.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"  -> new best checkpoint saved to {best_ckpt_path}")

    print(f"Training done. Best val_loss: {best_val_loss:.6f} ({best_ckpt_path})")