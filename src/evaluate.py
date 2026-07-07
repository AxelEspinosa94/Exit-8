"""
Evaluates a trained autoencoder as an anomaly detector.

Steps:
    1. Load the trained checkpoint.
    2. Compute per-image reconstruction error (MSE) on the "good" training
       images -> this distribution defines what "normal" error looks like.
    3. Pick a threshold from that distribution (e.g. the 99th percentile).
    4. Compute per-image reconstruction error on the full test set
       (good + every defect type), which DOES have labels.
    5. Report ROC-AUC (threshold-independent) and precision/recall/F1 at the
       chosen threshold. Save a histogram comparing good vs. anomalous scores.

Usage:
    python -m src.evaluate --config configs/default.yaml --checkpoint models/bottle_best.pt
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)
import matplotlib.pyplot as plt

from src.dataset import MVTecADDataset
from src.model import ConvAutoencoder


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def compute_scores(model, loader, device):
    """Returns (scores, labels) as numpy arrays. score = per-image mean MSE."""
    model.eval()
    all_scores = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        reconstructed = model(images)

        # per-image MSE: average over channels/height/width, keep batch dim
        per_image_error = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))

        all_scores.append(per_image_error.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def plot_score_distribution(scores, labels, threshold, out_path):
    good_scores = scores[labels == 0]
    anomalous_scores = scores[labels == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(good_scores, bins=30, alpha=0.6, label="good (test)", color="tab:green")
    plt.hist(anomalous_scores, bins=30, alpha=0.6, label="anomalous (test)", color="tab:red")
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold = {threshold:.5f}")
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Count")
    plt.title("Anomaly score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_pr_curves(labels, scores, out_path):
    """Saves a side-by-side ROC curve and Precision-Recall curve.
 
    Unlike the single accuracy/precision/recall numbers printed to console
    (which depend on ONE chosen threshold), these curves show every possible
    threshold at once -- useful for picking a threshold, or for comparing
    two different models regardless of where you'd set the cutoff.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
 
    precision, recall, _ = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    axes[0].plot(fpr, tpr, color="tab:blue", label=f"ROC curve (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")
 
    axes[1].plot(recall, precision, color="tab:orange",
                 label=f"PR curve (avg precision = {avg_precision:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left")
 
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold_percentile", type=float, default=99.0,
                         help="Percentile of the good-image train error used as threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load model ---
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ConvAutoencoder(
        img_size=cfg["img_size"],
        latent_channels=cfg["latent_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.6f})")

    # --- calibrate threshold on "good" training images ---
    train_good_ds = MVTecADDataset(
        root_dir=cfg["data_root"], category=cfg["category"],
        split="train", img_size=cfg["img_size"],
    )
    train_good_loader = DataLoader(train_good_ds, batch_size=cfg["batch_size"], shuffle=False)
    good_scores, _ = compute_scores(model, train_good_loader, device)
    threshold = float(np.percentile(good_scores, args.threshold_percentile))
    print(f"Threshold (p{args.threshold_percentile:.0f} of good-train scores): {threshold:.6f}")

    # --- evaluate on the labeled test set ---
    test_ds = MVTecADDataset(
        root_dir=cfg["data_root"], category=cfg["category"],
        split="test", img_size=cfg["img_size"],
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_scores, test_labels = compute_scores(model, test_loader, device)

    predictions = (test_scores > threshold).astype(int)

    auc = roc_auc_score(test_labels, test_scores)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    f2 = fbeta_score(test_labels, predictions, beta=2, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()

    print("\n--- Results on test set ---")
    print(f"Image-level ROC-AUC : {auc:.4f}  (threshold-independent, use this to compare models)")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1                  : {f1:.4f}")
    print(f"F2                  : {f2:.4f}")
    print(f"Confusion matrix    : TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    os.makedirs("outputs", exist_ok=True)
    plot_path = os.path.join("outputs", f"{cfg['category']}_score_distribution.png")
    plot_score_distribution(test_scores, test_labels, threshold, plot_path)
    print(f"\nScore distribution plot saved to {plot_path}")

    # Plot ROC and Precision-Recall curves
    roc_pr_path = os.path.join("outputs", f"{cfg['category']}_roc_pr_curves.png")
    plot_roc_pr_curves(test_labels, test_scores, roc_pr_path)
    print(f"ROC and PR curves saved to {roc_pr_path}")

if __name__ == "__main__":
    main()