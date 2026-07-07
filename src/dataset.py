"""
PyTorch Dataset for MVTec AD.

Assumes the .tar has already been extracted to disk, with the standard
MVTec AD layout:

    data/raw/<category>/train/good/*.png
    data/raw/<category>/test/good/*.png
    data/raw/<category>/test/<defect_type_1>/*.png
    data/raw/<category>/test/<defect_type_2>/*.png
    ...

This module does NOT extract .tar files. Extraction is a one-time setup
step (see README) and is intentionally kept out of the training/eval loop.
"""

import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(img_size: int):
    """Standard transform: resize, to-tensor, values in [0, 1]."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


class MVTecADDataset(Dataset):
    """
    split="train": returns only defect-free ("good") images, no labels.
                   This is the only data the autoencoder ever trains on.
    split="test":  returns all test images (good + every defect subfolder),
                   with a binary label: 0 = good, 1 = anomalous.
    """

    def __init__(self, root_dir: str, category: str, split: str = "train",
                 img_size: int = 128, transform=None):
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.img_size = img_size
        self.transform = transform or default_transform(img_size)

        self.samples = []  # list of (filepath, label)

        if split == "train":
            good_dir = os.path.join(root_dir, category, "train", "good")
            paths = sorted(glob.glob(os.path.join(good_dir, "*.png")))
            self.samples = [(p, 0) for p in paths]
        else:
            test_dir = os.path.join(root_dir, category, "test")
            defect_types = sorted(os.listdir(test_dir))
            for defect_type in defect_types:
                subdir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(subdir):
                    continue
                label = 0 if defect_type == "good" else 1
                paths = sorted(glob.glob(os.path.join(subdir, "*.png")))
                self.samples.extend((p, label) for p in paths)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for category='{category}', split='{split}' "
                f"under '{root_dir}'. Did you extract the .tar file?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # Quick sanity check: python -m src.dataset
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    category = sys.argv[2] if len(sys.argv) > 2 else "bottle"

    train_ds = MVTecADDataset(root, category, split="train")
    test_ds = MVTecADDataset(root, category, split="test")

    print(f"[{category}] train (good only): {len(train_ds)} images")
    print(f"[{category}] test (good + anomalous): {len(test_ds)} images")

    img, label = train_ds[0]
    print(f"Sample tensor shape: {tuple(img.shape)}, label: {label}")