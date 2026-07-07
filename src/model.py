"""
Simple convolutional autoencoder for image reconstruction.

This is intentionally minimal: a handful of strided convs downsampling the
image, a bottleneck, and transposed convs upsampling it back. No fancy
skip connections, no attention -- the goal is to build intuition, not to
chase benchmark numbers.

The model is trained ONLY on defect-free images. At inference time, a high
reconstruction error signals an anomaly, because the network was never
taught how to reconstruct that kind of pattern.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, img_size: int = 128, latent_channels: int = 128):
        super().__init__()
        self.img_size = img_size

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),  # 8
            nn.ReLU(inplace=True),
        )

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128
            nn.Sigmoid(),  # output in [0, 1], matches ToTensor() range
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


if __name__ == "__main__":
    # Quick sanity check: python -m src.model
    model = ConvAutoencoder(img_size=128)
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == dummy.shape, "Autoencoder output shape must match input"
    print("OK: shapes match.")