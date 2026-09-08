"""
Exports a trained checkpoint to an inference-ready format for Triton.

Triton doesn't run raw PyTorch checkpoints (.pt files saved via
torch.save({"model_state_dict": ...})). It needs either:

  - ONNX: a framework-agnostic graph format, served via Triton's
    onnxruntime backend. This is the default here -- it's portable and
    doesn't require libtorch inside the Triton container.
  - TorchScript: a serialized, traced version of the PyTorch module
    itself, served via Triton's pytorch (libtorch) backend.

Either way, after exporting we run the SAME input through the original
PyTorch model and the exported one, and compare outputs -- if they don't
match closely, the export is not trustworthy, no matter how "successful"
the export step itself looked.

Usage:
    python -m src.export_model --category bottle
    python -m src.export_model --category bottle --format torchscript
    python -m src.export_model --category bottle --format both
"""

import argparse
import os

import numpy as np
import torch

from src.model import ConvAutoencoder
from utils.helper_funcs import load_config


def load_trained_model(category: str, device):
    checkpoint_path = f"models/{category}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]

    model = ConvAutoencoder(
        img_size=cfg["img_size"],
        latent_channels=cfg["latent_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg


def export_onnx(model, dummy_input, out_path: str):
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        },
        opset_version=17,
    )


def export_torchscript(model, dummy_input, out_path: str):
    traced = torch.jit.trace(model, dummy_input)
    traced.save(out_path)
    return traced


@torch.no_grad()
def validate_onnx(model, dummy_input, onnx_path: str, atol: float = 1e-4):
    import onnxruntime as ort

    original_output = model(dummy_input).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.cpu().numpy()})[0]

    max_diff = float(np.max(np.abs(original_output - onnx_output)))
    matches = np.allclose(original_output, onnx_output, atol=atol)

    print(f"[ONNX check] max abs diff vs. original PyTorch output: {max_diff:.8f} "
          f"({'OK' if matches else 'MISMATCH'}, tolerance={atol})")
    return matches


@torch.no_grad()
def validate_torchscript(model, traced_model, dummy_input, atol: float = 1e-4):
    original_output = model(dummy_input).cpu().numpy()
    traced_output = traced_model(dummy_input).cpu().numpy()

    max_diff = float(np.max(np.abs(original_output - traced_output)))
    matches = np.allclose(original_output, traced_output, atol=atol)

    print(f"[TorchScript check] max abs diff vs. original PyTorch output: {max_diff:.8f} "
          f"({'OK' if matches else 'MISMATCH'}, tolerance={atol})")
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--format", type=str, default="onnx",
                         choices=["onnx", "torchscript", "both"])
    args = parser.parse_args()

    device = torch.device("cpu")  # export/validate on CPU for portability
    model, cfg = load_trained_model(args.category, device)

    # A single dummy image is enough to trace the graph / run the comparison --
    # we're checking that the exported computation is faithful, not benchmarking data.
    dummy_input = torch.randn(1, 3, cfg["img_size"], cfg["img_size"], device=device)

    os.makedirs("models", exist_ok=True)

    if args.format in ("onnx", "both"):
        onnx_path = f"models/{args.category}.onnx"
        export_onnx(model, dummy_input, onnx_path)
        print(f"Saved ONNX model to {onnx_path}")
        validate_onnx(model, dummy_input, onnx_path)

    if args.format in ("torchscript", "both"):
        ts_path = f"models/{args.category}_torchscript.pt"
        traced = export_torchscript(model, dummy_input, ts_path)
        print(f"Saved TorchScript model to {ts_path}")
        validate_torchscript(model, traced, dummy_input)


if __name__ == "__main__":
    main()