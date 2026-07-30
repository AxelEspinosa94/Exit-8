"""
FastAPI service wrapping the trained autoencoder for anomaly inference.

This is the "everything in one process" version described in the backlog:
the API loads the model directly and does its own reconstruction-error
math -- same logic as infer.py, just kept alive as a long-running service
instead of a one-off CLI call. A later ticket swaps out the "load the
model and do the math here" part for "call Triton and let it do the math",
without changing the request/response contract below.

Run locally with:
    uvicorn src.api:app --reload --port 8000

Then, in another terminal:
    curl -X POST http://localhost:8000/predict -F "file=@some_image.png"
    curl http://localhost:8000/health
"""

import io
import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.dataset import default_transform
from src.infer import load_model, calibrate_threshold

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "models/bottle_best.pt")
THRESHOLD_PERCENTILE = float(os.environ.get("THRESHOLD_PERCENTILE", "99.0"))

app = FastAPI(title="Anomaly detection API", version="0.1.0")

# Loaded once at startup, never per-request. This is the whole point of a
# long-lived service instead of re-running infer.py's CLI each time --
# the model and the calibrated threshold stay warm in memory.
_state = {"model": None, "cfg": None, "threshold": None, "device": None}


@app.on_event("startup")
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(CHECKPOINT_PATH, device)
    threshold = calibrate_threshold(model, cfg, device, THRESHOLD_PERCENTILE)

    _state["model"] = model
    _state["cfg"] = cfg
    _state["threshold"] = threshold
    _state["device"] = device

    print(f"Loaded checkpoint '{CHECKPOINT_PATH}' | "
          f"threshold (p{THRESHOLD_PERCENTILE}) = {threshold:.6f}")


@app.get("/health")
def health():
    ready = _state["model"] is not None
    return {
        "status": "ok" if ready else "not_ready",
        "threshold": _state["threshold"],
        "category": _state["cfg"]["category"] if ready else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    cfg = _state["cfg"]
    device = _state["device"]
    tensor = default_transform(cfg["img_size"])(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = _state["model"](tensor)
        score = torch.mean((reconstructed - tensor) ** 2).item()

    threshold = _state["threshold"]
    verdict = "ANOMALOUS" if score > threshold else "GOOD"

    return {
        "filename": file.filename,
        "anomaly_score": score,
        "threshold": threshold,
        "verdict": verdict,
    }