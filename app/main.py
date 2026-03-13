from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .inference import MultiModalPipeline
from .utils import load_image_from_bytes

app = FastAPI(title="Vehicle Intelligence Platform", version="1.0.0")
pipeline = MultiModalPipeline()


@app.get("/health")
def health():
    return {"status": "ok", "message": "Vehicle Intelligence API is running"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    customer_text: str = Form(...),
    metadata: str = Form("{}"),
):
    try:
        raw = await image.read()
        pil_image = load_image_from_bytes(raw)
        meta = json.loads(metadata) if metadata else {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}") from e

    result = pipeline.predict(pil_image, customer_text, meta)
    return result.model_dump()
