import io
import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def load_image_from_bytes(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


def safe_load_json(path: str | Path, default: Dict[str, Any] | list | None = None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
