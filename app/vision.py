from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

from .schemas import DetectionItem
from .utils import safe_load_json

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class VehicleClassifier:
    def __init__(self, model_path: str = "models/vehicle_classifier.pt", labels_path: str = "models/vehicle_classifier_labels.json"):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.labels = safe_load_json(self.labels_path, default=["bike", "bus", "car", "minivan", "pickup", "suv", "truck"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model()

    def _load_model(self):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.labels))
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state)
        model.eval().to(self.device)
        return model

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        return self.labels[idx], conf


class DamageDetector:
    def __init__(self, model_path: str = "models/damage_detector.pt"):
        self.model_path = Path(model_path)
        self.model = None
        if YOLO is not None and self.model_path.exists():
            self.model = YOLO(str(self.model_path))

    def predict(self, image: Image.Image) -> List[DetectionItem]:
        if self.model is None:
            return []

        results = self.model.predict(image, verbose=False)
        detections: List[DetectionItem] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                DetectionItem(label=str(names[cls_id]), confidence=conf, bbox=xyxy)
            )
        return detections
