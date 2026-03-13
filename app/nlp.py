from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib

from .utils import safe_load_json


class IntentClassifier:
    def __init__(self, model_dir: str = "models/intent_classifier"):
        model_dir = Path(model_dir)
        self.vectorizer_path = model_dir / "vectorizer.joblib"
        self.classifier_path = model_dir / "classifier.joblib"
        self.labels_path = model_dir / "labels.json"
        self.labels = safe_load_json(self.labels_path, default=[])
        self.vectorizer = joblib.load(self.vectorizer_path) if self.vectorizer_path.exists() else None
        self.classifier = joblib.load(self.classifier_path) if self.classifier_path.exists() else None

    def predict(self, text: str) -> Tuple[str, float]:
        text = (text or "").strip()
        if not text:
            return "general inquiry", 0.0

        # Rule-based fallback for demo mode
        t = text.lower()
        if self.vectorizer is None or self.classifier is None:
            if "insurance" in t or "claim" in t:
                return "insurance claim", 0.91
            if "urgent" in t or "asap" in t or "immediately" in t:
                return "urgent repair", 0.87
            if "service" in t or "repair" in t or "fix" in t:
                return "repair request", 0.84
            if "estimate" in t or "cost" in t or "quote" in t:
                return "repair estimate", 0.82
            return "general inquiry", 0.70

        X = self.vectorizer.transform([text])
        pred = self.classifier.predict(X)[0]
        if hasattr(self.classifier, "predict_proba"):
            proba = float(self.classifier.predict_proba(X).max())
        else:
            proba = 0.80
        return str(pred), proba
