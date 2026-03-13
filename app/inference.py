from __future__ import annotations

from typing import Any, Dict

from .fusion import build_service_record
from .nlp import IntentClassifier
from .schemas import ServiceRecord
from .vision import DamageDetector, VehicleClassifier


class MultiModalPipeline:
    def __init__(self):
        self.vehicle_classifier = VehicleClassifier()
        self.damage_detector = DamageDetector()
        self.intent_classifier = IntentClassifier()

    def predict(self, image, customer_text: str, metadata: Dict[str, Any]) -> ServiceRecord:
        vehicle_type, vehicle_conf = self.vehicle_classifier.predict(image)
        damages = self.damage_detector.predict(image)
        intent, intent_conf = self.intent_classifier.predict(customer_text)

        return build_service_record(
            vehicle_type=vehicle_type,
            vehicle_confidence=vehicle_conf,
            damages=damages,
            customer_intent=intent,
            intent_confidence=intent_conf,
            metadata=metadata,
        )
