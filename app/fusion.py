from __future__ import annotations

from typing import Dict, List

from .schemas import DetectionItem, ServiceRecord


HIGH_DAMAGE_CLASSES = {"dent", "shatter", "dislocation", "broken_glass", "glass_shatter"}
MEDIUM_DAMAGE_CLASSES = {"scratch"}
HIGH_URGENCY_INTENTS = {"insurance claim", "urgent repair"}


def compute_service_priority(
    intent: str,
    damages: List[DetectionItem],
    metadata: Dict,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    priority = "low"

    damage_labels = {d.label.lower() for d in damages}
    year = metadata.get("year")

    if intent.lower() in HIGH_URGENCY_INTENTS:
        priority = "high"
        reasons.append(f"intent '{intent}' indicates urgency")

    if damage_labels & HIGH_DAMAGE_CLASSES:
        priority = "high"
        reasons.append("high-severity visible damage detected")
    elif damage_labels & MEDIUM_DAMAGE_CLASSES and priority != "high":
        priority = "medium"
        reasons.append("surface-level visible damage detected")

    if year and isinstance(year, int) and year >= 2020 and priority == "medium":
        reasons.append("newer vehicle, faster service recommended")

    if not reasons:
        reasons.append("no severe damage or urgent customer intent detected")

    return priority, reasons


def build_service_record(
    vehicle_type: str,
    vehicle_confidence: float,
    damages: List[DetectionItem],
    customer_intent: str,
    intent_confidence: float,
    metadata: Dict,
) -> ServiceRecord:
    priority, reasons = compute_service_priority(customer_intent, damages, metadata)

    return ServiceRecord(
        vehicle_type=vehicle_type,
        vehicle_confidence=vehicle_confidence,
        detected_damage=[d.label for d in damages],
        damage_detections=damages,
        customer_intent=customer_intent,
        intent_confidence=intent_confidence,
        service_priority=priority,
        metadata=metadata,
        reasoning=reasons,
    )
