from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VehicleMetadata(BaseModel):
    model: Optional[str] = None
    fuel_type: Optional[str] = None
    year: Optional[int] = None
    ownership: Optional[str] = None
    price: Optional[float] = None


class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox: Optional[List[float]] = None


class ServiceRecord(BaseModel):
    vehicle_type: str
    vehicle_confidence: float
    detected_damage: List[str] = Field(default_factory=list)
    damage_detections: List[DetectionItem] = Field(default_factory=list)
    customer_intent: str
    intent_confidence: float
    service_priority: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)
