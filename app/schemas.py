"""
Response/request models shared across the API and services.
Keeping these separate from the services themselves means the Android
client's contract stays stable even as the diagnosis engine's internals
change in later phases.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SpeciesCandidate(BaseModel):
    name: str
    common_name: Optional[str] = None
    probability: float = Field(ge=0.0, le=1.0)


class IdentifyResponse(BaseModel):
    candidates: list[SpeciesCandidate]
    is_plant_probability: float = Field(ge=0.0, le=1.0)


class SignalScore(BaseModel):
    """One scorer's vote. `issue` must match a key in ISSUE_LIBRARY."""
    signal_name: str
    issue: str
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None
    image_index: int = 0  # which uploaded photo this vote came from


class DiagnosisResult(BaseModel):
    issue_key: str
    issue_label: str
    mood_emoji: str
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    fix_steps: list[str]
    manual_check_recommended: bool = False
    supporting_photo_count: int = 1  # photos whose top signal agreed with the final diagnosis
    total_photo_count: int = 1
    agreement_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class ScanResponse(BaseModel):
    identification: Optional[IdentifyResponse] = None
    diagnosis: DiagnosisResult
    signals: list[SignalScore]  # transparency: what voted for what
