"""
Response/request models shared across the API and services.
Keeping these separate from the services themselves means the Android
client's contract stays stable even as the diagnosis engine's internals
change in later phases.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ReferencePhoto(BaseModel):
    """
    A real, community-submitted photo of the identified species from
    iNaturalist (https://www.inaturalist.org/) — genuinely free, no key,
    backed by the California Academy of Sciences and National
    Geographic Society. Covers any species (houseplants, vegetables,
    trees, wild plants alike), which is what lets identification widen
    beyond houseplants without needing a separate source per category.
    """
    url: str
    attribution: Optional[str] = None  # real photographer credit — always show this alongside the image


class SpeciesCandidate(BaseModel):
    name: str
    common_name: Optional[str] = None
    probability: float = Field(ge=0.0, le=1.0)
    genus: Optional[str] = None
    family: Optional[str] = None
    gbif_id: Optional[int] = None  # links to the Global Biodiversity
    # Information Facility's record for this species — free to query
    # further (native range, occurrence data) if ever wanted later.


class CareInfo(BaseModel):
    """
    From Perenual (https://perenual.com/), keyed off the top species
    match. Deliberately excludes Perenual's edible_fruit/edible_leaf
    fields — see care_service.py for why.
    """
    watering: Optional[str] = None
    sunlight: list[str] = Field(default_factory=list)
    cycle: Optional[str] = None
    toxic_to_pets: Optional[bool] = None
    source: str = "Perenual"


class WikiSummary(BaseModel):
    """
    A short fallback description from Wikipedia, shown only when
    Perenual's structured care data isn't available. Free, no key, and
    — unlike iNaturalist — genuinely documented and stable, so this one
    carries less first-deployment risk than Day 1's reference photo did.
    """
    extract: str
    url: Optional[str] = None


class GbifTaxonomy(BaseModel):
    """
    Extra taxonomic context from GBIF (gbif.org), fetched using the
    gbif_id already returned alongside identification — no separate
    name lookup needed, no key, no signup, no meaningful rate limit
    for this kind of light use.
    """
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    plant_class: Optional[str] = None  # GBIF calls this "class"; renamed to avoid the Python keyword
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    canonical_name: Optional[str] = None
    rank: Optional[str] = None
    source: str = "GBIF"


class IdentifyResponse(BaseModel):
    candidates: list[SpeciesCandidate]
    is_plant_probability: float = Field(ge=0.0, le=1.0)
    care: Optional[CareInfo] = None
    reference_photo: Optional[ReferencePhoto] = None
    wiki_summary: Optional[WikiSummary] = None
    taxonomy: Optional[GbifTaxonomy] = None


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
    plant_voice_line: str = ""  # short first-person quip from the plant's "perspective"
    # Below: AI-diagnosis-only fields. Left at their defaults (empty
    # list / None / False) by the rule-based engine, which has no
    # concept of "what did I literally observe" separate from "what's
    # my verdict" — that separation is exactly what a vision model
    # can do that colour/texture heuristics can't.
    observations: list[str] = Field(default_factory=list)  # what was actually seen, kept separate from the verdict
    uncertainty_reason: Optional[str] = None  # filled in when confidence is low, explaining why
    follow_up_photo_needed: bool = False  # true when the photo itself is the limiting factor (blur, distance, light)
    follow_up_photo_instruction: Optional[str] = None  # specific guidance for the second photo, e.g. "underside of an affected leaf, close up" — set only when follow_up_photo_needed is True


class AiInsights(BaseModel):
    """
    Extra species-level detail only the AI vision path can offer —
    kept separate from IdentifyResponse's authoritative, trained-model
    identification data so the two are never confused with each other.
    Only populated when use_ai_diagnosis=true and identification
    succeeded.
    """
    species_verification_note: Optional[str] = None  # AI's read on PlantNet's candidate list, e.g. "candidate 1 fits best"
    variety_guess: Optional[str] = None  # cultivar-level guess, e.g. "possibly a Lollo Rosso lettuce" — explicitly a guess, not authoritative
    soil_type_guidance: Optional[str] = None
    bee_friendly: Optional[str] = None  # "yes" | "no" | "unsure"
    bee_friendly_reason: Optional[str] = None
    weather_comment: Optional[str] = None  # short, friendly note tied to current/tomorrow's weather at the scan location — only set when latitude/longitude were provided
    organic_tip: Optional[str] = None  # one standout organic/natural remedy specific to this diagnosis, shown as its own card
    biodynamic_tip: Optional[str] = None  # one standout biodynamic preparation/practice specific to this diagnosis, shown as its own card
    biodynamic_day_type: Optional[str] = None  # "root" | "leaf" | "flower" | "fruit" — computed independently from sidereal moon position, not AI-generated
    biodynamic_day_description: Optional[str] = None  # short caption for biodynamic_day_type, e.g. "Flower day — a day best suited to work on flowering plants."


class ScanResponse(BaseModel):
    identification: Optional[IdentifyResponse] = None
    diagnosis: DiagnosisResult
    signals: list[SignalScore]  # transparency: what voted for what
    ai_insights: Optional[AiInsights] = None
