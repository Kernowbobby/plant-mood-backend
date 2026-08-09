"""
Phase 2 diagnosis path: sends the plant photo(s) directly to a
vision-capable Claude model instead of relying on the hand-written
colour/texture heuristics in diagnosis_engine.py.

Uses Anthropic's Structured Outputs feature (output_config.format,
generally available on Haiku 4.5 and later) rather than asking for
JSON in free text and hoping — the API constrains generation so the
response is guaranteed to match the schema below. No more stripping
```json fences or catching json.loads failures.

Design choices, informed by outside review (two independent AI
reviews of this project both converged on these points):
  - Observations are returned separately from the diagnosis verdict,
    so a wrong guess is grounded in what was actually seen rather than
    presented as a bare confident-sounding label.
  - The model is given PlantNet's own candidate list (name + probability)
    and asked to verify/correct/reject against it, rather than guessing
    species from scratch — this is a verification task, which vision
    models handle far more reliably than open-ended classification.
  - The model has explicit permission to say it's unsure or that the
    photo itself is the problem (too blurry, too far away, bad light),
    rather than being forced into a confident-sounding diagnosis.

MOCK MODE IS THE DEFAULT. This costs real (small) money per call once
switched on — see Settings.ai_diagnosis_mock_mode. Nothing here is
called with a real API key until that's deliberately set in Railway.
Even then, each scan costs a fraction of a cent (Haiku 4.5 pricing).
"""
import base64
import json
import logging

import httpx

from app.config import get_settings
from app.schemas import AiInsights, DiagnosisResult, SignalScore, SpeciesCandidate

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_TIMEOUT_SECONDS = 90.0  # first use of a schema compiles it server-side on Anthropic's end,
# which can genuinely take a while — 30s was too eager and caused silent,
# untraceable-looking fallbacks to the rule-based engine. See Android's
# ApiClient.kt for the matching client-side timeout.

# Keep in sync with issue_library.py's keys — see that file for why.
_KNOWN_ISSUES = [
    "overwatering", "underwatering", "light_stress", "nutrient_yellowing",
    "powdery_mildew", "spider_mites", "healthy", "unclear",
]

_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short, literal observations about what's visible in the photo(s) — "
                            "e.g. 'lower leaves yellowing', 'soil looks dry'. Kept separate from any verdict.",
        },
        "species_verification_note": {
            "type": ["string", "null"],
            "description": "If candidate species were provided: which one (if any) best fits what's "
                            "visible, and why. State plainly if none of them look right. Null if no "
                            "candidates were provided.",
        },
        "issue_key": {"type": "string", "enum": _KNOWN_ISSUES},
        "issue_label": {"type": "string"},
        "mood_emoji": {"type": "string"},
        "confidence": {
            "type": "number",
            "description": "0 to 1. This is diagnosis confidence specifically — keep it honest and "
                            "separate from how sure you are about the species.",
        },
        "summary": {"type": "string"},
        "fix_steps": {"type": "array", "items": {"type": "string"}},
        "manual_check_recommended": {"type": "boolean"},
        "uncertainty_reason": {
            "type": ["string", "null"],
            "description": "If confidence is low, explain why in one short phrase. Null if confidence is high.",
        },
        "follow_up_photo_needed": {
            "type": "boolean",
            "description": "True if the photo itself is the limiting factor — too blurry, too far away, "
                            "bad lighting, only part of the plant visible — rather than the plant's condition.",
        },
        "follow_up_photo_instruction": {
            "type": ["string", "null"],
            "description": "Only when follow_up_photo_needed is true: precise guidance for the next photo — "
                    "name the exact plant part and what to look for, e.g. 'underside of a yellowing "
                    "leaf, close enough to see any insects or eggs'. Null otherwise.",
        },
        "variety_guess": {
            "type": ["string", "null"],
            "description": "A cultivar/variety-level guess if the photo gives enough detail to attempt "
                            "one, e.g. 'possibly a Lollo Rosso lettuce, given the ruffled red-tinged leaves'. "
                            "Explicitly a guess, not an authoritative ID. Null if there's not enough detail.",
        },
        "soil_type_guidance": {
            "type": ["string", "null"],
            "description": "A brief note on soil preference for this plant, if identifiable enough to say. "
                            "Null if the plant couldn't be identified with any confidence.",
        },
        "bee_friendly": {
            "type": "string",
            "enum": ["yes", "no", "unsure"],
            "description": "Whether this plant is generally considered good for bees/pollinators.",
        },
        "bee_friendly_reason": {
            "type": ["string", "null"],
            "description": "A short, friendly reason for the bee_friendly answer, e.g. 'open, "
                            "nectar-rich flowers that pollinators favour'. Null if 'unsure'.",
        },
        "plant_voice_line": {
            "type": "string",
            "description": "A short, first-person, slightly humorous line as if the plant itself were "
                            "speaking about how it feels.",
        },
    },
    "required": [
        "observations", "species_verification_note", "issue_key", "issue_label", "mood_emoji",
        "confidence", "summary", "fix_steps", "manual_check_recommended", "uncertainty_reason",
        "follow_up_photo_needed", "variety_guess", "soil_type_guidance", "bee_friendly",
        "bee_friendly_reason", "plant_voice_line",
    ],
}

_DIAGNOSIS_PROMPT_HEADER = """You are looking at one or more photos of a houseplant or garden \
plant, taken by an amateur gardener on their phone. First note what you actually observe, \
then diagnose its condition.

Be honest about uncertainty — a modest confidence with a clear uncertainty_reason is far more \
useful than a confident-sounding guess. If the photo doesn't contain a plant at all, or is too \
blurry/distant/poorly lit to assess, say so plainly rather than forcing a diagnosis.
If you cannot confidently distinguish between two or more meaningfully different diagnoses (e.g. which pest, which disease, deficiency vs. early disease) from what's visible, and a specific close-up would resolve that ambiguity, set follow_up_photo_needed to true.

These two fields serve different purposes and must NOT be merged: uncertainty_reason explains WHY you're unsure in general terms (for the human reading the result). follow_up_photo_instruction is a SEPARATE, ACTIONABLE instruction telling the camera app exactly what photo to take next — name the exact plant part, what to look for, and any framing guidance (e.g. "underside of a yellowing leaf, close enough to see any insects or eggs"). Whenever follow_up_photo_needed is true, follow_up_photo_instruction must be filled in with this actionable guidance — it is never left null when follow_up_photo_needed is true, even if uncertainty_reason already touches on similar detail.

Do not request a follow-up photo just because you're not 100% certain — only when a specific photo would genuinely resolve a real ambiguity. If you can identify the plant and see a clear issue, give your best diagnosis with an honest confidence level instead."""


def _format_candidates(candidates: list[SpeciesCandidate]) -> str:
    if not candidates:
        return ""
    lines = ["\nA separate species-identification model suggested these candidates for this photo "
             "(most likely first) — use these as a reference, but trust what you actually see in the "
             "photo over the list if they conflict:"]
    for c in candidates[:5]:
        label = f"{c.name}" + (f" ({c.common_name})" if c.common_name else "")
        lines.append(f"- {label} — {c.probability * 100:.0f}% confidence")
    return "\n".join(lines)
def _format_wiki(wiki_summary: str | None) -> str:
    if not wiki_summary:
        return ""
    return ("\n\nHere is a general Wikipedia summary for this species, provided as background only:\n"
            f'"{wiki_summary}"\n\n'
            "Only mention something from this background if it is genuinely relevant to your "
            "observations or advice (e.g. a known growth habit that explains what you're seeing). "
            "Do not summarise or quote this text for its own sake, and ignore it entirely if it "
            "doesn't help explain what's in the photo.")


class AiDiagnosisService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def diagnose(
        self,
        images: list[bytes],
        candidates: list[SpeciesCandidate] | None = None,
        wiki_summary: str | None = None,
    ) -> tuple[DiagnosisResult, list[SignalScore], AiInsights]:
        if not images:
            raise ValueError("At least one photo is required.")

        if self._settings.ai_diagnosis_mock_mode:
            return self._mock_diagnose(images)

        try:
            return await self._real_diagnose(images, candidates or [], wiki_summary)
        except Exception:
            logger.exception("AI diagnosis call failed; caller should fall back to the rule-based engine.")
            raise

    # ------------------------------------------------------------------
    # Mock mode — zero cost, zero network calls, realistic shape.
    # ------------------------------------------------------------------
    def _mock_diagnose(self, images: list[bytes]) -> tuple[DiagnosisResult, list[SignalScore], AiInsights]:
        logger.info("AI diagnosis running in MOCK mode — no API key set, no real call made.")
        result = DiagnosisResult(
            issue_key="underwatering",
            issue_label="Underwatered (mock)",
            mood_emoji="🥵",
            confidence=0.8,
            summary="[MOCK RESPONSE — no real AI call was made] Leaves show some dry, "
                    "crisping edges consistent with underwatering.",
            fix_steps=[
                "Water thoroughly until it drains from the bottom of the pot.",
                "Check soil moisture with a finger 2 inches down before watering again.",
            ],
            manual_check_recommended=False,
            supporting_photo_count=len(images),
            total_photo_count=len(images),
            agreement_ratio=1.0,
            plant_voice_line="This is a mock response — set ANTHROPIC_API_KEY to go live.",
            observations=["[MOCK] some leaf edges appear dry", "[MOCK] soil surface looks pale/dry"],
            uncertainty_reason=None,
            follow_up_photo_needed=False,
        )
        signals = [SignalScore(
            signal_name="ai_vision_mock",
            issue="underwatering",
            confidence=0.8,
            notes="Mock mode — no real API call was made.",
            image_index=0,
        )]
        insights = AiInsights(
            species_verification_note="[MOCK] would confirm/reject PlantNet's top candidate here.",
            variety_guess=None,
            soil_type_guidance="[MOCK] well-draining potting mix.",
            bee_friendly="unsure",
            bee_friendly_reason="[MOCK RESPONSE]",
        )
        return result, signals, insights

    # ------------------------------------------------------------------
    # Real mode — actual API call, small real cost per scan.
    # ------------------------------------------------------------------
    async def _real_diagnose(
        self,
        images: list[bytes],
        candidates: list[SpeciesCandidate],
        wiki_summary: str | None = None,
    ) -> tuple[DiagnosisResult, list[SignalScore], AiInsights]:
        content: list[dict] = []
        for img_bytes in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode("ascii"),
                },
            })
        prompt = _DIAGNOSIS_PROMPT_HEADER + _format_candidates(candidates) + _format_wiki(wiki_summary)
        content.append({"type": "text", "text": prompt})

        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._settings.ai_diagnosis_model,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _RESPONSE_SCHEMA,
                },
            },
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(_ANTHROPIC_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            payload_out = resp.json()

        raw_text = "".join(
            block.get("text", "") for block in payload_out.get("content", []) if block.get("type") == "text"
        )
        # Structured outputs guarantees schema-valid JSON, but a defensive
        # strip costs nothing and protects against edge cases (e.g. a
        # truncated response if max_tokens is hit).
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)

        issue_key = parsed.get("issue_key", "unclear")
        if issue_key not in _KNOWN_ISSUES:
            issue_key = "unclear"

        result = DiagnosisResult(
            issue_key=issue_key,
            issue_label=parsed.get("issue_label", "Unclear"),
            mood_emoji=parsed.get("mood_emoji", "🤔"),
            confidence=float(parsed.get("confidence", 0.3)),
            summary=parsed.get("summary", ""),
            fix_steps=parsed.get("fix_steps", []),
            manual_check_recommended=bool(parsed.get("manual_check_recommended", False)),
            supporting_photo_count=len(images),
            total_photo_count=len(images),
            agreement_ratio=1.0,
            plant_voice_line=parsed.get("plant_voice_line", ""),
            observations=parsed.get("observations", []),
            uncertainty_reason=parsed.get("uncertainty_reason"),
            follow_up_photo_needed=bool(parsed.get("follow_up_photo_needed", False)),
            follow_up_photo_instruction=parsed.get("follow_up_photo_instruction"),
        )
        if result.follow_up_photo_needed and not result.follow_up_photo_instruction:
                        result.follow_up_photo_instruction = result.uncertainty_reason or "Please take a closer, clearer photo of the affected area."
  
signals = [SignalScore(
            signal_name="ai_vision",
            issue=issue_key,
            confidence=result.confidence,
            notes=f"AI vision diagnosis via {self._settings.ai_diagnosis_model}. "
                  f"{len(result.observations)} observation(s) noted.",
            image_index=0,
        )]
        insights = AiInsights(
            species_verification_note=parsed.get("species_verification_note"),
            variety_guess=parsed.get("variety_guess"),
            soil_type_guidance=parsed.get("soil_type_guidance"),
            bee_friendly=parsed.get("bee_friendly"),
            bee_friendly_reason=parsed.get("bee_friendly_reason"),
        )
        return result, signals, insights
