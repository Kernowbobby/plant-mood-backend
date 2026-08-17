"""
Phase 2 diagnosis path: sends the plant photo(s) directly to a
vision-capable Claude model instead of relying on the hand-written
colour/texture heuristics in diagnosis_engine.py.

Uses Anthropic's Structured Outputs feature (output_config.format,
generally available on Haiku 4.5 and later) rather than asking for
JSON in free text and hoping — the API constrains generation so the
response is guaranteed to match the schema below. No more stripping
fenced code markers or catching json.loads failures.

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
        "fix_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete, actionable steps to address the issue, favouring organic/natural "
                            "methods throughout — e.g. neem oil, insecticidal soap, companion planting, "
                            "beneficial insects, cultural controls (better airflow, watering habits, "
                            "pruning), and organic feeds (compost, seaweed or fish emulsion, blood/fish/"
                            "bone) rather than synthetic pesticides, fungicides, or chemical NPK "
                            "fertilizers. Only mention a synthetic option if the situation is severe "
                            "enough that organic methods genuinely won't be sufficient, and even then "
                            "note it as a last resort rather than the default recommendation.",
        },
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
        "fallback_species_guess": {
            "type": ["string", "null"],
            "description": "ONLY fill this in when told below that no species candidates were provided "
                            "(i.e. the reference identification lookup found nothing). In that case, give "
                            "your own best-effort guess at the plant's species from the photo — common name "
                            "and, if you're confident enough, a scientific name, e.g. 'Likely a tomato plant "
                            "(Solanum lycopersicum), though this hasn't been confirmed against a reference "
                            "database.' Always phrase it as an estimate, never as a confirmed identification. "
                            "If candidates WERE provided, or the photo genuinely gives no basis for even a "
                            "rough guess, leave this null.",
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
        "weather_comment": {
            "type": ["string", "null"],
            "description": "A short, friendly, ONE-sentence comment tying today's or tomorrow's weather "
                            "(given below, if provided) to practical gardening advice for this plant — "
                            "e.g. 'Dry and mild today, good day to check the soil' or 'Rain's due "
                            "tomorrow, you can probably skip watering'. Only fill this in if weather "
                            "information was given below. Null if no weather information was provided.",
        },
        "organic_tip": {
            "type": ["string", "null"],
            "description": "One short, standout organic/natural remedy or tip specifically relevant to "
                            "this diagnosis, shown as its own highlighted card separate from fix_steps — "
                            "e.g. 'Neem oil applied weekly should clear this within a couple of weeks' or "
                            "'Companion planting with marigolds nearby can help deter this pest long-term.' "
                            "Null if there's nothing organic-specific worth calling out beyond the general "
                            "fix_steps (e.g. the plant is healthy, or the fix is purely a watering/light "
                            "adjustment with no organic-remedy angle).",
        },
        "biodynamic_tip": {
            "type": ["string", "null"],
            "description": "One short, standout biodynamic gardening suggestion specifically relevant to "
                            "this diagnosis, shown as its own highlighted card. Draw only on well-established, "
                            "widely-known biodynamic preparations and practices — e.g. stinging nettle tea "
                            "(general tonic and pest deterrent), horn silica/BD 501 (sprayed in early morning "
                            "to strengthen light response and vigour), horn manure/BD 500 (soil and root "
                            "vitality), chamomile or yarrow preparations, or biodynamic compost teas. Only "
                            "suggest one if it genuinely fits this specific issue — null if there's no "
                            "natural fit (e.g. a purely mechanical/environmental issue like direct-sun "
                            "scorch, where no preparation is relevant).",
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
        "bee_friendly_reason", "weather_comment", "organic_tip", "biodynamic_tip", "plant_voice_line",
        "fallback_species_guess",
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

Do not request a follow-up photo just because you're not 100% certain — only when a specific photo would genuinely resolve a real ambiguity. If you can identify the plant and see a clear issue, give your best diagnosis with an honest confidence level instead.

Write every field in plain prose for a gardener. Where a dash is wanted, use a real em dash (—). Never write two hyphens (--) in place of one: it is rendered literally in the app and reads as a typo."""


def _format_candidates(candidates: list[SpeciesCandidate]) -> str:
    if not candidates:
        return ("\n\nNo species candidates were provided — the separate reference identification "
                "lookup found nothing (it may have failed or hit its usage limit). Species_verification_note "
                "and variety_guess should be null in this case, since there's nothing to verify against. "
                "Instead, attempt your own best-effort species guess purely from what's visible in the "
                "photo and fill in fallback_species_guess accordingly — see that field's description.")
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


def _format_weather(weather_summary: str | None) -> str:
    if not weather_summary:
        return ""
    return ("\n\nHere is the current weather at the plant's location:\n"
            f'"{weather_summary}"\n\n'
            "If — and only if — this weather is genuinely relevant to caring for this plant right now, "
            "fill in weather_comment with one short, friendly sentence of practical advice (e.g. skip "
            "watering if rain's coming, or a note about heat stress on a hot day). Leave weather_comment "
            "null if the weather doesn't suggest anything worth saying. Judge temperature by UK gardening "
            "norms unless the location clearly indicates otherwise: for most temperate-climate plants, "
            "high 20s°C is warm, and 30°C+ is genuinely hot and worth flagging as heat stress risk — "
            "don't describe such temperatures as mild.")


def _format_season(season_context: str | None) -> str:
    if not season_context:
        return ""
    return ("\n\nSeason context: "
            f"{season_context}\n"
            "Let this inform your observations and diagnosis where relevant — the same appearance can "
            "mean different things at different times of year (e.g. sparser growth or yellowing lower "
            "leaves is often completely normal in autumn/winter dormancy, but more concerning in "
            "spring/summer active growth). Don't mention the season explicitly unless it's genuinely "
            "useful context for the diagnosis.")


class AiDiagnosisService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def diagnose(
        self,
        images: list[bytes],
        candidates: list[SpeciesCandidate] | None = None,
        wiki_summary: str | None = None,
        weather_summary: str | None = None,
        season_context: str | None = None,
    ) -> tuple[DiagnosisResult, list[SignalScore], AiInsights]:
        if not images:
            raise ValueError("At least one photo is required.")

        if self._settings.ai_diagnosis_mock_mode:
            return self._mock_diagnose(images)

        try:
            return await self._real_diagnose(images, candidates or [], wiki_summary, weather_summary, season_context)
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
            weather_comment="[MOCK] Dry and mild today — good day to check the soil.",
            organic_tip="[MOCK] Neem oil applied weekly should help clear this.",
            biodynamic_tip="[MOCK] A diluted stinging nettle tea makes a good general tonic here.",
            fallback_species_guess=None,
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
        weather_summary: str | None = None,
        season_context: str | None = None,
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
        prompt = (
            _DIAGNOSIS_PROMPT_HEADER
            + _format_candidates(candidates)
            + _format_wiki(wiki_summary)
            + _format_weather(weather_summary)
            + _format_season(season_context)
        )
        content.append({"type": "text", "text": prompt})

        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._settings.ai_diagnosis_model,
            "max_tokens": 1500,
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
            weather_comment=parsed.get("weather_comment"),
            organic_tip=parsed.get("organic_tip"),
            biodynamic_tip=parsed.get("biodynamic_tip"),
            fallback_species_guess=parsed.get("fallback_species_guess"),
        )
        return result, signals, insights
