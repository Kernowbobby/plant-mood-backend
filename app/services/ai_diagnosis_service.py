"""
Phase 2 diagnosis path: sends the plant photo(s) directly to a
vision-capable Claude model instead of relying on the hand-written
colour/texture heuristics in diagnosis_engine.py. A vision model can
tell soil from a leaf, read subtle cues, and reason about ambiguous
cases the way a person looking at the photo would — the rule-based
engine fundamentally can't do this (see the soil-mistaken-for-browning
bug found and fixed in Phase 1).

MOCK MODE IS THE DEFAULT. This costs real (small) money per call once
switched on — see Settings.ai_diagnosis_mock_mode. Nothing here is
called with a real API key until that's deliberately set in Railway.
Even then, each scan costs a fraction of a cent (Haiku 4.5 pricing).

Returns the exact same (DiagnosisResult, list[SignalScore]) shape as
diagnosis_engine.diagnose(), so main.py can call either one
interchangeably without any other code needing to change.
"""
import base64
import json
import logging
import random

import httpx

from app.config import get_settings
from app.schemas import DiagnosisResult, SignalScore

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_TIMEOUT_SECONDS = 30.0

# Keep this list in sync with issue_library.py's keys, so whatever the
# model picks always has a matching entry for label/emoji/fix steps if
# main.py chooses to look one up. Not strictly required — the model
# also returns its own label/summary/fix_steps directly — but keeping
# issue_key on the same vocabulary means downstream code (e.g. the
# not_a_plant gate in diagnosis_engine.py) still recognizes them.
_KNOWN_ISSUES = [
    "overwatering", "underwatering", "light_stress", "nutrient_yellowing",
    "powdery_mildew", "spider_mites", "healthy", "unclear",
]

_DIAGNOSIS_PROMPT = """You are looking at one or more photos of a houseplant or garden \
plant, taken by an amateur gardener on their phone. Diagnose its condition.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "issue_key": one of {issues},
  "issue_label": short human-readable label, e.g. "Overwatered",
  "mood_emoji": a single emoji capturing the plant's "mood",
  "confidence": a number between 0 and 1,
  "summary": one sentence explaining what you see and why,
  "fix_steps": a list of 2-3 short, concrete action items,
  "manual_check_recommended": true only if the photo is genuinely ambiguous,
  "plant_voice_line": a short, first-person, slightly humorous line as if \
the plant itself were speaking about how it feels

Be honest about uncertainty — use "unclear" with modest confidence rather \
than a confident-sounding guess if the photo doesn't show enough to tell. \
If the photo doesn't contain a plant at all, use issue_key "unclear" and \
say so plainly in the summary.""".format(issues=_KNOWN_ISSUES)


class AiDiagnosisService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def diagnose(
        self,
        images: list[bytes],
    ) -> tuple[DiagnosisResult, list[SignalScore]]:
        if not images:
            raise ValueError("At least one photo is required.")

        if self._settings.ai_diagnosis_mock_mode:
            return self._mock_diagnose(images)

        try:
            return await self._real_diagnose(images)
        except Exception:
            logger.exception("AI diagnosis call failed; caller should fall back to the rule-based engine.")
            raise

    # ------------------------------------------------------------------
    # Mock mode — zero cost, zero network calls, realistic shape.
    # ------------------------------------------------------------------
    def _mock_diagnose(self, images: list[bytes]) -> tuple[DiagnosisResult, list[SignalScore]]:
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
        )
        signals = [SignalScore(
            signal_name="ai_vision_mock",
            issue="underwatering",
            confidence=0.8,
            notes="Mock mode — no real API call was made.",
            image_index=0,
        )]
        return result, signals

    # ------------------------------------------------------------------
    # Real mode — actual API call, small real cost per scan.
    # ------------------------------------------------------------------
    async def _real_diagnose(self, images: list[bytes]) -> tuple[DiagnosisResult, list[SignalScore]]:
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
        content.append({"type": "text", "text": _DIAGNOSIS_PROMPT})

        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._settings.ai_diagnosis_model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": content}],
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(_ANTHROPIC_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            payload_out = resp.json()

        raw_text = "".join(
            block.get("text", "") for block in payload_out.get("content", []) if block.get("type") == "text"
        )
        # Models sometimes wrap JSON in ```json fences despite instructions — strip defensively.
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
        )
        signals = [SignalScore(
            signal_name="ai_vision",
            issue=issue_key,
            confidence=result.confidence,
            notes=f"AI vision diagnosis via {self._settings.ai_diagnosis_model}",
            image_index=0,
        )]
        return result, signals
