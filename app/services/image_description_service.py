"""
What to say when the photo isn't a plant.

Buttervilla's presence gate catches photos with no plant in them and,
until now, answered every one of them with the same canned line from
ISSUE_LIBRARY: "this photo doesn't look like it contains a plant".
Correct, and completely uninteresting.

It is also a missed opportunity. Testers photograph a LEGO model, a
cat, a garden gate, and what comes back is a small demonstration of
what the thing can actually see. That has turned out to be one of the
most enjoyable parts of the app for the people using it, and it costs
almost nothing to do properly.

WHY THIS IS A SEPARATE SERVICE, AND NOT A BRANCH OF ai_diagnosis_service
------------------------------------------------------------------------
The presence gate exists because of a real failure: fed a photo of an
empty vase, the diagnosis model invented a jade plant, invented leggy
growth, and reported 75% confidence with "2 of 2 photos agreed". The
lesson was that a model handed a diagnosis schema will fill in the
diagnosis schema.

So this call gets its own schema, and that schema has nowhere to put a
diagnosis. There is no issue_key, no fix_steps, no species field, no
confidence in a plant's health. The model cannot diagnose a vase here
because there is no field in which to do it. That is a structural
guarantee rather than an instruction the model might drift from, and it
is the same reasoning that gates the care/wiki/taxonomy lookups on
confidence band "none": if we won't name it, we don't look it up.

It also means the gate itself is untouched. Nothing that was being
refused before is being allowed now — the refusal simply became
informative.

MODEL
-----
Deliberately a stronger model than diagnosis uses (see config.py).
Naming an unfamiliar object draws on general world knowledge rather
than close visual comparison, and this call only ever fires on photos
that contain no plant. Set IMAGE_DESCRIPTION_MODEL in Railway to change
it without a code change.

FAILURE
-------
Never worse than today. Any exception, any missing key, any mock mode,
and the caller falls back to the existing canned not_a_plant response.
A photo that isn't a plant is not worth failing a scan over.
"""
import base64
import json
import logging

import httpx

from app.config import get_settings
from app.schemas import DiagnosisResult, SignalScore

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"

# Shorter than the diagnosis timeout. This is a bonus, not the product —
# if it's slow, the canned answer is a perfectly good outcome and the
# user gets their result sooner.
_TIMEOUT_SECONDS = 45.0

_DESCRIPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {
            "type": "string",
            "description": "A short, plain headline naming the subject — three or four words at "
                            "most, e.g. 'A LEGO model', 'A tabby cat', 'A wrought-iron gate'. This "
                            "is displayed as the title of the result.",
        },
        "mood_emoji": {
            "type": "string",
            "description": "A single emoji suiting the subject of the photo. Not a plant-health "
                            "emoji — there is no plant here.",
        },
        "description": {
            "type": "string",
            "description": "Two or three sentences of warm, plain prose saying what the photo "
                            "shows. This is the entire answer the user reads.",
        },
    },
    "required": ["headline", "mood_emoji", "description"],
}

_DESCRIPTION_PROMPT = """This photo was taken in a gardening app, but it doesn't contain a plant. \
That's fine and it happens often — people test what the app can see, children point the phone at \
whatever is in front of them, and sometimes a photo is just a photo. Your job is simply to say \
what it shows.

Be warm and plain-spoken. Do not scold anyone for photographing something that isn't a plant, and \
do not ask for a better photo — they know perfectly well what they pointed the camera at. Nothing \
is being got wrong here, so nothing needs apologising for.

State what you are sure of first, and mark anything less certain as a guess in the same breath: \
"A carved wooden bench, oak by the grain — the setting looks like a walled garden, though I \
couldn't say where." Name a genuinely famous building, landmark or artwork if you recognise it. \
For anywhere ordinary — a street, a beach, a field, a back garden — describe the kind of place \
and say plainly that you can't place it, rather than naming somewhere plausible. A wrong place \
name said confidently is worse than no place name at all.

Two firm limits, and they are not negotiable:

Never identify a person. Do not name anyone, do not guess who someone is, and do not describe \
individuals in a way that would help identify them. "Two people on a beach" is the right level of \
detail about people.

Never infer or state a location for anything that looks like somebody's home, garden, yard, \
street or school, even when the architecture, planting, signage or number plates would let you. \
These photos are often taken by children and families, and where somebody lives is not ours to \
work out.

Beyond those, be interesting. What a thing is made of, roughly when it's from, what it's for, how \
it works, a bit of history behind it — all welcome. The person reading this is curious, not in \
trouble.

Where a dash is wanted, use a real em dash (—). Never write two hyphens in place of one: it is \
rendered literally in the app and reads as a typo."""


class ImageDescriptionService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def describe(self, images: list[bytes]) -> tuple[DiagnosisResult, list[SignalScore]]:
        """
        Describe a photo that contains no plant.

        Raises on any failure — including mock mode — so the caller can
        fall back to the canned not_a_plant response. Deliberately not
        returning a mock string here: a "[MOCK]" description would look
        like a bug to a tester, whereas the canned answer is a real,
        sensible answer that has been shipping for weeks.
        """
        if not images:
            raise ValueError("At least one photo is required.")
        if self._settings.ai_diagnosis_mock_mode:
            raise RuntimeError("No ANTHROPIC_API_KEY set — no description available in mock mode.")

        # One photo only. If the first frame has no plant in it, the
        # others are near-certainly the same subject, and there is
        # nothing to cross-check the way a diagnosis cross-checks
        # multiple angles of a leaf.
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(images[0]).decode("ascii"),
                },
            },
            {"type": "text", "text": _DESCRIPTION_PROMPT},
        ]

        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._settings.image_description_model,
            "max_tokens": 600,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _DESCRIPTION_SCHEMA,
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
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)

        description = (parsed.get("description") or "").strip()
        if not description:
            # An empty description is a failed call wearing a valid
            # response's clothing. Treat it as a failure.
            raise ValueError("The description model returned no description.")

        result = DiagnosisResult(
            issue_key="not_a_plant",
            issue_label=(parsed.get("headline") or "Not a plant").strip(),
            mood_emoji=parsed.get("mood_emoji") or "📷",
            # Confidence here would be a category error. The number the
            # app prints means "how sure am I of this diagnosis", and
            # there is no diagnosis. Kept at 1.0 because what is being
            # asserted — that this isn't a plant — is the one thing we
            # are certain of.
            confidence=1.0,
            summary=description,
            # Everything below is empty on purpose. No advice, no voice
            # line, no follow-up request. The description is the whole
            # answer, and an empty fix_steps is also what stops the
            # app's "What to do" heading appearing with nothing under it.
            fix_steps=[],
            manual_check_recommended=False,
            supporting_photo_count=1,
            total_photo_count=len(images),
            agreement_ratio=1.0,
            plant_voice_line="",
            observations=[],
            uncertainty_reason=None,
            follow_up_photo_needed=False,
            follow_up_photo_instruction=None,
        )

        signals = [SignalScore(
            signal_name="image_description",
            issue="not_a_plant",
            confidence=1.0,
            notes=f"No plant present. Described via {self._settings.image_description_model}.",
            image_index=0,
        )]

        return result, signals
