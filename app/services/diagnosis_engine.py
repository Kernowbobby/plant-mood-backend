"""
Combines independent signal scorers into one diagnosis. This is the
piece the brief calls out as important to keep modular: Phase 2 adds
weather_modifier(), Phase 3 adds category_modifier() — both just get
appended to `_run_signals()`'s returned list. Nothing below the
combine step needs to change shape.
"""
import random

from app.schemas import DiagnosisResult, SignalScore
from app.services.image_analysis import (
    droop_shape_score,
    leaf_colour_pattern_score,
    leaf_texture_score,
)
from app.services.issue_library import ISSUE_LIBRARY, MANUAL_CHECK_ISSUES

# Per-signal weight in the combined vote. Colour and texture are the
# most reliable single-photo signals right now; droop is barely
# weighted until Phase 2+ gives it something to compare against.
SIGNAL_WEIGHTS = {
    "leaf_colour_pattern": 1.0,
    "leaf_texture": 1.0,
    "droop_shape": 0.2,
}

CONFIDENCE_FLOOR_FOR_DIAGNOSIS = 0.3  # below this, we call it "unclear"

# Below this, identification itself doesn't think there's a plant in
# frame — the colour/texture signals below have no way to know that,
# so they'll happily score a wall, a hand, or a rug as "overwatered"
# with high confidence. Caught a live case: a photo of framed prints
# came back "Underwatered, 85% confidence." This check runs first and
# skips signal-scoring entirely rather than letting it produce a
# confident-sounding but meaningless result.
NOT_A_PLANT_PROBABILITY_THRESHOLD = 0.4


def _run_signals(image_bytes: bytes) -> list[SignalScore]:
    signals: list[SignalScore] = []
    signals += leaf_colour_pattern_score(image_bytes)
    signals += leaf_texture_score(image_bytes)
    signals += droop_shape_score(image_bytes)
    # Phase 2: signals += weather_modifier(location, days=14)
    # Phase 3: signals += category_modifier(category)
    return signals


def _combine(signals: list[SignalScore]) -> tuple[str, float]:
    """Weighted vote per issue; returns (winning_issue, confidence)."""
    scores: dict[str, float] = {}
    weight_totals: dict[str, float] = {}

    for sig in signals:
        weight = SIGNAL_WEIGHTS.get(sig.signal_name, 1.0)
        scores[sig.issue] = scores.get(sig.issue, 0.0) + sig.confidence * weight
        weight_totals[sig.issue] = weight_totals.get(sig.issue, 0.0) + weight

    if not scores:
        return "unclear", 0.0

    # Normalize each issue's score by the weight that could have voted
    # for it, so an issue backed by one strong signal isn't automatically
    # beaten by "healthy" showing up as a low-confidence default in two.
    normalized = {issue: scores[issue] / weight_totals[issue] for issue in scores}
    best_issue = max(normalized, key=normalized.get)
    return best_issue, normalized[best_issue]


def _diagnose_single(image_bytes: bytes) -> tuple[str, float, list[SignalScore]]:
    """One photo's worth of signals, combined. Building block for both
    single- and multi-photo scans — the multi-photo path just calls this
    once per photo and reconciles the results."""
    signals = _run_signals(image_bytes)
    issue_key, confidence = _combine(signals)
    if confidence < CONFIDENCE_FLOOR_FOR_DIAGNOSIS:
        issue_key = "unclear"
    return issue_key, confidence, signals


def diagnose(
    images: list[bytes],
    is_plant_probability: float | None = None,
) -> tuple[DiagnosisResult, list[SignalScore]]:
    """
    Accepts one or more photos. Each is diagnosed independently, then
    reconciled: photos that agree on the same issue reinforce each
    other's confidence; a lone outlier photo doesn't get full credit.
    With a single photo this reduces to the old single-image behaviour
    exactly (agreement_ratio is always 1.0 with nothing to disagree with).

    is_plant_probability, when provided by the identification step, is
    checked before any signal scoring runs. Below the threshold, we
    skip straight to a "not_a_plant" result rather than let the
    colour/texture heuristics — which have no concept of "is this even
    a plant" — produce a confident-sounding but meaningless diagnosis.
    """
    if not images:
        raise ValueError("At least one photo is required.")

    if (
        is_plant_probability is not None
        and is_plant_probability < NOT_A_PLANT_PROBABILITY_THRESHOLD
    ):
        entry = ISSUE_LIBRARY["not_a_plant"]
        voice_lines = entry.get("voice_lines") or [""]
        result = DiagnosisResult(
            issue_key="not_a_plant",
            issue_label=entry["label"],
            mood_emoji=entry["mood_emoji"],
            confidence=round(1.0 - is_plant_probability, 2),
            summary=entry["summary"],
            fix_steps=entry["fix_steps"],
            manual_check_recommended=False,
            supporting_photo_count=0,
            total_photo_count=len(images),
            agreement_ratio=0.0,
            plant_voice_line=random.choice(voice_lines),
        )
        return result, []

    per_image: list[tuple[str, float, list[SignalScore]]] = []
    for idx, image_bytes in enumerate(images):
        issue_key, confidence, signals = _diagnose_single(image_bytes)
        for sig in signals:
            sig.image_index = idx
        per_image.append((issue_key, confidence, signals))

    all_signals = [sig for _, _, signals in per_image for sig in signals]

    # Group photos by which issue they individually landed on.
    votes: dict[str, list[float]] = {}
    for issue_key, confidence, _ in per_image:
        votes.setdefault(issue_key, []).append(confidence)

    # Winner = most photos agreeing; ties broken by summed confidence.
    winning_issue = max(votes, key=lambda issue: (len(votes[issue]), sum(votes[issue])))
    supporting_confidences = votes[winning_issue]
    supporting_count = len(supporting_confidences)
    total_count = len(images)
    agreement_ratio = supporting_count / total_count
    avg_confidence = sum(supporting_confidences) / supporting_count

    # Reward agreement, penalize disagreement. At agreement_ratio=1.0 this
    # is a no-op (matches single-photo behaviour); as agreement drops,
    # confidence is pulled down since the photos are telling different
    # stories. Tune the 0.7/0.3 split against the project's test set.
    if total_count > 1:
        adjusted_confidence = avg_confidence * (0.7 + 0.3 * agreement_ratio)
    else:
        adjusted_confidence = avg_confidence

    if adjusted_confidence < CONFIDENCE_FLOOR_FOR_DIAGNOSIS:
        winning_issue = "unclear"

    entry = ISSUE_LIBRARY.get(winning_issue, ISSUE_LIBRARY["unclear"])
    summary = entry["summary"]
    if total_count > 1 and agreement_ratio < 1.0:
        summary += f" ({supporting_count} of {total_count} photos support this — the rest were less clear.)"

    voice_lines = entry.get("voice_lines") or [""]
    voice_line = random.choice(voice_lines)

    result = DiagnosisResult(
        issue_key=winning_issue,
        issue_label=entry["label"],
        mood_emoji=entry["mood_emoji"],
        confidence=round(adjusted_confidence, 2),
        summary=summary,
        fix_steps=entry["fix_steps"],
        manual_check_recommended=winning_issue in MANUAL_CHECK_ISSUES,
        supporting_photo_count=supporting_count,
        total_photo_count=total_count,
        agreement_ratio=round(agreement_ratio, 2),
        plant_voice_line=voice_line,
    )
    return result, all_signals
