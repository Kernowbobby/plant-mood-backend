"""
Phase 1 signal scorers. Each function inspects the raw image and votes
on how strongly it supports specific issues, via SignalScore objects.

These are v1 heuristics — thresholds are starting points, not tuned
values. Per the project brief's testing approach: run these against a
fixed 20-30 photo test set and adjust thresholds based on measured
accuracy, don't hand-tune from intuition alone.

Kept as pure functions (image bytes in, SignalScore list out) so each
one can be unit tested and toggled independently, per the "signals"
pipeline design in the brief.
"""
import cv2
import numpy as np

from app.schemas import SignalScore

# HSV hue ranges (OpenCV: H is 0-179)
_HUE_YELLOW = (20, 34)
_HUE_GREEN = (35, 85)
_HUE_BROWN_LOW = (5, 19)  # brown reads as low-sat orange/red-ish hue


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported or corrupt file.")
    # Normalize size so thresholds behave consistently across photos.
    h, w = img.shape[:2]
    target = 800
    scale = target / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _plant_mask(hsv: np.ndarray) -> np.ndarray:
    """Rough mask of plant material (greens + yellows + browns) vs background."""
    lower = np.array([_HUE_BROWN_LOW[0], 25, 20])
    upper = np.array([_HUE_GREEN[1], 255, 255])
    return cv2.inRange(hsv, lower, upper)


def leaf_colour_pattern_score(image_bytes: bytes) -> list[SignalScore]:
    """
    Votes based on proportion of yellow / brown / pale / dark-mushy
    pixels within the plant region. Covers: overwatering, underwatering,
    light_stress, nutrient_yellowing.
    """
    img = _decode(image_bytes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _plant_mask(hsv)
    plant_pixels = max(int(np.count_nonzero(mask)), 1)  # avoid div-by-zero

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    def frac(cond: np.ndarray) -> float:
        return float(np.count_nonzero(cond & (mask > 0))) / plant_pixels

    yellow_frac = frac((h >= _HUE_YELLOW[0]) & (h <= _HUE_YELLOW[1]) & (s > 60))
    brown_frac = frac((h >= _HUE_BROWN_LOW[0]) & (h <= _HUE_BROWN_LOW[1]) & (v < 160))
    pale_frac = frac((s < 40) & (v > 170))  # bleached/scorched look
    dark_mushy_frac = frac((v < 60))  # near-black soft patches

    signals: list[SignalScore] = []

    if dark_mushy_frac > 0.03:
        signals.append(SignalScore(
            signal_name="leaf_colour_pattern",
            issue="overwatering",
            confidence=min(dark_mushy_frac * 6, 0.9),
            notes=f"{dark_mushy_frac:.1%} of plant area reads as dark/mushy",
        ))

    if yellow_frac > 0.08 and dark_mushy_frac <= 0.03:
        # Diffuse yellowing without mushiness leans nutrient over overwatering.
        signals.append(SignalScore(
            signal_name="leaf_colour_pattern",
            issue="nutrient_yellowing",
            confidence=min(yellow_frac * 3, 0.85),
            notes=f"{yellow_frac:.1%} of plant area shows yellowing",
        ))

    if brown_frac > 0.05:
        signals.append(SignalScore(
            signal_name="leaf_colour_pattern",
            issue="underwatering",
            confidence=min(brown_frac * 4, 0.85),
            notes=f"{brown_frac:.1%} of plant area shows dry/crisped browning",
        ))

    if pale_frac > 0.10:
        signals.append(SignalScore(
            signal_name="leaf_colour_pattern",
            issue="light_stress",
            confidence=min(pale_frac * 3, 0.85),
            notes=f"{pale_frac:.1%} of plant area appears bleached/scorched",
        ))

    if not signals:
        signals.append(SignalScore(
            signal_name="leaf_colour_pattern",
            issue="healthy",
            confidence=0.4,
            notes="No strong colour anomaly detected",
        ))

    return signals


def leaf_texture_score(image_bytes: bytes) -> list[SignalScore]:
    """
    Votes based on fine surface texture. Covers: powdery_mildew (bright,
    low-saturation, low-variance fuzzy patches) and spider_mites (high
    local edge density in small speckled clusters).
    """
    img = _decode(image_bytes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _plant_mask(hsv)
    plant_pixels = max(int(np.count_nonzero(mask)), 1)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Powdery mildew: bright + desaturated + on top of plant material.
    powdery_cond = (s < 50) & (v > 190) & (mask > 0)
    powdery_frac = float(np.count_nonzero(powdery_cond)) / plant_pixels

    # Spider mites: fine stippling → high local variance in small
    # neighborhoods, restricted to the plant region.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    local_var = cv2.GaussianBlur(laplacian ** 2, (9, 9), 0)
    speckle_cond = (local_var > np.percentile(local_var, 92)) & (mask > 0)
    speckle_frac = float(np.count_nonzero(speckle_cond)) / plant_pixels

    signals: list[SignalScore] = []

    if powdery_frac > 0.04:
        signals.append(SignalScore(
            signal_name="leaf_texture",
            issue="powdery_mildew",
            confidence=min(powdery_frac * 5, 0.85),
            notes=f"{powdery_frac:.1%} of plant area shows a pale powdery texture",
        ))

    if speckle_frac > 0.10:
        signals.append(SignalScore(
            signal_name="leaf_texture",
            issue="spider_mites",
            confidence=min((speckle_frac - 0.08) * 4, 0.75),
            notes=f"Elevated fine speckling detected across {speckle_frac:.1%} of plant area",
        ))

    if not signals:
        signals.append(SignalScore(
            signal_name="leaf_texture",
            issue="healthy",
            confidence=0.35,
            notes="No unusual surface texture detected",
        ))

    return signals


def droop_shape_score(image_bytes: bytes) -> list[SignalScore]:
    """
    Phase 1 approximation only. True droop detection needs pose/depth
    or a reference (upright) baseline photo, neither of which we have
    from a single frame — so this returns a low-confidence, honest
    placeholder rather than a false-precision guess. Revisit if/when
    the app captures a follow-up angle or a stored "healthy" baseline.
    """
    return [SignalScore(
        signal_name="droop_shape",
        issue="unclear",
        confidence=0.15,
        notes="Droop detection from a single photo is unreliable in v1 — not weighted meaningfully",
    )]
