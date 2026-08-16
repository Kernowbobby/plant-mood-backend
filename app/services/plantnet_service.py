"""
Wraps the Pl@ntNet species-ID API (https://my.plantnet.org/) — free of
use up to 500 identification queries/day per Pl@ntNet's own terms of
use, no card required. Chosen deliberately over paid alternatives
(Kindwise/Plant.id) for exactly that reason.

Design note: this is the ONLY place that knows about Pl@ntNet's
request/response shape. If we swap providers again later, only this
file changes — schemas.py and the diagnosis engine never see raw API
responses.
"""
import logging

import httpx

from app.config import get_settings
from app.schemas import IdentifyResponse, SpeciesCandidate

logger = logging.getLogger(__name__)

# Pl@ntNet gives us no explicit "is this even a plant" score, so we
# infer one. Three distinct situations have to stay distinguishable,
# because the diagnosis gate downstream treats them very differently:
#
#   MATCHED     Pl@ntNet returned candidates. Almost certainly a plant.
#   NO_MATCH    Pl@ntNet answered and found nothing (404). It looked and
#               came up empty — real evidence against there being a
#               plant in the frame.
#   UNAVAILABLE Pl@ntNet never answered: quota exhausted (429), bad key
#               (401), server error, timeout. This is evidence of
#               NOTHING about the photo. It must sit ABOVE the
#               not-a-plant threshold in diagnosis_engine.py, or a spent
#               daily quota would report every real plant as "not a
#               plant" and take fallback_species_guess down with it.
PLANT_PROBABILITY_MATCHED = 0.95
PLANT_PROBABILITY_NO_MATCH = 0.0
PLANT_PROBABILITY_UNAVAILABLE = 0.5

# --- Confidence banding ------------------------------------------------
# Pl@ntNet scores low even when it is right. Real numbers from Melbourne
# and Cornwall testing on 16 Aug 2026:
#
#   0.38  Solanum lycopersicum   correct
#   0.27  Ctenanthe setosa       correct
#   0.21  Salvia elegans         correct
#   0.16  Alocasia odora         genus correct, species wrong
#
# A threshold set where a percentage "feels" confident — 0.5, say — would
# have marked all four uncertain and taught testers to ignore the app.
# These bands are fitted to the observed distribution instead, and should
# be re-fitted as more labelled scans come in rather than reasoned about
# from first principles.
BAND_CONFIRMED_AT = 0.40
BAND_PROBABLE_AT = 0.20

# How many candidates to consult when asking "do these at least agree on
# a genus?". Beyond about three, Pl@ntNet's tail is noise.
GENUS_AGREEMENT_DEPTH = 3


def _genus_is_agreed(candidates: list[SpeciesCandidate]) -> str | None:
    """
    Returns the genus when the leading candidates all point at the same
    one, otherwise None.

    This is what lets the app say "Alocasia" plainly while refusing to
    commit to a species. Two candidates disagreeing about the species
    within one genus is much weaker evidence against the genus than it
    looks — often it is the same plant under two names.
    """
    considered = [c for c in candidates[:GENUS_AGREEMENT_DEPTH] if c.genus]
    if not considered:
        return None
    first = considered[0].genus
    return first if all(c.genus == first for c in considered) else None


def _build_display(candidates: list[SpeciesCandidate]) -> tuple[str, str | None, str | None, str | None]:
    """
    Decides what the app should actually put on screen.

    Returns (band, title, subtitle, note). The app renders these as given
    and makes no judgement of its own, so the thresholds and the wording
    can both be retuned here without shipping a new APK — which matters
    while testers are sideloading debug builds.
    """
    if not candidates:
        return "none", None, None, None

    top = candidates[0]
    agreed_genus = _genus_is_agreed(candidates)
    common = top.common_name

    if top.probability >= BAND_CONFIRMED_AT:
        return (
            "confirmed",
            common or top.name,
            top.name if common else None,
            "Confident match",
        )

    if top.probability >= BAND_PROBABLE_AT:
        if agreed_genus:
            note = (
                f"Likely match. {agreed_genus} is well supported — the species is the "
                "less certain part. Compare with the reference photo below."
            )
        else:
            note = "Likely match — compare with the reference photo below before relying on it."
        return "probable", common or top.name, top.name if common else None, note

    # Below the probable band. The species is not supportable. If the
    # leading candidates agree on a genus, claim only that — it is very
    # often right when the species is wrong, and genus-level care advice
    # is usually identical across the species anyway.
    if agreed_genus:
        return (
            "uncertain",
            agreed_genus,
            None,
            f"Genus only — this photo doesn't support naming the species. "
            f"Closest match was {top.name}. Compare with the reference photo below.",
        )

    return (
        "uncertain",
        common or top.name,
        top.name if common else None,
        "Uncertain — treat this as a starting point rather than an identification. "
        "Compare with the reference photo below.",
    )


class PlantNetService:
    def __init__(self):
        self.settings = get_settings()

    async def identify(self, image_bytes: bytes) -> IdentifyResponse:
        if self.settings.plant_id_mock_mode:
            return self._mock_identify()
        return await self._call_plantnet(image_bytes)

    async def _call_plantnet(self, image_bytes: bytes) -> IdentifyResponse:
        url = f"{self.settings.plantnet_base_url}/identify/all"
        params = {
            "api-key": self.settings.plantnet_api_key,
            "lang": "en",
        }
        files = {"images": ("photo.jpg", image_bytes, "image/jpeg")}
        data = {"organs": "auto"}

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, params=params, files=files, data=data)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 404:
                # Pl@ntNet answered and found no match at all. This is
                # the response a photo of a vase, a wall or a dog gets.
                logger.info("Pl@ntNet found no match for this image (404).")
                return IdentifyResponse(
                    candidates=[],
                    is_plant_probability=PLANT_PROBABILITY_NO_MATCH,
                )
            # 429 quota, 401 bad key, 5xx outage — the service failed us,
            # the photo is not implicated. Stay neutral.
            logger.error("Pl@ntNet unavailable (HTTP %s): %s", status, exc)
            return IdentifyResponse(
                candidates=[],
                is_plant_probability=PLANT_PROBABILITY_UNAVAILABLE,
            )
        except httpx.HTTPError as exc:
            # Timeouts, DNS, connection resets. Same reasoning as above:
            # tells us nothing about what's in the photo.
            logger.error("Pl@ntNet call failed before a response: %s", exc)
            return IdentifyResponse(
                candidates=[],
                is_plant_probability=PLANT_PROBABILITY_UNAVAILABLE,
            )

        results = payload.get("results", [])
        candidates = []
        for r in results[: self.settings.plant_id_top_k]:
            species = r.get("species", {}) or {}
            common_names = species.get("commonNames") or []
            genus = (species.get("genus") or {}).get("scientificNameWithoutAuthor")
            family = (species.get("family") or {}).get("scientificNameWithoutAuthor")
            gbif_id = (r.get("gbif") or {}).get("id")
            candidates.append(
                SpeciesCandidate(
                    name=species.get("scientificNameWithoutAuthor", "Unknown"),
                    common_name=common_names[0] if common_names else None,
                    probability=r.get("score", 0.0),
                    genus=genus,
                    family=family,
                    gbif_id=gbif_id,
                )
            )

        # A 200 with an empty results list is the same situation as a
        # 404: Pl@ntNet looked and found nothing.
        is_plant_probability = (
            PLANT_PROBABILITY_MATCHED if candidates else PLANT_PROBABILITY_NO_MATCH
        )

        band, title, subtitle, note = _build_display(candidates)

        return IdentifyResponse(
            candidates=candidates,
            is_plant_probability=is_plant_probability,
            confidence_band=band,
            display_title=title,
            display_subtitle=subtitle,
            display_note=note,
        )

    def _mock_identify(self) -> IdentifyResponse:
        """Lets you build/test the rest of the pipeline with no API key."""
        candidates = [
            SpeciesCandidate(
                name="Monstera deliciosa", common_name="Monstera", probability=0.82,
                genus="Monstera", family="Araceae", gbif_id=2874790,
            ),
            SpeciesCandidate(
                name="Epipremnum aureum", common_name="Pothos", probability=0.11,
                genus="Epipremnum", family="Araceae", gbif_id=2867949,
            ),
            SpeciesCandidate(
                name="Philodendron hederaceum", common_name="Philodendron", probability=0.05,
                genus="Philodendron", family="Araceae", gbif_id=2872468,
            ),
        ]
        band, title, subtitle, note = _build_display(candidates)
        return IdentifyResponse(
            candidates=candidates,
            is_plant_probability=0.97,
            confidence_band=band,
            display_title=title,
            display_subtitle=subtitle,
            display_note=note,
        )
