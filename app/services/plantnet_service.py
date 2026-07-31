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
        except httpx.HTTPError as exc:
            logger.error("Pl@ntNet API call failed: %s", exc)
            # Fail soft: diagnosis can still run without a confirmed
            # species, just with less species-specific weighting.
            return IdentifyResponse(candidates=[], is_plant_probability=0.0)

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

        # Pl@ntNet doesn't return an explicit "is this even a plant"
        # probability the way Kindwise did — it just returns matches
        # (or an empty list, or a 4xx, if the image doesn't look like
        # a plant at all). Approximate: any results at all implies high
        # confidence it's a plant; none implies the opposite.
        is_plant_probability = 0.95 if candidates else 0.0

        return IdentifyResponse(candidates=candidates, is_plant_probability=is_plant_probability)

    def _mock_identify(self) -> IdentifyResponse:
        """Lets you build/test the rest of the pipeline with no API key."""
        return IdentifyResponse(
            candidates=[
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
            ],
            is_plant_probability=0.97,
        )
