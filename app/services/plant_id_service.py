"""
Wraps the Kindwise (Plant.id) species-ID API.

Design note: this is the ONLY place that knows about Plant.id's request/
response shape. If we swap providers later, only this file changes —
schemas.py and the diagnosis engine never see raw API responses.
"""
import base64
import logging

import httpx

from app.config import get_settings
from app.schemas import IdentifyResponse, SpeciesCandidate

logger = logging.getLogger(__name__)


class PlantIdService:
    def __init__(self):
        self.settings = get_settings()

    async def identify(self, image_bytes: bytes) -> IdentifyResponse:
        if self.settings.plant_id_mock_mode:
            return self._mock_identify()
        return await self._call_kindwise(image_bytes)

    async def _call_kindwise(self, image_bytes: bytes) -> IdentifyResponse:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "images": [b64_image],
            "similar_images": False,
        }
        headers = {
            "Api-Key": self.settings.plant_id_api_key,
            "Content-Type": "application/json",
        }
        url = f"{self.settings.plant_id_base_url}/identification"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Plant.id API call failed: %s", exc)
            # Fail soft: diagnosis can still run without a confirmed
            # species, just with less species-specific weighting.
            return IdentifyResponse(candidates=[], is_plant_probability=0.0)

        result = data.get("result", {})
        is_plant_prob = result.get("is_plant", {}).get("probability", 0.0)
        suggestions = result.get("classification", {}).get("suggestions", [])

        candidates = [
            SpeciesCandidate(
                name=s.get("name", "Unknown"),
                common_name=(s.get("details", {}) or {}).get("common_names", [None])[0]
                if s.get("details")
                else None,
                probability=s.get("probability", 0.0),
            )
            for s in suggestions[: self.settings.plant_id_top_k]
        ]
        return IdentifyResponse(candidates=candidates, is_plant_probability=is_plant_prob)

    def _mock_identify(self) -> IdentifyResponse:
        """Lets you build/test the rest of the pipeline with no API key."""
        return IdentifyResponse(
            candidates=[
                SpeciesCandidate(name="Monstera deliciosa", common_name="Monstera", probability=0.82),
                SpeciesCandidate(name="Epipremnum aureum", common_name="Pothos", probability=0.11),
                SpeciesCandidate(name="Philodendron hederaceum", common_name="Philodendron", probability=0.05),
            ],
            is_plant_probability=0.97,
        )
