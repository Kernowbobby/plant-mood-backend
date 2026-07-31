"""
Wraps Perenual (https://perenual.com/) for plant care info — watering,
sunlight, growth cycle, pet toxicity — keyed off whatever species
Pl@ntNet identified. Free tier covers roughly the 3,000 most common
species.

IMPORTANT — deliberate scope limit, not an oversight:
Perenual also exposes edible_fruit / edible_leaf fields. This file
never reads them. The reasoning (agreed on explicitly, not assumed):
photo-based species ID isn't reliable enough to safely tell someone
whether a wild plant is edible — the most dangerous mix-ups are
between edible plants and toxic look-alikes in the same family, which
is exactly where a ~80%-accurate classifier is most likely to be
wrong. Pet-toxicity stays in scope because the worst case there is
low-stakes and recoverable ("call the vet"); human foraging safety
doesn't have that same margin. Do not add edibility fields here
without deliberately revisiting that reasoning first.
"""
import logging

import httpx

from app.config import get_settings
from app.schemas import CareInfo

logger = logging.getLogger(__name__)


class CareService:
    def __init__(self):
        self.settings = get_settings()

    async def get_care(self, species_name: str) -> CareInfo | None:
        if not species_name or species_name == "Unknown":
            return None
        if self.settings.care_mock_mode:
            return self._mock_care()
        return await self._call_perenual(species_name)

    async def _call_perenual(self, species_name: str) -> CareInfo | None:
        # Perenual splits this across two endpoints: species-list is a
        # lightweight search that only returns id/name — the actual
        # watering/sunlight/cycle/toxicity fields live on a separate
        # species/details/{id} call. Confirmed against Perenual's own
        # Postman documentation after the single-call version came back
        # with every care field empty despite a successful match.
        list_url = f"{self.settings.perenual_base_url}/species-list"
        list_params = {"key": self.settings.perenual_api_key, "q": species_name}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                list_resp = await client.get(list_url, params=list_params)
                list_resp.raise_for_status()
                list_payload = list_resp.json()

                results = list_payload.get("data") or []
                if not results:
                    return None
                species_id = results[0].get("id")
                if species_id is None:
                    return None

                details_url = f"{self.settings.perenual_base_url}/species/details/{species_id}"
                details_params = {"key": self.settings.perenual_api_key}
                details_resp = await client.get(details_url, params=details_params)
                details_resp.raise_for_status()
                top = details_resp.json()
        except httpx.HTTPError as exc:
            logger.error("Perenual API call failed: %s", exc)
            # Fail soft: care info is a bonus, not required for the
            # core diagnosis — a failed lookup shouldn't break the scan.
            return None

        sunlight = top.get("sunlight") or []
        if isinstance(sunlight, str):
            sunlight = [sunlight]

        poisonous_to_pets = top.get("poisonous_to_pets")
        # Perenual returns this as 0/1 (or sometimes missing). Keep it
        # as an honest three-state value: True, False, or None/unknown
        # — don't guess "not toxic" just because the field is absent.
        toxic_to_pets = bool(poisonous_to_pets) if poisonous_to_pets is not None else None

        return CareInfo(
            watering=top.get("watering"),
            sunlight=sunlight,
            cycle=top.get("cycle"),
            toxic_to_pets=toxic_to_pets,
        )

    def _mock_care(self) -> CareInfo:
        """Lets you build/test without a Perenual key."""
        return CareInfo(
            watering="Average",
            sunlight=["Part shade", "Filtered sunlight"],
            cycle="Perennial",
            toxic_to_pets=True,  # matches the real Monstera deliciosa — a fitting mock default
        )
