"""
Wraps GBIF's (Global Biodiversity Information Facility, gbif.org) free
species API for extra taxonomic context — kingdom/family/genus and
similar classification detail beyond what identification already
returns.

No key, no signup, and no meaningful rate limit for this kind of
light single-lookup use (GBIF's documented limits are aimed at bulk
occurrence downloads, not single species reads).

The identification step already returns a gbif_id for the top
species candidate as a free byproduct of matching against GBIF's
backbone taxonomy — so this skips straight to a direct ID lookup
rather than doing a second name search from scratch.
"""
import logging

import httpx

from app.schemas import GbifTaxonomy

logger = logging.getLogger(__name__)

GBIF_SPECIES_BASE_URL = "https://api.gbif.org/v1/species"
_TIMEOUT_SECONDS = 10.0

# Same courtesy as the Wikipedia service — identify the app rather
# than sending an anonymous request, even though GBIF doesn't
# currently enforce this the way Wikipedia does.
_USER_AGENT = "PlantMood/1.0 (https://github.com/Kernowbobby/plant-mood-backend)"


class GbifService:
    async def get_taxonomy(self, gbif_id: int | None) -> GbifTaxonomy | None:
        if not gbif_id:
            return None

        url = f"{GBIF_SPECIES_BASE_URL}/{gbif_id}"

        try:
            headers = {"User-Agent": _USER_AGENT}
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 404:
                    # gbif_id came from identification but no longer
                    # resolves — not an error, just nothing to show.
                    return None
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            logger.error("GBIF API call failed: %s", exc)
            return None
        except Exception:
            logger.exception("Unexpected error parsing GBIF response")
            return None

        try:
            return GbifTaxonomy(
                kingdom=payload.get("kingdom"),
                phylum=payload.get("phylum"),
                plant_class=payload.get("class"),
                order=payload.get("order"),
                family=payload.get("family"),
                genus=payload.get("genus"),
                canonical_name=payload.get("canonicalName"),
                rank=payload.get("rank"),
            )
        except Exception:
            logger.exception("Unexpected shape in GBIF response")
            return None
