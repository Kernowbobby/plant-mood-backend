"""
Wraps iNaturalist (https://www.inaturalist.org/) for a real, community-
submitted reference photo of the identified species. No API key at
all — genuinely free, no signup, no quota to run out. Backed by the
California Academy of Sciences and National Geographic Society.

IMPORTANT HONESTY NOTE for future maintenance: iNaturalist's own docs
page (api.inaturalist.org/v1/docs/) blocks automated access via
robots.txt, so unlike plantnet_service.py and care_service.py, this
file was written from well-established third-party documentation
(pyinaturalist, the official API reference at inaturalist.org/api)
rather than a directly-fetched, verified response example. The parsing
below is deliberately defensive — every field access is optional and
any failure returns None rather than raising — specifically because
the exact response shape is less certain here than for the other two
providers. Watch the first few real deployments of this file more
closely than usual.

Also deliberately in scope only: a single reference photo. iNaturalist
can return whole galleries and rich taxonomic data (native range,
observation counts, etc.) — starting with one photo is the smallest
testable slice, per the project's own stated preference for
incremental, verified steps over broad speculative builds.
"""
import logging

import httpx

from app.schemas import ReferencePhoto

logger = logging.getLogger(__name__)

INATURALIST_BASE_URL = "https://api.inaturalist.org/v1"

# iNaturalist's own guidance is informal ("be respectful, ~1 req/sec") —
# there's no key to misconfigure and no documented daily cap the way
# Perenual had, which is exactly why this source was chosen first.
_TIMEOUT_SECONDS = 10.0


class INaturalistService:
    async def get_reference_photo(self, species_name: str) -> ReferencePhoto | None:
        if not species_name or species_name == "Unknown":
            return None

        url = f"{INATURALIST_BASE_URL}/taxa"
        params = {"q": species_name, "per_page": 1, "rank": "species"}

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            logger.error("iNaturalist API call failed: %s", exc)
            # Fail soft: a reference photo is a nice-to-have, never
            # something that should break the rest of a scan.
            return None
        except Exception:
            logger.exception("Unexpected error parsing iNaturalist response")
            return None

        try:
            results = payload.get("results") or []
            if not results:
                return None

            top = results[0] or {}
            photo = top.get("default_photo") or {}
            photo_url = photo.get("medium_url") or photo.get("square_url")
            if not photo_url:
                return None

            attribution = photo.get("attribution")
            return ReferencePhoto(url=photo_url, attribution=attribution)
        except Exception:
            logger.exception("Unexpected shape in iNaturalist response — see docstring note")
            return None
