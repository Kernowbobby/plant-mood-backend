"""
Wraps Wikipedia's free REST summary API for a fallback description
when Perenual's structured care data isn't available (which, pending
their support, is most of the time right now). No key, no signup,
generous documented rate limits, and genuinely multilingual if ever
needed later — matching the original idea of trying a different-
language edition when the English article is thin.

Only called when care lookup came back empty — see main.py. This is
a fallback, not a replacement: Perenual's structured watering/sunlight
fields are more useful when they're actually available, this just
makes sure the screen isn't blank in the meantime.
"""
import logging
from urllib.parse import quote

import httpx

from app.schemas import WikiSummary

logger = logging.getLogger(__name__)

WIKIPEDIA_SUMMARY_BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
_TIMEOUT_SECONDS = 10.0
_MAX_EXTRACT_CHARS = 500  # keep it to a short blurb, not a full article dump


class WikipediaService:
    async def get_summary(self, species_name: str) -> WikiSummary | None:
        if not species_name or species_name == "Unknown":
            return None

        url = f"{WIKIPEDIA_SUMMARY_BASE_URL}/{quote(species_name)}"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    # No article under this exact name — not an error,
                    # just nothing to show. Common for cultivars/hybrids.
                    return None
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Wikipedia API call failed: %s", exc)
            return None
        except Exception:
            logger.exception("Unexpected error parsing Wikipedia response")
            return None

        try:
            extract = (payload.get("extract") or "").strip()
            if not extract:
                return None
            if len(extract) > _MAX_EXTRACT_CHARS:
                extract = extract[:_MAX_EXTRACT_CHARS].rsplit(" ", 1)[0] + "\u2026"

            page_url = ((payload.get("content_urls") or {}).get("desktop") or {}).get("page")
            return WikiSummary(extract=extract, url=page_url)
        except Exception:
            logger.exception("Unexpected shape in Wikipedia response")
            return None
