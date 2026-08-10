"""
Wraps Open-Meteo's free, keyless forecast API (https://open-meteo.com/) to
give the AI diagnosis a little real-world weather context — current
conditions plus tomorrow's outlook, phrased in plain English so it can be
dropped straight into the diagnosis prompt.

No API key, no signup, no rate limit worth worrying about at this scale —
matches the project's existing preference for free/keyless data sources
(GBIF, iNaturalist, Wikipedia) over anything requiring an account.

Only called when the client sends latitude/longitude — see main.py. A
missing or failed weather lookup should never block a scan, so every
failure path here returns None rather than raising.
"""
import logging

import httpx

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_SECONDS = 8.0

# WMO weather interpretation codes -- the standard Open-Meteo reports in --
# collapsed down to short, plain-English phrases. See
# https://open-meteo.com/en/docs for the full table this is derived from.
_WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with frost",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


def _describe(code: int | None) -> str:
    if code is None:
        return "unknown conditions"
    return _WEATHER_CODE_DESCRIPTIONS.get(code, "unsettled conditions")


class WeatherService:
    async def get_summary(self, latitude: float, longitude: float) -> str | None:
        """
        Returns a short, plain-English weather summary suitable for
        dropping straight into the diagnosis prompt, or None if the
        lookup fails for any reason.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,precipitation,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 2,
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(_FORECAST_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Open-Meteo call failed: %s", exc)
            return None
        except Exception:
            logger.exception("Unexpected error calling Open-Meteo")
            return None

        try:
            current = payload.get("current", {})
            daily = payload.get("daily", {})

            current_temp = current.get("temperature_2m")
            current_desc = _describe(current.get("weather_code"))

            lines = []
            if current_temp is not None:
                lines.append(f"Right now: {current_desc}, {current_temp:.0f}°C.")
            else:
                lines.append(f"Right now: {current_desc}.")

            # daily[0] is today, daily[1] is tomorrow, per forecast_days=2
            tmax = daily.get("temperature_2m_max", [])
            tmin = daily.get("temperature_2m_min", [])
            codes = daily.get("weather_code", [])
            rain_chance = daily.get("precipitation_probability_max", [])

            if len(tmax) > 1 and len(tmin) > 1 and len(codes) > 1:
                tomorrow_desc = _describe(codes[1])
                line = f"Tomorrow: {tomorrow_desc}, {tmin[1]:.0f}–{tmax[1]:.0f}°C"
                if len(rain_chance) > 1 and rain_chance[1] is not None:
                    line += f", {rain_chance[1]:.0f}% chance of rain"
                lines.append(line + ".")

            return " ".join(lines)
        except Exception:
            logger.exception("Unexpected shape in Open-Meteo response")
            return None
