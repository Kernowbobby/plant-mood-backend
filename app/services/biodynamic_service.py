"""
Computes today's biodynamic "day type" (root / leaf / flower / fruit)
from the moon's SIDEREAL zodiac position, independently of any
published biodynamic calendar.

Why independent: Maria Thun's calendar itself is copyrighted. The
underlying method -- classing days by which of the four classical
elements (earth/water/air/fire) the moon is passing through, using
the sidereal (fixed-star) zodiac rather than the tropical (seasonal)
one -- is long-published astronomical/astrological common knowledge,
not Thun-specific IP. This module computes that from raw astronomy
(via `ephem`) rather than looking anything up from Stella Natura,
Thun's calendar, or any other published source.

Caveats, deliberately kept visible rather than hidden:
  - This uses equal 30-degree sidereal zodiac signs (a standard
    simplification), not the unequal real constellation boundaries
    Thun herself used -- so it should be treated as a reasonable
    approximation of the method, not a guaranteed match to any
    specific published biodynamic calendar for a given day.
  - The tropical -> sidereal offset (the "ayanamsa") is computed with
    a simple, widely-published linear approximation centred on the
    J2000.0 epoch. Good enough for a day-level gardening suggestion;
    not intended for precise astronomical/astrological work.

No network call, no external data file -- `ephem` computes moon
position analytically, so like season_service this is always
available and can't fail for network reasons.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import ephem

# Ayanamsa (precession offset between the tropical and sidereal
# zodiacs) at the J2000.0 epoch, in degrees, plus its drift rate.
# Widely published approximate constants (Lahiri-style), not sourced
# from any biodynamic-specific text.
_AYANAMSA_AT_J2000_DEG = 23.85
_AYANAMSA_DRIFT_DEG_PER_YEAR = 50.2564 / 3600.0  # ~50.26 arcseconds/year

# Sidereal sign -> classical element, at 30-degree intervals starting
# from sidereal 0 degrees (Aries). Long-published astrological
# convention, independent of any biodynamic calendar text.
_ELEMENT_BY_SIGN_INDEX = [
    "fire", "earth", "air", "water",  # Aries, Taurus, Gemini, Cancer
    "fire", "earth", "air", "water",  # Leo, Virgo, Libra, Scorpio
    "fire", "earth", "air", "water",  # Sagittarius, Capricorn, Aquarius, Pisces
]

_DAY_TYPE_BY_ELEMENT = {
    "fire": "fruit",
    "earth": "root",
    "air": "flower",
    "water": "leaf",
}

_DESCRIPTION_BY_DAY_TYPE = {
    "fruit": "Fruit day — a good day for working with fruiting plants and seed-bearing crops.",
    "root": "Root day — a good day for working with root crops and below-ground growth.",
    "flower": "Flower day — a good day for working with flowering plants.",
    "leaf": "Leaf day — a good day for working with leafy greens and foliage growth.",
}


def _ayanamsa_degrees(when: datetime) -> float:
    year_fraction = when.year + (when.timetuple().tm_yday / 365.25)
    return _AYANAMSA_AT_J2000_DEG + (year_fraction - 2000.0) * _AYANAMSA_DRIFT_DEG_PER_YEAR


def _moon_sidereal_longitude_degrees(when: datetime) -> float:
    observer = ephem.Observer()
    observer.date = when
    moon = ephem.Moon(observer)
    tropical_lon_deg = math.degrees(ephem.Ecliptic(moon).lon)
    sidereal_lon_deg = (tropical_lon_deg - _ayanamsa_degrees(when)) % 360.0
    return sidereal_lon_deg


def get_biodynamic_day(reference_date: date | None = None) -> tuple[str, str]:
    """
    Returns (day_type, description) for the given date (today, UTC, if
    omitted) -- day_type is one of "root"/"leaf"/"flower"/"fruit".
    Purely a function of date/time, not location: the moon's sidereal
    position is the same for every observer on Earth on a given day.
    """
    when = (
        datetime.combine(reference_date, datetime.min.time(), tzinfo=timezone.utc)
        if reference_date
        else datetime.now(timezone.utc)
    )
    sidereal_lon = _moon_sidereal_longitude_degrees(when)
    sign_index = int(sidereal_lon // 30) % 12
    element = _ELEMENT_BY_SIGN_INDEX[sign_index]
    day_type = _DAY_TYPE_BY_ELEMENT[element]
    return day_type, _DESCRIPTION_BY_DAY_TYPE[day_type]
