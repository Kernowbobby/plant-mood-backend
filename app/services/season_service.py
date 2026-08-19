"""
Derives an approximate hemisphere and meteorological season from
latitude and the current date, purely as extra context for the AI
diagnosis prompt -- so "this plant looks a bit sparse" reads differently
in a UK winter versus a UK summer, or a Melbourne summer versus a
Melbourne winter (Robert's daughter is in Melbourne, hence hemisphere
awareness rather than assuming everyone's in the UK).

Deliberately simple and local-only: meteorological (calendar-month)
seasons rather than astronomical (solstice/equinox) ones, swapped six
months for the Southern Hemisphere. No network call, no API, nothing
that can fail -- this is pure arithmetic on numbers the client already
sends, so it's always available whenever latitude is.

THE DATE IS NOT ALWAYS TODAY. A photo imported from the gallery may
have been taken months or years ago -- a June garden visit scanned in
November is the obvious case. Reasoning about it in November's terms is
the app asserting something it has not established, so the caller may
pass the photo's own capture date and the wording changes to match:
"this photo was taken in June" rather than "it is currently June". The
distinction matters to the model, which would otherwise have no way to
tell that the plant it is looking at is not in the state the season
sentence describes.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

# Meteorological seasons for the Northern Hemisphere, keyed by month
# number. Flipped by six months for the Southern Hemisphere.
_NORTHERN_SEASON_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
_SEASON_SWAP: dict[str, str] = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}

# Within this many degrees of the equator, seasonal temperature swings
# are weak enough that "season" in the temperate sense is a much softer
# signal (wet/dry patterns matter more there) -- flagged in the context
# text rather than silently treated the same as a temperate latitude.
_TROPICS_LATITUDE_THRESHOLD = 10.0


def get_hemisphere(latitude: float) -> str:
    """"southern" for negative latitudes, "northern" otherwise (including the equator)."""
    return "southern" if latitude < 0 else "northern"


def get_season_context(latitude: float, reference_date: date | None = None) -> str:
    """
    Returns a short, plain-English sentence describing the season at
    this latitude, suitable for dropping into the diagnosis prompt.
    Always returns something -- hemisphere and month are always
    derivable from a valid latitude and a date, nothing here can fail.

    reference_date is the date the PHOTO was taken. Pass None (the
    default) when that is unknown, which is the normal case for a
    camera capture taken seconds ago. Pass the capture date for a
    gallery import, where it may be nothing like today.
    """
    today = datetime.now(timezone.utc).date()
    effective_date = reference_date or today

    hemisphere = get_hemisphere(latitude)
    season = _NORTHERN_SEASON_BY_MONTH[effective_date.month]
    if hemisphere == "southern":
        season = _SEASON_SWAP[season]

    month_name = effective_date.strftime("%B")

    # Same season, different sentence. A model told "it is currently
    # summer" will reason about a plant in its summer state; told "this
    # photo was taken in summer" it knows the plant was in that state
    # when the shutter went, which is the only thing the photo can
    # actually show. Only the month is compared, not the day: a photo
    # from earlier this month is close enough to now that the simpler
    # wording is the truer one.
    if reference_date is not None and (
        reference_date.month != today.month or reference_date.year != today.year
    ):
        sentence = (
            f"This photo was taken in {month_name} {reference_date.year}, which is "
            f"{season} in the {hemisphere} hemisphere at this location. Judge the plant by "
            f"the season it was photographed in, not by today's date."
        )
    else:
        sentence = f"It is currently {season} ({month_name}) in the {hemisphere} hemisphere at this location."

    if abs(latitude) < _TROPICS_LATITUDE_THRESHOLD:
        sentence += (
            " This location is close to the equator, so temperate-climate seasonal patterns apply only "
            "loosely here -- local wet/dry seasons likely matter more than this label."
        )
    return sentence
