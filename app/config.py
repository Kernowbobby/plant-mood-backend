"""
Central config. Everything that varies between dev/prod or is a secret
lives here, pulled from environment variables. Nothing else in the app
should call os.environ directly.
"""
import os
from functools import lru_cache


class Settings:
    # Pl@ntNet API — used for species identification. Free of use up to
    # 500 identification queries per day (per Pl@ntNet's own terms of
    # use — not a trial, a genuine free tier), no card required.
    # Get a key at https://my.plantnet.org/ (free account signup).
    # If unset, the app runs in MOCK mode so you can build/test the rest
    # of the pipeline without a key at all.
    plantnet_api_key: str = os.getenv("PLANTNET_API_KEY", "")
    plantnet_base_url: str = "https://my-api.plantnet.org/v2"

    # How many species candidates to return for user confirmation.
    plant_id_top_k: int = int(os.getenv("PLANT_ID_TOP_K", "3"))

    # Max upload size, in bytes (10 MB default).
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

    @property
    def plant_id_mock_mode(self) -> bool:
        return not self.plantnet_api_key

    # Perenual API — used for care info (watering, sunlight, growth
    # cycle, pet toxicity) once a species is identified. Free tier
    # covers roughly the 3,000 most common species. Get a key at
    # https://perenual.com/docs/api . If unset, runs in MOCK mode.
    #
    # Deliberately NOT wired up: Perenual's edible_fruit/edible_leaf
    # fields. We only ever read poisonous_to_pets — see care_service.py
    # for why (foraging-safety decision, not an oversight).
    perenual_api_key: str = os.getenv("PERENUAL_API_KEY", "")
    perenual_base_url: str = "https://perenual.com/api/v2"

    @property
    def care_mock_mode(self) -> bool:
        return not self.perenual_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
