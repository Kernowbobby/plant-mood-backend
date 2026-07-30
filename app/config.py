"""
Central config. Everything that varies between dev/prod or is a secret
lives here, pulled from environment variables. Nothing else in the app
should call os.environ directly.
"""
import os
from functools import lru_cache


class Settings:
    # Kindwise (Plant.id) API — used for species identification.
    # Get a key at https://web.plant.id/ . If unset, the app runs in
    # MOCK mode so you can build/test the rest of the pipeline without
    # burning API credits.
    plant_id_api_key: str = os.getenv("PLANT_ID_API_KEY", "")
    plant_id_base_url: str = "https://api.plant.id/v3"

    # How many species candidates to return for user confirmation.
    plant_id_top_k: int = int(os.getenv("PLANT_ID_TOP_K", "3"))

    # Max upload size, in bytes (10 MB default).
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

    @property
    def plant_id_mock_mode(self) -> bool:
        return not self.plant_id_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
