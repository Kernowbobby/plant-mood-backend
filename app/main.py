"""
Plant Mood — Phase 1 backend.

Deliberately minimal per the project brief: one endpoint, no auth, no
DB. The goal of this phase is a rock-solid photo -> ID -> diagnosis
pipeline, tested in isolation, before anything else gets bolted on.
"""
import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import ScanResponse
from app.services.care_service import CareService
from app.services.diagnosis_engine import diagnose
from app.services.gbif_service import GbifService
from app.services.inaturalist_service import INaturalistService
from app.services.plantnet_service import PlantNetService
from app.services.wikipedia_service import WikipediaService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Mood API",
    description="Photo -> species ID -> rule-based diagnosis. Phase 1.",
    version="0.1.0",
)

# Wide open for now — an Android app in dev has no fixed origin to
# restrict to. Tighten this once there's a real client/domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

plant_id_service = PlantNetService()
care_service = CareService()
inaturalist_service = INaturalistService()
wikipedia_service = WikipediaService()
gbif_service = GbifService()


@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "plant_id_mode": "mock" if settings.plant_id_mock_mode else "live",
    }


MAX_PHOTOS_PER_SCAN = 4


@app.post("/scan", response_model=ScanResponse)
async def scan(
    photos: list[UploadFile] = File(..., description="One to four photos of the same plant/issue."),
    skip_id: bool = Query(
        default=False,
        description="If true, skip species identification and go straight to diagnosis.",
    ),
):
    settings = get_settings()

    if len(photos) < 1:
        raise HTTPException(status_code=400, detail="At least one photo is required.")
    if len(photos) > MAX_PHOTOS_PER_SCAN:
        raise HTTPException(
            status_code=400,
            detail=f"Send at most {MAX_PHOTOS_PER_SCAN} photos per scan.",
        )

    image_byte_list: list[bytes] = []
    for photo in photos:
        if photo.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(status_code=415, detail="Upload JPEG, PNG, or WEBP images.")
        image_bytes = await photo.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="One of the uploaded files is empty.")
        if len(image_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="One of the images is too large.")
        image_byte_list.append(image_bytes)

    identification = None
    if not skip_id:
        try:
            # Species ID runs on the first photo only — identification
            # doesn't benefit from multiple angles the way diagnosis does.
            identification = await plant_id_service.identify(image_byte_list[0])
        except Exception:
            logger.exception("Species identification failed; continuing without it.")
            identification = None

        if identification and identification.candidates:
            try:
                top_species = identification.candidates[0].name
                identification.care = await care_service.get_care(top_species)
            except Exception:
                logger.exception("Care lookup failed; continuing without it.")
            try:
                identification.reference_photo = await inaturalist_service.get_reference_photo(top_species)
            except Exception:
                logger.exception("Reference photo lookup failed; continuing without it.")
            if identification.care is None:
                try:
                    identification.wiki_summary = await wikipedia_service.get_summary(top_species)
                except Exception:
                    logger.exception("Wikipedia fallback lookup failed; continuing without it.")
            try:
                top_gbif_id = identification.candidates[0].gbif_id
                identification.taxonomy = await gbif_service.get_taxonomy(top_gbif_id)
            except Exception:
                logger.exception("GBIF taxonomy lookup failed; continuing without it.")

    plant_probability = identification.is_plant_probability if identification else None

    try:
        diagnosis_result, signals = diagnose(image_byte_list, is_plant_probability=plant_probability)
    except ValueError as exc:
        # Raised by image decoding in image_analysis._decode
        raise HTTPException(status_code=400, detail=str(exc))

    return ScanResponse(
        identification=identification,
        diagnosis=diagnosis_result,
        signals=signals,
    )
