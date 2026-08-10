"""
Plant Mood — Phase 1 backend.

Deliberately minimal per the project brief: one endpoint, no auth, no
DB. The goal of this phase is a rock-solid photo -> ID -> diagnosis
pipeline, tested in isolation, before anything else gets bolted on.
"""
import asyncio
import logging

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import ScanResponse
from app.services.ai_diagnosis_service import AiDiagnosisService
from app.services.care_service import CareService
from app.services.diagnosis_engine import diagnose
from app.services.gbif_service import GbifService
from app.services.inaturalist_service import INaturalistService
from app.services.plantnet_service import PlantNetService
from app.services.wikipedia_service import WikipediaService
from app.services.weather_service import WeatherService

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
ai_diagnosis_service = AiDiagnosisService()
weather_service = WeatherService()


@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "plant_id_mode": "mock" if settings.plant_id_mock_mode else "live",
    }


MAX_PHOTOS_PER_SCAN = 4


async def _fetch_care_and_wiki(top_species: str):
    """Wiki is only fetched as a fallback when care lookup comes back
    empty, so these two stay chained together — but this whole chain
    now runs in parallel with reference photo, taxonomy, and diagnosis
    rather than blocking them."""
    care = None
    wiki = None
    try:
        care = await care_service.get_care(top_species)
    except Exception:
        logger.exception("Care lookup failed; continuing without it.")
    if care is None:
        try:
            wiki = await wikipedia_service.get_summary(top_species)
        except Exception:
            logger.exception("Wikipedia fallback lookup failed; continuing without it.")
    return care, wiki


async def _fetch_reference_photo(top_species: str):
    try:
        return await inaturalist_service.get_reference_photo(top_species)
    except Exception:
        logger.exception("Reference photo lookup failed; continuing without it.")
        return None


async def _fetch_taxonomy(gbif_id):
    try:
        return await gbif_service.get_taxonomy(gbif_id)
    except Exception:
        logger.exception("GBIF taxonomy lookup failed; continuing without it.")
        return None


async def _fetch_weather(latitude: float | None, longitude: float | None):
    if latitude is None or longitude is None:
        return None
    try:
        return await weather_service.get_summary(latitude, longitude)
    except Exception:
        logger.exception("Weather lookup failed; continuing without it.")
        return None


async def _run_diagnosis(
    image_byte_list, use_ai_diagnosis, plant_probability, candidates, wiki_summary=None, weather_summary=None,
):
    if use_ai_diagnosis:
        try:
            result, signals, insights = await ai_diagnosis_service.diagnose(
                image_byte_list, candidates, wiki_summary, weather_summary,
            )
            return result, signals, insights
        except Exception:
            logger.exception("AI diagnosis failed; falling back to the rule-based engine.")
    # diagnose() is CPU-bound (OpenCV), not I/O — run it in a worker
    # thread so it doesn't block the event loop while the other
    # lookups are running concurrently on it. Rule-based has no
    # concept of ai_insights, so that slot is always None here.
    result, signals = await asyncio.to_thread(diagnose, image_byte_list, is_plant_probability=plant_probability)
    return result, signals, None


@app.post("/scan", response_model=ScanResponse)
async def scan(
    photos: list[UploadFile] = File(..., description="One to four photos of the same plant/issue."),
    skip_id: bool = Query(
        default=False,
        description="If true, skip species identification and go straight to diagnosis.",
    ),
    use_ai_diagnosis: bool = Query(
        default=False,
        description="Phase 2, testing only: use AI vision diagnosis instead of the "
                    "rule-based colour/texture engine. Defaults to mock mode unless "
                    "ANTHROPIC_API_KEY is set on the server — see ai_diagnosis_service.py.",
    ),
    latitude: float | None = Query(
        default=None,
        description="Optional. Used to fetch local weather context for the AI diagnosis prompt, "
                    "and later for hemisphere/season awareness. Ignored entirely if omitted.",
    ),
    longitude: float | None = Query(
        default=None,
        description="Optional, paired with latitude — see above.",
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

    # Started now so it runs concurrently with identification below rather
    # than adding its own sequential wait — by the time it's actually
    # needed (just before diagnosis), it's usually already finished.
    weather_task = asyncio.create_task(_fetch_weather(latitude, longitude))

    # Identification has to run first — everything else either needs
    # to know the species name/gbif_id, or needs is_plant_probability
    # for the diagnosis gate. But once it's done (or skipped), every
    # remaining lookup is independent of every other one, so they all
    # run at the same time instead of queuing up one after another.
    # On a typical scan this turns ~5 sequential network calls into 1.
    identification = None
    if not skip_id:
        try:
            identification = await plant_id_service.identify(image_byte_list[0])
        except Exception:
            logger.exception("Species identification failed; continuing without it.")
            identification = None

    plant_probability = identification.is_plant_probability if identification else None
    candidates = identification.candidates if identification else []

    weather_summary = await weather_task

    try:
        if identification and identification.candidates:
            top_species = identification.candidates[0].name
            top_gbif_id = identification.candidates[0].gbif_id

            # Wiki has to be fetched *before* diagnosis now, so the AI can
            # actually read it — this is the one part of the pipeline that
            # can no longer be fully parallel. Photo and taxonomy are
            # unaffected and still run alongside diagnosis.
            care, wiki = await _fetch_care_and_wiki(top_species)

            diagnosis_task = _run_diagnosis(
                image_byte_list, use_ai_diagnosis, plant_probability, candidates, wiki, weather_summary,
            )
            photo_task = _fetch_reference_photo(top_species)
            taxonomy_task = _fetch_taxonomy(top_gbif_id)

            diagnosis_outcome, reference_photo, taxonomy = await asyncio.gather(
                diagnosis_task, photo_task, taxonomy_task,
            )
            identification.care = care
            identification.wiki_summary = wiki
            identification.reference_photo = reference_photo
            identification.taxonomy = taxonomy
        else:
            diagnosis_task = _run_diagnosis(
                image_byte_list, use_ai_diagnosis, plant_probability, candidates, weather_summary=weather_summary,
            )
            diagnosis_outcome = await diagnosis_task
    except ValueError as exc:
        # Raised by image decoding in image_analysis._decode
        raise HTTPException(status_code=400, detail=str(exc))

    diagnosis_result, signals, ai_insights = diagnosis_outcome

    return ScanResponse(
        identification=identification,
        diagnosis=diagnosis_result,
        signals=signals,
        ai_insights=ai_insights,
    )
