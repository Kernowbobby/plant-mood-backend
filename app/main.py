"""
Plant Mood — Phase 1 backend.

Deliberately minimal per the project brief: one endpoint, no auth, no
DB. The goal of this phase is a rock-solid photo -> ID -> diagnosis
pipeline, tested in isolation, before anything else gets bolted on.
"""
import asyncio
import logging

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import ScanResponse
from app.services.ai_diagnosis_service import AiDiagnosisService
from app.services.care_service import CareService
from app.services.diagnosis_engine import diagnose, NOT_A_PLANT_PROBABILITY_THRESHOLD
from app.services.gbif_service import GbifService
from app.services.inaturalist_service import INaturalistService
from app.services.plantnet_service import PlantNetService
from app.services.wikipedia_service import WikipediaService
from app.services.weather_service import WeatherService
from app.services.season_service import get_season_context
from app.services.biodynamic_service import get_biodynamic_day

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


# The long edge Anthropic's vision API scales images down to before the
# model ever sees them. Anything larger is bytes we base64-encode (which
# adds a third again), upload, and pay latency for, so that a resize can
# be done at the far end and the extra detail thrown away.
_VISION_MAX_EDGE_PX = 1568
_VISION_JPEG_QUALITY = 85


def _downscale(image_bytes: bytes) -> bytes:
    """
    Reduce a camera original to something the pipeline can actually use.

    A modern phone photograph arrives at eight or twelve megapixels and
    several megabytes. Nothing downstream wants that. The vision model
    resizes to 1568px on its own side regardless; Pl@ntNet resizes on
    theirs; and the rule-based engine already normalises to 800px before
    it looks at a single pixel. The full-size upload was pure cost — on
    the phone's connection, in the base64 payload, and in memory here.

    Returns the ORIGINAL bytes untouched if the photo is already small
    enough, so an image that needs nothing doing to it is not put through
    a needless JPEG generation and quietly degraded.

    Never raises. A photo this cannot decode is handed on exactly as it
    arrived, and whatever comes next deals with it — a resize failing is
    not a reason for a scan to fail.
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge <= _VISION_MAX_EDGE_PX:
            return image_bytes

        scale = _VISION_MAX_EDGE_PX / long_edge
        # INTER_AREA is the right filter for shrinking; it averages the
        # pixels being collapsed rather than sampling one of them, which
        # matters when the thing being diagnosed is leaf texture.
        resized = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            resized,
            [int(cv2.IMWRITE_JPEG_QUALITY), _VISION_JPEG_QUALITY],
        )
        if not ok:
            return image_bytes
        return encoded.tobytes()
    except Exception:
        logger.exception("Could not downscale an image; sending it on as uploaded.")
        return image_bytes


async def _run_diagnosis(
    image_byte_list, use_ai_diagnosis, plant_probability, candidates,
    wiki_summary=None, weather_summary=None, season_context=None,
):
    # PRESENCE GATE — runs before either diagnosis path, not just the
    # rule-based one. Previously this check lived only inside diagnose(),
    # which meant the AI path skipped it entirely: fed a photo with no
    # plant in it, the vision model would invent a species, invent a
    # diagnosis for it, and report high confidence. Confirmed in
    # Melbourne testing on 16 Aug — a photo of an empty vase came back
    # as "jade plant, leggy growth, 75%, 2 of 2 photos agreed".
    #
    # plant_probability is None only when identification was skipped
    # entirely (skip_id=true), in which case we have no basis to gate on
    # and must let the diagnosis run. See plantnet_service.py for why a
    # failed API call now scores 0.5 rather than 0.0 — an unreachable
    # Pl@ntNet must not be read as "no plant here".
    if (
        plant_probability is not None
        and plant_probability < NOT_A_PLANT_PROBABILITY_THRESHOLD
    ):
        # Delegate to the rule-based engine, which already builds the
        # not_a_plant result from ISSUE_LIBRARY. Reusing it keeps one
        # definition of that response rather than a second copy here.
        # ai_insights stays None, so no fallback_species_guess is
        # offered for something that isn't a plant.
        result, signals = await asyncio.to_thread(
            diagnose, image_byte_list, is_plant_probability=plant_probability,
        )
        return result, signals, None

    if use_ai_diagnosis:
        try:
            result, signals, insights = await ai_diagnosis_service.diagnose(
                image_byte_list, candidates, wiki_summary, weather_summary, season_context,
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
        # Shrunk once, here, so every downstream consumer gets the smaller
        # version: Pl@ntNet, the AI call, and the rule-based engine.
        image_byte_list.append(await asyncio.to_thread(_downscale, image_bytes))

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

    # Pure arithmetic on latitude + today's date -- no network call, so
    # this is computed synchronously rather than as its own task. None
    # when latitude wasn't sent, same as weather.
    season_context = get_season_context(latitude) if latitude is not None else None

    try:
        # Every lookup below is keyed off the top candidate's species
        # name or gbif_id. When confidence_band is "none" there is no
        # species being claimed — either Pl@ntNet returned nothing, or it
        # returned scattered low-scoring guesses that disagree with each
        # other (see plantnet_service._build_display). Fetching care,
        # taxonomy and a wiki summary for the top guess anyway is how a
        # harbour photograph ended up with hollyhock watering advice
        # attached to it on 16 Aug 2026. If we won't name it, we don't
        # look it up either.
        species_is_claimed = (
            identification
            and identification.candidates
            and identification.confidence_band != "none"
        )
        if species_is_claimed:
            top_species = identification.candidates[0].name
            top_gbif_id = identification.candidates[0].gbif_id

            # Photo and taxonomy don't depend on wiki/care at all -- only
            # on top_species/top_gbif_id, both already known here -- so
            # they're started immediately as real tasks rather than
            # waiting for care/wiki to finish first. Wiki is the one
            # genuine bottleneck: diagnosis needs its result, so that
            # part alone can't be made concurrent with diagnosis itself.
            photo_task = asyncio.create_task(_fetch_reference_photo(top_species))
            taxonomy_task = asyncio.create_task(_fetch_taxonomy(top_gbif_id))

            care, wiki = await _fetch_care_and_wiki(top_species)

            diagnosis_task = _run_diagnosis(
                image_byte_list, use_ai_diagnosis, plant_probability, candidates,
                wiki, weather_summary, season_context,
            )

            diagnosis_outcome, reference_photo, taxonomy = await asyncio.gather(
                diagnosis_task, photo_task, taxonomy_task,
            )
            identification.care = care
            identification.wiki_summary = wiki
            identification.reference_photo = reference_photo
            identification.taxonomy = taxonomy
        else:
            diagnosis_task = _run_diagnosis(
                image_byte_list, use_ai_diagnosis, plant_probability, candidates,
                weather_summary=weather_summary, season_context=season_context,
            )
            diagnosis_outcome = await diagnosis_task
    except ValueError as exc:
        # Raised by image decoding in image_analysis._decode
        raise HTTPException(status_code=400, detail=str(exc))

    diagnosis_result, signals, ai_insights = diagnosis_outcome

    # Pure date arithmetic, same as season_context -- no network call,
    # not location-dependent (moon's sidereal position is the same for
    # every observer on a given day), so it's cheap to attach here
    # rather than asking the AI to compute or state it. Only attached
    # when ai_insights exists, since that's the only response shape the
    # Android app currently reads this kind of extra card data from.
    if ai_insights is not None:
        ai_insights.biodynamic_day_type, ai_insights.biodynamic_day_description = get_biodynamic_day()

    return ScanResponse(
        identification=identification,
        diagnosis=diagnosis_result,
        signals=signals,
        ai_insights=ai_insights,
    )
