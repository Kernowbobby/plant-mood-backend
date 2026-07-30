# Plant Mood — Backend (Phase 1)

Photo in → species ID + rule-based diagnosis out. No auth, no DB, on purpose —
this phase is about getting the core pipeline rock solid before anything
else (weather, categories, gamification) gets layered on top.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — works fine with PLANT_ID_API_KEY blank
uvicorn app.main:app --reload
```

Server starts at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

Without a `PLANT_ID_API_KEY` set, species identification runs in **mock
mode** — it returns a fixed, structurally valid set of candidates so you
can build and test the diagnosis pipeline for free. Get a real key at
https://web.plant.id/ when you're ready.

## Try it

```bash
curl -X POST "http://127.0.0.1:8000/scan" \
  -F "photos=@/path/to/leaf1.jpg;type=image/jpeg" \
  -F "photos=@/path/to/leaf2.jpg;type=image/jpeg"
```

`photos` accepts 1–4 images (repeat the `-F "photos=@..."` flag per file). Send
more than one when you want a confidence check — extra angles or extra affected
leaves that agree with each other raise confidence; ones that disagree pull it
down and get called out in the response (`agreement_ratio`,
`supporting_photo_count` / `total_photo_count`). A single photo behaves exactly
as before.

Add `?skip_id=true` to skip species ID and go straight to diagnosis
(mirrors the Android app's "skip, just diagnose" button). Species ID always
runs on the first photo only — identification doesn't benefit from multiple
angles the way diagnosis does.

## What's actually implemented

- `POST /scan` — the only real endpoint. Takes a photo, returns:
  - `identification`: top species candidates (or `null` if skipped/failed)
  - `diagnosis`: issue, mood emoji, confidence, plain-English summary, fix steps
  - `signals`: every individual scorer's vote, for transparency/debugging
- `GET /health` — confirms the server is up and whether Plant.id is live or mocked

## Diagnosis engine — how it's structured

```
app/services/image_analysis.py    → independent signal scorers (pure functions)
app/services/diagnosis_engine.py  → runs all scorers, combines via weighted vote
app/services/issue_library.py     → static data: labels, mood emoji, fix steps
app/services/plant_id_service.py  → wraps the Kindwise/Plant.id API (+ mock mode)
```

Three signals ship in Phase 1:

- **`leaf_colour_pattern_score`** — HSV proportions of yellow/brown/pale/dark-mushy
  pixels within the plant region. Covers overwatering, underwatering, light stress,
  nutrient yellowing.
- **`leaf_texture_score`** — bright/desaturated patches (powdery mildew) and local
  variance speckling (spider mites).
- **`droop_shape_score`** — intentionally a low-confidence placeholder. Real droop
  detection needs a second angle or a stored baseline photo, which Phase 1 doesn't
  have. It's wired into the pipeline now so Phase 2+ can upgrade it without
  restructuring anything else.

Per the brief: **Phase 2's `weather_modifier()` and Phase 3's `category_modifier()`
are meant to be added as new entries in the signal list in
`diagnosis_engine._run_signals()` — nothing else in this file should need to change.**

## Known limitations (by design, not oversight)

- Thresholds in `image_analysis.py` are starting points, not tuned values. Build
  the ~20-30 photo test set the brief calls for and adjust from measured accuracy.
- `droop_shape_score` is a stub — see above.
- Root rot and other below-soil-line issues are out of scope for photo analysis;
  the `manual_check_recommended` flag exists in the schema for exactly this,
  though no issue currently routes to it in Phase 1's 6-issue set.
- No persistence — nothing is logged or stored yet. That's Phase 1.5/DB work,
  intentionally deferred per the brief.

## Next steps (not built yet, on purpose)

1. Build the fixed test set (20-30 real photos, known issues) and score this
   against it — tune the thresholds in `image_analysis.py` from real numbers.
2. Add Postgres + a `scans` table to log every request for later model training.
3. Phase 2: `weather_modifier()` using Open-Meteo, keyed off device location.
