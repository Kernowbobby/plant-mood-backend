"""
Static data describing each detectable issue. Kept separate from the
scoring logic so copy can be edited freely, and so Phase 3 can later
swap this for DB-backed, per-category entries without touching the
diagnosis engine's control flow.
"""

ISSUE_LIBRARY: dict[str, dict] = {
    "overwatering": {
        "label": "Overwatered",
        "mood_emoji": "🥴",
        "summary": "Leaves show soft yellowing and dark, mushy patches typical of too much water.",
        "fix_steps": [
            "Check drainage — make sure the pot has holes and isn't sitting in standing water.",
            "Let the top 2-3 inches of soil dry out fully before watering again.",
            "Remove any obviously mushy or blackened leaves.",
        ],
    },
    "underwatering": {
        "label": "Underwatered",
        "mood_emoji": "🥵",
        "summary": "Crispy, browning leaf edges and a dry, curling look point to too little water.",
        "fix_steps": [
            "Water thoroughly until it drains from the bottom of the pot.",
            "Check soil moisture with a finger 2 inches down before each future watering.",
            "Trim off fully crisped leaf tips — they won't recover, but new growth will.",
        ],
    },
    "light_stress": {
        "label": "Light Stressed",
        "mood_emoji": "😵‍💫",
        "summary": "Pale, bleached, or scorched patches suggest the plant is getting more direct light than it wants.",
        "fix_steps": [
            "Move the plant a few feet back from direct sun or behind a sheer curtain.",
            "Rotate the pot weekly so growth doesn't lean toward one light source.",
            "Reintroduce brighter light gradually over 1-2 weeks if you do want to increase it.",
        ],
    },
    "spider_mites": {
        "label": "Spider Mites",
        "mood_emoji": "🕸️",
        "summary": "Fine speckling and faint webbing on the leaves are consistent with a spider mite infestation.",
        "fix_steps": [
            "Rinse leaves (top and underside) under lukewarm running water.",
            "Apply insecticidal soap or neem oil every 5-7 days for 2-3 rounds.",
            "Isolate the plant from others until mites are gone — they spread fast.",
        ],
    },
    "powdery_mildew": {
        "label": "Powdery Mildew",
        "mood_emoji": "🤧",
        "summary": "A white, flour-like coating on the leaf surface is characteristic of powdery mildew.",
        "fix_steps": [
            "Improve airflow around the plant — space it away from neighbors.",
            "Remove and discard the worst-affected leaves (don't compost them).",
            "Apply a sulfur or potassium-bicarbonate fungicide spray weekly until clear.",
        ],
    },
    "nutrient_yellowing": {
        "label": "Nutrient Deficient",
        "mood_emoji": "😪",
        "summary": "Yellowing between the veins while the veins stay green often signals a nutrient shortfall.",
        "fix_steps": [
            "Feed with a balanced liquid fertilizer at half strength.",
            "Check soil pH if this persists — nutrient lockout can look identical to deficiency.",
            "Hold off on repotting until new growth shows the yellowing has stopped.",
        ],
    },
    "healthy": {
        "label": "Looking Good",
        "mood_emoji": "🌿",
        "summary": "No strong signs of stress, pests, or disease detected.",
        "fix_steps": [
            "Keep up the current watering and light routine.",
            "Recheck in a couple of weeks, especially after any season change.",
        ],
    },
    "unclear": {
        "label": "Inconclusive",
        "mood_emoji": "🤔",
        "summary": "The photo didn't show a clear enough signal to diagnose with confidence.",
        "fix_steps": [
            "Try a closer, well-lit photo of the most affected leaves.",
            "Include both the top and underside of an affected leaf if possible.",
        ],
    },
}

# Issues that photo analysis alone can't reliably confirm — per the
# brief's "what's realistically detectable" section. These get flagged
# for manual follow-up rather than stated with confidence.
MANUAL_CHECK_ISSUES = {"root_rot_suspected"}
