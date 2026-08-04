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
        "voice_lines": [
            "I'm drowning down here, not thriving.",
            "Please, no more water. I'm begging you.",
            "My roots are basically going swimming at this point.",
            "I squelch when you poke my soil. That's not a good sign.",
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
        "voice_lines": [
            "I am so thirsty right now, it's not even funny.",
            "My leaves are crunchy. Leaves shouldn't be crunchy.",
            "Water. Please. Any time now.",
            "I've been rationing for days. Send help.",
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
        "voice_lines": [
            "It's a bit much over here, honestly.",
            "I did not sign up for a tanning session.",
            "Could we dim it down a notch? Just a suggestion.",
            "I'm squinting. Plants can't squint. That's how bad it is.",
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
        "voice_lines": [
            "I've picked up some... unwelcome tenants.",
            "There are things living on me that I did not invite.",
            "Send reinforcements. I'm being overrun.",
            "I'm itchy. I don't even have nerves and I'm itchy.",
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
        "voice_lines": [
            "I appear to be dusted in something. Not by choice.",
            "I feel like I've been left out in flour. Rude.",
            "Something powdery and unwanted has moved in.",
            "A little stuffy in here — could we get some air moving?",
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
        "voice_lines": [
            "I'm running a bit low on... everything, really.",
            "Feels like I've been skipping meals lately.",
            "Could use a proper feed. It's been a while.",
            "I'm surviving, not thriving. There's a difference.",
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
        "voice_lines": [
            "Honestly? Living my best life.",
            "No notes. Keep doing exactly what you're doing.",
            "I'm thriving and I'd like everyone to know it.",
            "Green, hydrated, unbothered. This is the dream.",
        ],
    },
    "not_a_plant": {
        "label": "Doesn't Look Like a Plant",
        "mood_emoji": "🖼️",
        "summary": "This photo doesn't look like it contains a plant, so a watering/health diagnosis wouldn't be reliable here.",
        "fix_steps": [
            "Try a photo focused closely on the plant itself, with good lighting.",
            "Make sure the leaves take up most of the frame.",
        ],
        "voice_lines": [
            "I don't think that's me in that photo...",
            "Not sure who that is, but it's not a plant.",
            "Try pointing the camera at an actual plant next time?",
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
        "voice_lines": [
            "I have things to say, but that photo didn't quite capture them.",
            "Get in closer next time — I've got more to show you.",
            "Hard to explain myself from that angle.",
            "Try again with a bit more light on me?",
        ],
    },
}

# Issues that photo analysis alone can't reliably confirm — per the
# brief's "what's realistically detectable" section. These get flagged
# for manual follow-up rather than stated with confidence.
MANUAL_CHECK_ISSUES = {"root_rot_suspected"}
