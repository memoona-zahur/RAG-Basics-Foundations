"""Part B - the small corpus for today's kata.

9 real sentences across 3 topics form the Part B collection (task 3).
The support-ticket sentence is NOT part of that base set: it is added
explicitly in task 5 ("Add one sentence to your Part B collection").
"""
from __future__ import annotations

BASE: list[tuple[str, str]] = [
    # coffee (topic: coffee)
    ("Espresso is a concentrated coffee brewed by forcing hot water through finely ground beans.", "coffee"),
    ("Pour-over coffee uses a paper filter and a gooseneck kettle to brew a bright, clean cup.", "coffee"),
    ("Cold brew is made by steeping coarse coffee grounds in cold water for 12 to 24 hours.", "coffee"),
    # cycling (topic: cycling)
    ("Road bikes are lightweight and built for speed on paved surfaces with drop handlebars.", "cycling"),
    ("Mountain bikes use wide, knobby tires for traction on dirt trails and rough descents.", "cycling"),
    ("A bike helmet is the single most important piece of safety equipment for a cyclist.", "cycling"),
    # machine learning / RAG (topic: ml)
    ("An embedding is a vector of numbers that captures the meaning of a piece of text.", "ml"),
    ("Vector databases index embeddings so similarity search over millions of vectors returns in milliseconds.", "ml"),
    ("RAG retrieves relevant document chunks and grounds the model's answer in them.", "ml"),
]

# The exact-ID bait, added to the collection only in task 5 (id 9).
BAIT_TEXT = "Ticket REF-4471 was resolved by rotating the API key."
BAIT_TOPIC = "support"
BAIT_ID = 9