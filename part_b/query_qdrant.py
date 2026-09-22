"""Task 4 - query the collection and verify real retrieval.

Query for one specific coffee sentence and check the top hit is the one a
human would pick, not merely that the search runs without error.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.qdrant_store import build_collection, search  # noqa: E402

client, records = build_collection()
QUERY = "What kind of coffee is made by steeping ground beans in cold water?"
EXPECTED_ID = 2  # the cold-brew sentence

hits = search(client, QUERY, k=3)
print(f"QUERY: {QUERY}\n")
for h in hits:
    marker = "  <-- expected by hand" if h["id"] == EXPECTED_ID else ""
    print(f"  #{h['id']} score={h['score']:.4f} [{h['topic']}]{marker}")
    print(f"      {h['text']}")

top = hits[0]
print(f"\nTop result id={top['id']} (expected {EXPECTED_ID}).")
print("Human check: does the top hit answer the question?")
print(f"  -> {top['text']}")
assert top["id"] == EXPECTED_ID, "top hit is not the human-picked sentence"
print("  -> YES, verified by inspection: search surfaced the cold-brew sentence.")