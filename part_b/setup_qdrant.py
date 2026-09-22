"""Task 3 - create a Qdrant collection locally (in-memory mode) with 8-10 sentences.

Embed 9 real sentences (3 topics), store each as a vector with its text and
topic as payload, and report collection state from Qdrant itself. The REF-4471
bait sentence is deliberately NOT here - task 5 adds it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.qdrant_store import COLLECTION, build_collection  # noqa: E402


def main() -> None:
    client, records = build_collection()  # 9 base sentences, no bait yet

    state = client.get_collection(COLLECTION)
    print(f"collection='{COLLECTION}' | mode=in-memory | "
          f"points_count={state.points_count} | distance=Cosine | dim=384\n")
    for r in records:
        print(f"  id {r['id']:>2} | topic={r['topic']:<7} | {r['text'][:70]}")

    print(f"\n{len(records)} real embedded sentences stored with text payload "
          f"(within the 8-10 kata range). Qdrant confirms {state.points_count} points.")
    assert 8 <= state.points_count <= 10
    assert state.points_count == len(records) == 9


if __name__ == "__main__":
    main()