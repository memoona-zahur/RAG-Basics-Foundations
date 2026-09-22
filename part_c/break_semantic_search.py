"""Task 5 - break semantic search on purpose.

The REF-4471 sentence is added to a freshly built base Part B collection
(A), then the raw exact ID string is queried with pure vector similarity
search (B), honestly reporting whether the exact-match chunk surfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.qdrant_store import add_bait, build_collection, search  # noqa: E402

BAIT_ID = 9
BAIT_QUERIES = ("REF-4471", "TICK-9981")


def exact_id_hits(client, records, k: int = 3) -> dict[str, list[dict]]:
    """Run the exact-ID queries against pure vector search; return {query: hits}."""
    out = {}
    for query in BAIT_QUERIES:
        out[query] = search(client, query, k=k)
    return out


def main() -> None:
    client, records = build_collection()  # base Part B collection (9 sentences)
    add_bait(client, records)              # kata: "Add one sentence to your Part B collection"

    state = client.get_collection("mixed_topics")
    print(f"(A) Added '{records[-1]['text']}'\n    -> collection now has {state.points_count} points "
          f"(9 base + 1 bait).\n")

    print("(B) Exact-ID query against PURE vector similarity search\n")

    for query, hits in exact_id_hits(client, records).items():
        exists = any(query in r["text"] for r in records)
        print(f"QUERY (exact ID token): '{query}' | exists in corpus: {exists}\n")
        for h in hits:
            marker = "  <-- the REF-4471 chunk" if h["id"] == BAIT_ID else ""
            print(f"  rank {hits.index(h) + 1} | id={h['id']} | score={h['score']:.4f} | "
                  f"[{h['topic']}]{marker}")
            print(f"        {h['text']}")

        rank_of_bait = next((i + 1 for i, h in enumerate(hits) if h["id"] == BAIT_ID), None)
        print("\n  Honest verdict:")
        if exists and rank_of_bait:
            print(f"  '{query}' is IN the corpus and pure vector search returned it at rank {rank_of_bait}.")
            print("  (Lucky tiny-corpus case: the token string literally appears in one of 10 sentences.)")
        elif exists:
            print(f"  '{query}' is IN the corpus but pure vector search did not surface it in top 3.")
        else:
            print(f"  '{query}' is NOT in the corpus. Right answer: 'no such ID'. What vector search did:")
            print("  return the nearest arbitrary chunks anyway, with no way to say 'not found'.")
        print("  Pure vector similarity has no notion of exact-token match, so exact-ID lookup")
        print("  is unreliable by construction - the case hybrid search exists to fix.\n")


if __name__ == "__main__":
    main()