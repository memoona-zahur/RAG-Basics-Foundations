"""Part B - build an in-memory Qdrant collection from the corpus.

Reusable store so setup/query/break/evals all run the same way:
in-memory Qdrant (no server), cosine distance, text + topic in the payload.

The base Part B collection holds 9 sentences (3 topics). Task 5 literally
ADDS the REF-4471 bait sentence on top of a freshly built base collection
via add_bait(), which is why build_collection(include_bait=False) by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from common import embed  # noqa: E402

COLLECTION = "mixed_topics"
VECTOR_SIZE = 384


def build_collection(include_bait: bool = False) -> tuple[QdrantClient, list[dict]]:
    """Embed the base corpus (9 sentences), create the collection, optionally add the bait."""
    from part_b.corpus import BAIT_ID, BAIT_TEXT, BAIT_TOPIC, BASE

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    vectors = embed([text for text, _ in BASE])
    records = [
        {"id": i, "text": text, "topic": topic}
        for i, ((text, topic), vec) in enumerate(zip(BASE, vectors))
    ]
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=r["id"], vector=vectors[i].tolist(),
                        payload={"text": r["text"], "topic": r["topic"]})
            for i, r in enumerate(records)
        ],
    )
    if include_bait:
        _add_bait_impl(client, records)
    return client, records


def add_bait(client: QdrantClient, records: list[dict]) -> None:
    """Task 5: add the REF-4471 sentence to an existing base collection."""
    _add_bait_impl(client, records)


def _add_bait_impl(client: QdrantClient, records: list[dict]) -> None:
    from part_b.corpus import BAIT_ID, BAIT_TEXT, BAIT_TOPIC

    vec = embed([BAIT_TEXT])[0]
    if BAIT_ID not in {r["id"] for r in records}:
        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(id=BAIT_ID, vector=vec.tolist(),
                            payload={"text": BAIT_TEXT, "topic": BAIT_TOPIC})
            ],
        )
        records.append({"id": BAIT_ID, "text": BAIT_TEXT, "topic": BAIT_TOPIC})


def search(client: QdrantClient, query: str, k: int = 3) -> list[dict]:
    """Embed a query and return the top-k hits as [{id, score, text, topic}]."""
    qvec = embed([query])[0]
    results = client.query_points(
        collection_name=COLLECTION, query=qvec.tolist(), limit=k
    )
    return [
        {
            "id": h.id,
            "score": round(h.score, 4),
            "text": h.payload["text"],
            "topic": h.payload["topic"],
        }
        for h in results.points
    ]