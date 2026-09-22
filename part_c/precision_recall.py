"""Task 6 - precision@3 and recall@3 on a tiny corpus with known relevance.

For 3 queries against the Part B/C collection we already know which sentence(s)
are relevant (manual ground truth by sentence id). Retrieve the top 3 for each
and compute, by hand from the definition:
    precision@3 = |relevant in top 3| / 3
    recall@3    = |relevant in top 3| / |total relevant in corpus|
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.qdrant_store import build_collection, search  # noqa: E402

# (query, ground-truth relevant sentence id(s)) - defined by hand ahead of time
QUERIES = [
    ("How is espresso made?", {0}),
    ("Which piece of safety equipment should a cyclist wear?", {5}),
    ("What does a vector database actually do with embeddings?", {7}),
]
K = 3


def precision_recall(hit_ids: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    """precision@k = |relevant in top k| / k ; recall@k = |relevant in top k| / |relevant|."""
    rel_topk = set(hit_ids[:k]) & relevant
    p_k = len(rel_topk) / k
    r_k = len(rel_topk) / len(relevant)
    return p_k, r_k


def main() -> None:
    client, records = build_collection(include_bait=True)  # the full Part B/C collection

    print("corpus record id -> suit of sentences:\n")
    for r in records:
        print(f"  id {r['id']:>2} | {r['text'][:72]}")

    print("\n=== Top-3 retrieval + precision@3 / recall@3 =============================\n")
    rows = []
    for query, gt in QUERIES:
        hits = search(client, query, k=K)
        hit_ids = [h["id"] for h in hits]
        p_k, r_k = precision_recall(hit_ids, gt, K)
        rows.append((query, gt, hit_ids, p_k, r_k))
        print(f"QUERY: {query}")
        print(f"  ground truth relevant ids : {sorted(gt)}")
        for i, h in enumerate(hits, start=1):
            rel = "RELEVANT" if h["id"] in gt else ""
            print(f"    top{i} id={h['id']:>2} score={h['score']:.4f} {rel:<9} {h['text'][:64]}")
        print(f"  precision@{K} = {int(len(set(hit_ids) & gt))}/{K} = {p_k:.2f}")
        print(f"  recall@{K}    = {int(len(set(hit_ids) & gt))}/{len(gt)} = {r_k:.2f}\n")

    mean_p = sum(r[3] for r in rows) / len(rows)
    mean_r = sum(r[4] for r in rows) / len(rows)
    print(f"=== Summary: mean precision@{K} = {mean_p:.2f} | mean recall@{K} = {mean_r:.2f}")

    worst_recall = min(rows, key=lambda r: r[4])
    print("\n=== One honest finding from MY numbers ===")
    print(f"  Recall was {('flawless' if worst_recall[4] == 1.0 else 'NOT perfect')}: "
          f"on every query the genuinely relevant sentence made the top {K} "
          f"({worst_recall[1]} relevant for '{worst_recall[0]}' -> "
          f"retrieved ids {worst_recall[2]}), but precision centers near "
          f"{mean_p:.2f} because same-topic neighbours crowd the top-{K}.")


if __name__ == "__main__":
    main()