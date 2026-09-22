"""Task 2 - cosine similarity by hand on 3 real embedded sentences.

Three real sentences: two about coffee, one about cycling. Embed them with the
same local model used everywhere today and compute pairwise cosine similarity
using plain NumPy: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from common import cosine, embed  # noqa: E402

SENTENCES = {
    "s1_coffee_cold_brew": "Cold brew coffee is made by steeping coarse grounds in cold water for 12 to 24 hours.",
    "s2_coffee_pourover": "Pour-over coffee uses a paper filter and a kettle to brew a bright, clean cup.",
    "s3_cycling_road": "Road bikes are built for speed on paved surfaces with drop handlebars.",
}


def main() -> None:
    texts = list(SENTENCES.values())
    vecs = embed(texts)

    print(f"model: all-MiniLM-L6-v2 | {vecs.shape[1]}-dim embeddings\n")
    for name, v in zip(SENTENCES, vecs):
        print(f"  {name:<28} norm={np.linalg.norm(v):.4f}")

    pairs = [("s1 x s2 (related: coffee)", 0, 1),
             ("s1 x s3 (unrelated)", 0, 2),
             ("s2 x s3 (unrelated)", 1, 2)]

    print("\nPairwise cosine similarity (hand-computed with NumPy):")
    for label, i, j in pairs:
        print(f"  {label:<24} = {cosine(vecs[i], vecs[j]):.4f}")

    related = cosine(vecs[0], vecs[1])
    unrelated = max(cosine(vecs[0], vecs[2]), cosine(vecs[1], vecs[2]))
    ok = related > unrelated
    print(f"\nRelated-pair similarity ({related:.4f}) > both unrelated pairs (best {unrelated:.4f})? {ok}")
    if not ok:
        raise SystemExit("Assertion failed: related pair must outscore the other pairs.")


if __name__ == "__main__":
    main()