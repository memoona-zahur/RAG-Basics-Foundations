"""Assert-style tests for the Week 08 Day 1 kata.

Two layers, mirroring the reference repos:
  * fast pure-logic tests (chunk math, cosine formula, precision/recall formula);
  * model-backed integration tests that pin the headline numbers in the
    evidence, so any drift between what the docs claim and what the code
    produces fails loudly ("markdown == live recompute").

Run: python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

import common
from part_a import chunk_document as cd
from part_a.cosine_similarity import SENTENCES
from part_b.corpus import BAIT_ID, BAIT_TEXT
from part_b.qdrant_store import add_bait, build_collection, search
from part_c.break_semantic_search import exact_id_hits
from part_c.precision_recall import K, QUERIES, precision_recall

# --------------------------------------------------------------------------
# Task 1 - chunking
# --------------------------------------------------------------------------


def test_chunk_words_sizes_no_overlap():
    words, n = cd.load_words()
    assert n == 1342, "source document word count must stay put"
    chunks = cd.chunk_words(words, cd.CHUNK_TOKENS, 0)
    sizes = [len(c.split()) for c in chunks]
    assert sizes[:3] == [400, 400, 400]
    assert sizes[-1] == 142  # remainder tail


def test_overlap_fraction_is_15_percent():
    assert round(cd.CHUNK_TOKENS * cd.OVERLAP_FRAC) == 60


def test_awkward_cut_found_and_fixed_by_overlap():
    words, _ = cd.load_words()
    no_overlap = cd.chunk_words(words, cd.CHUNK_TOKENS, 0)
    with_overlap = cd.chunk_words(words, cd.CHUNK_TOKENS, 60)

    awkward = cd.awkward_boundaries(no_overlap)
    assert awkward, "no-overlap version must sever at least one sentence"
    # prove the boundary is genuinely mid-sentence: the previous chunk ends on a
    # lowercase continuation and the next chunk starts on a lowercase continuation
    tail, head = awkward[0][1], awkward[0][2]
    assert tail.split()[-1][0].islower()
    assert head.split()[0][0].islower()

    cut_at = awkward[0][0]
    severed = cd.severed_sentence_tokens_for(
        no_overlap[cut_at - 1].split(), no_overlap[cut_at].split()
    )
    assert any(cd.sequence_contained(severed, c) for c in with_overlap), \
        "overlapping split must keep the severed sentence intact in >=1 chunk"


# --------------------------------------------------------------------------
# Task 2 - cosine similarity
# --------------------------------------------------------------------------


def test_cosine_formula_hand_math():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    z = np.array([-1.0, 0.0, 0.0])
    assert common.cosine(x, y) == pytest.approx(1.0)
    assert common.cosine(x, -x) == pytest.approx(-1.0)
    assert common.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_related_pair_scores_higher_and_matches_evidence():
    vecs = common.embed(list(SENTENCES.values()))
    related = common.cosine(vecs[0], vecs[1])
    unrelated = max(
        common.cosine(vecs[0], vecs[2]), common.cosine(vecs[1], vecs[2])
    )
    assert related > unrelated
    assert related == pytest.approx(0.4430, abs=0.005)  # headline number in evidence


# --------------------------------------------------------------------------
# Tasks 3 & 4 - collection and retrieval
# --------------------------------------------------------------------------


def test_base_collection_has_9_points_3_topics():
    client, records = build_collection()
    assert client.get_collection("mixed_topics").points_count == 9
    assert len(records) == 9
    assert len({r["topic"] for r in records}) == 3
    assert all("text" in r and "topic" in r for r in records)  # payload alongside vector


def test_add_bait_makes_10():
    client, records = build_collection()
    add_bait(client, records)
    assert client.get_collection("mixed_topics").points_count == 10
    assert len(records) == 10
    assert records[-1]["text"] == BAIT_TEXT and records[-1]["id"] == BAIT_ID


def test_query_top_hit_is_human_pick():
    client, _ = build_collection()
    hits = search(
        client, "What kind of coffee is made by steeping ground beans in cold water?", k=3
    )
    assert hits[0]["id"] == 2
    assert "Cold brew" in hits[0]["text"]
    assert hits[0]["topic"] == "coffee"


# --------------------------------------------------------------------------
# Task 5 - breaking semantic search with an exact ID
# --------------------------------------------------------------------------


def test_exact_id_query_surfaces_bait_in_tiny_corpus():
    client, records = build_collection()
    add_bait(client, records)
    hits = exact_id_hits(client, records)["REF-4471"]
    assert hits[0]["id"] == BAIT_ID  # honest result: it happened, on a 10-sentence corpus


def test_fabricated_id_returns_nearest_noise_with_no_not_found():
    client, records = build_collection()
    add_bait(client, records)
    hits = exact_id_hits(client, records)["TICK-9981"]
    assert hits, "search always returns something"
    # the fabricated ID appears in NO retrieved text: vectors cannot say 'no such ID'
    assert all("TICK-9981" not in h["text"] for h in hits)


# --------------------------------------------------------------------------
# Task 6 - precision@k / recall@k
# --------------------------------------------------------------------------


def test_precision_recall_formula():
    assert precision_recall([0, 1, 2], {0}, 3) == (1 / 3, 1.0)
    assert precision_recall([1, 2, 3], {0}, 3) == (0.0, 0.0)
    assert precision_recall([2, 3, 0], {0, 2}, 3) == (2 / 3, 1.0)
    assert precision_recall([4, 5, 0], {0, 4, 6}, 3) == (2 / 3, 2 / 3)


def test_headline_numbers_no_drift():
    """Evidence says P@3 = 1/3 and R@3 = 1.00 across all 3 queries. Recompute."""
    client, records = build_collection(include_bait=True)
    for query, gt in QUERIES:
        hit_ids = [h["id"] for h in search(client, query, k=K)]
        p_k, r_k = precision_recall(hit_ids, gt, K)
        assert p_k == pytest.approx(1 / 3)
        assert r_k == pytest.approx(1.0)