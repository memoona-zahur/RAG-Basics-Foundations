# Week 08 · Day 1 — RAG Basics Evidence Report

**Task → evidence mapping** for all six "Today's tasks". Every claim below is
reproducible by running the named script; captured live outputs live in
`evidence/`.

| Task | Deliverable | Script | Evidence file |
|---|---|---|---|
| 1 | Document chunked two ways; awkward cut found and shown fixed | `part_a/chunk_document.py` | `evidence/task1_chunking.txt` |
| 2 | Cosine similarity by hand on 3 real embedded sentences | `part_a/cosine_similarity.py` | `evidence/task2_cosine_similarity.txt` |
| 3 | Qdrant collection created locally (in-memory) with 9 embedded sentences | `part_b/setup_qdrant.py` | `evidence/task3_qdrant_setup.txt` |
| 4 | Real query run; top result verified sensible by inspection | `part_b/query_qdrant.py` | `evidence/task4_query.txt` |
| 5 | Exact-ID query against pure vector search, honestly reported | `part_c/break_semantic_search.py` | `evidence/task5_break_semantic.txt` |
| 6 | Precision@3 / recall@3 computed by hand on 3 real queries | `part_c/precision_recall.py` | `evidence/task6_precision_recall.txt` |

Environment: Python 3.14.6, NumPy 2.4.6, sentence-transformers 5.6.0
(`all-MiniLM-L6-v2`, 384-dim), qdrant-client 1.18.0 (in-memory mode, no server).
No hosted API, no network beyond the one-time model download.

---

## Task 1 — Chunking (no overlap vs ~15% overlap)

Source: `part_a/source_document.md` (this week's lesson text, 1,342 words used as
token stand-ins). Fixed size 400 tokens, overlap 0 vs 60 (~15%).

**Awkward cut found (no-overlap):** boundary after chunk 1 splits the sentence
"…can cut a **sentence or** | **an idea in half**…" — i.e. paragraph 3's
"Fixed-size chunking … is simple but can cut a sentence or an idea in half at an
arbitrary boundary; overlap — repeating a portion of one chunk's end at the start
of the next — mitigates that…" is severed mid-thought between chunk 1 and 2.

**Overlap fix confirmed:** the same full sentence appears contiguous and intact in
chunk 2 of the ~15% overlap split. Verdict: overlap does exactly what the lesson
claims — it keeps the cut-off context alive in a neighbouring chunk.

## Task 2 — Cosine similarity by hand

Sentences: two about coffee (`cold brew`, `pour-over`), one about cycling (`road
bikes`). Embeddings from `all-MiniLM-L6-v2`. Hand-computed with
`np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`:

| Pair | Cosine |
|---|---|
| coffee × coffee (related) | **0.4430** |
| coffee × cycling (unrelated) | 0.0972 |
| coffee × cycling (unrelated) | 0.0548 |

Related pair outscores both unrelated pairs, as expected.

## Task 3 — Qdrant collection (in-memory)

`QdrantClient(":memory:")` (qdrant-client 1.18.0 — note API changes: `search` →
`query_points`, `vectors_count` → `points_count`). Collection `mixed_topics`,
distance = Cosine, dim = 384. **9 real sentences** across 3 topics (coffee ×3,
cycling ×3, ml ×3) — within the 8–10 range — stored with `text` + `topic` as
payload. Qdrant confirms 9 points. The REF-4471 sentence is deliberately **not**
here; task 5 adds it.

## Task 4 — Query and verification

Query: *"What kind of coffee is made by steeping ground beans in cold water?"*
Top hit (score 0.6930): **id 2 — "Cold brew is made by steeping coarse coffee
grounds in cold water for 12 to 24 hours."** Verified by inspection to be the
sentence a human would pick; ranks 2–3 are the other coffee sentences.

## Task 5 — Breaking semantic search (honest result)

The bait sentence is **added to a freshly built base collection**
("collection now has 10 points: 9 base + 1 bait"), then the raw ID string is
queried against pure vector search:

- `"REF-4471"` **is** in the corpus → surfaced at **rank 1** (score 0.3412).
  Honest verdict: it worked here, but this is the lucky tiny-corpus case — the
  token string literally appears in one of ten sentences.
- `"TICK-9981"` is **not** in the corpus → vector search returned nearest
  arbitrary chunks anyway (rank 1 was the REF-4471 sentence, "remotely a ticket")
  with **no ability to say "no such ID"**.

Pure vector similarity has no notion of exact-token match, so exact-ID lookup is
unreliable by construction — the concrete case hybrid search exists to fix. On a
bigger corpus of tickets the exact-match chunk would drown in semantically "close"
neighbours.

## Task 6 — Precision@3 / recall@3 (by hand)

Ground truth defined per query by hand (single relevant sentence each).

| Query | GT ids | Retrieved top-3 | P@3 | R@3 |
|---|---|---|---|---|
| How is espresso made? | {0} | [0, 1, 2] | 0.33 | 1.00 |
| Safety equipment a cyclist should wear? | {5} | [5, 3, 4] | 0.33 | 1.00 |
| What does a vector database do with embeddings? | {7} | [7, 6, 8] | 0.33 | 1.00 |

Mean P@3 = 0.33, mean R@3 = 1.00.

**Honest finding from my own numbers:** recall is flawless while precision is
capped at 1/3, because the corpus is topic-clustered — the other same-topic
sentences reliably crowd the top-3 next to the genuinely relevant one. Precision
measures the "noise in the prompt" the model actually sees; on this tiny corpus a
retrieval that always finds the needle carries two related-but-irrelevant chunks
with it. (Also visible in Q3: the embedding and vector-database sentences scored
0.6599 vs 0.6579 — nearly tied, so a slightly reworded query could flip the order
and drop recall there.)