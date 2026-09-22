# Technical Summary — Week 08 Day 1 (RAG Basics)

## What this project proves

A full retrieval pipeline on 10 real sentences, end to end: chunking a real
document, embedding with a local model, storing and searching in an in-memory
vector database, deliberately breaking semantic search with an exact-ID query,
and scoring the whole thing with precision@k / recall@k.

## Environment

- Python 3.14.6 on Windows, CPU-only.
- NumPy 2.4.6; sentence-transformers 5.6.0 with `all-MiniLM-L6-v2` (384-dim,
  ~80 MB, local — no API key anywhere).
- qdrant-client 1.18.0, **in-memory mode** (no server process). Two API surface
  changes vs older docs tripped me up and are worth knowing: `client.search()` is
  now `client.query_points()`, and `CollectionInfo.vectors_count` is now
  `points_count`.

## Approach and choices

- **Chunk size = 400 tokens** with word count as the sanctioned token stand-in,
  overlap 0 vs 60 words (~15%). The awkward cut appears exactly where the lesson
  predicts: a fixed boundary severs "sentence or | an idea in half".
- **One embedding model everywhere** (`common.py`) so vectors across parts A/B/C
  stay comparable. Cosine similarity computed by hand with NumPy, not a library.
- **Corpus designed for evaluation**: base Part B collection = 9 real sentences
  (3 coffee + 3 cycling + 3 ml). The support/REF-4471 bait sentence is kept out
  of the base set and **added literally in task 5** ("Add one sentence to your
  Part B collection"), which is why later scripts pass
  `build_collection(include_bait=True)`. Topic clusters make precision@k
  meaningfully smaller than recall@k.

## Findings

1. Overlap repairs real severing: the sentence cut by no-overlap chunking is
   intact in the overlapping split. Confirmed programmatically, not by eye.
2. Cosine behaves as advertised: related coffee pair 0.443 vs ≤ 0.097 for
   cross-topic pairs, from a hand-computed formula.
3. Vector search gets the human-picked answer top-ranked for a clear semantic
   query (cold-brew sentence, score 0.69, next two are coffee neighbours).
4. Exact-ID lookup is unreliable by construction: the existing token surfaced
   (tiny-corpus luck), but a non-existent ID returned nearest arbitrary chunks
   with no "not found" signal. This is precisely why production uses hybrid
   search.
5. Evaluation: mean P@3 = 0.33, mean R@3 = 1.00. The honest takeaway is that a
   top-k retrieval on a topic-clustered corpus always finds the needle but packs
   the slot with same-topic neighbours — precision@k is the right vocabulary for
   the "extra junk in the prompt" cost.

## Limitations

- 10 sentences is a toy corpus; HNSW/ANN speedups and the "milliseconds over
  millions" claim are not exercised here.
- Fixed-size chunking by raw word counts ignores sentence/paragraph boundaries by
  design (that is the point). Real systems would use recursive/structured
  splitting plus tuning against retrieval metrics.
- In-memory Qdrant does not persist; every script rebuilds the collection.
- No hybrid/BM25 leg and no cross-encoder re-ranking — both are this week's later
  days, not today.