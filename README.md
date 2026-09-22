# RAG Basics — Week 08 · Mon · Learn

**Chunking, embeddings, a real vector database, and retrieval evaluation — all
run locally, all verifiable.**

This is the foundation day for RAG: it builds and evaluates the retrieval half of
a RAG pipeline on a tiny real corpus. Everything runs on your own machine with no
API key: a small local embedding model (`all-MiniLM-L6-v2`), Qdrant in-memory, and
plain NumPy. The evaluation uses Week 6's precision/recall vocabulary applied to
*retrieved chunks*.

Read the whole story top-to-bottom without opening code in
[`REPORT.md`](REPORT.md); the task → evidence mapping is in
[`EVIDENCE_REPORT.md`](EVIDENCE_REPORT.md).

---

## What it does

| Task | What it demonstrates |
|---|---|
| Chunking by hand | Fixed-size chunking (400 tokens, word-count stand-in), overlap 0 vs ~15%; a real severed sentence found **and** shown fixed by overlap |
| Cosine similarity by hand | `np.dot(a, b) / (||a||·||b||)` on 3 real embedded sentences; related pair outranks unrelated pairs |
| Qdrant collection | In-memory Qdrant (no server), 9 real sentences / 3 topics, text stored as payload next to the vector |
| Query and verify | Real query; top result is the sentence a human would pick, confirmed by inspection |
| Break semantic search | Exact-ID query (`REF-4471`) against pure vector search — honestly reported |
| Precision@3 / recall@3 | Hand-computed P@3 / R@3 on 3 known-relevance queries + one honest finding |

## Key findings

1. Overlap repairs real damage: the sentence cut in the no-overlap split stays
   intact in the ~15% overlap split.
2. Cosine behaves as advertised: related coffee pair ≈ 0.443 vs ≤ 0.097
   cross-topic, from the hand-computed NumPy formula.
3. Vector search picks the human-expected sentence top-ranked on a clear query.
4. Pure vector search has **no exact-token semantics**: a fabricated ticket ID
   returns nearest noise instead of "not found" — the case hybrid search fixes.
5. Mean P@3 = 0.33, mean R@3 = 1.00: on this tiny topic-clustered corpus recall
   is perfect while precision is capped because same-topic neighbours crowd top-3.

## Project layout

```
common.py                      shared embedding model + NumPy cosine
part_a/                        chunking + cosine-similarity scripts
  chunk_document.py            Task 1
  cosine_similarity.py         Task 2
  source_document.md           real multi-paragraph document chunked by Task 1
part_b/                        the vector database
  corpus.py                    9 base sentences (3 topics) + the REF-4471 bait
  qdrant_store.py              in-memory Qdrant store, build/search/add_bait
  setup_qdrant.py              Task 3
  query_qdrant.py              Task 4
part_c/                        hybrid-search motivation + evaluation
  break_semantic_search.py     Task 5
  precision_recall.py          Task 6
tests/test_kata.py             assert-style tests, incl. no-drift headline checks
evidence/                      captured live output for every task
REPORT.md                      the top-to-bottom story (no code reading required)
EVIDENCE_REPORT.md             task -> evidence mapping
technical_summary.md           technical write-up
self_review.md                 pre-submission checklist
verify_project.py              integrity audit (structure + scripts + tests + secrets)
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # numpy, sentence-transformers, qdrant-client, pytest
```

The embedding model downloads once from HuggingFace Hub on first use (no token
needed); after that everything is offline.

Run any task script standalone, in any order:

```powershell
python part_a/chunk_document.py
python part_a/cosine_similarity.py
python part_b/setup_qdrant.py
python part_b/query_qdrant.py
python part_c/break_semantic_search.py
python part_c/precision_recall.py
```

## Tests

```powershell
python -m pytest tests/ -v
```

Assert-style tests covering the chunking logic, the hand-computed cosine formula,
the Qdrant store (9 base → 10 with bait), the exact-ID behaviour, the P@3/R@3
formula, and a **no-drift** suite that recomputes the headline numbers in the
documentation and fails if the code stops matching the claims.

## Verification

```powershell
python verify_project.py
```

Full integrity audit: expected structure present, all six scripts exit 0, pytest
green, and no secrets committed.

## Environment

Python 3.14.6, Windows; numpy 2.4.6, sentence-transformers 5.6.0
(`all-MiniLM-L6-v2`, 384-dim), qdrant-client 1.18.0 (in-memory), pytest 9.1.0.
No hosted API used anywhere.

Note: qdrant-client 1.18 replaced `client.search()` with `client.query_points()`
and `vectors_count` with `points_count` in collection info — the store in
`part_b/qdrant_store.py` targets the current API.

## About

Week 08 Day 1 kata — foundations for retrieval-augmented generation. No
production build today; every concept here is used directly from tomorrow on.