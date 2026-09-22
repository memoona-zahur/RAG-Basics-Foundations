# Self-review — Week 08 Day 1

## Today's tasks checklist

- [x] **Document chunked two ways** — `chunk_document.py`: 400-token fixed-size,
      overlap 0 vs 60 words (~15%) on a real multi-paragraph document.
- [x] **One real awkward cut identified and shown fixed** — "sentence or | an
      idea in half" severed in no-overlap; intact in chunk 2 of the overlap split.
- [x] **Cosine similarity computed by hand** — `cosine_similarity.py`: plain
      `np.dot / (norm*norm)` on 3 real embedded sentences; related pair (0.443)
      higher than both unrelated pairs (≤ 0.097).
- [x] **Qdrant collection created locally** — `setup_qdrant.py`: in-memory mode,
      10 real embedded sentences, text+topic payload, Qdrant confirms 10 points.
- [x] **Real query run and top result verified** — `query_qdrant.py`: cold-brew
      sentence top-ranked and confirmed by inspection, not just "code ran".
- [x] **Exact-ID query tested, result honestly reported** — `break_semantic_search.py`:
      existing ID surfaced (small-corpus luck) AND non-existent ID returned
      nearest noise with no "not found". No cherry-picking.
- [x] **Precision@3 and recall@3 computed by hand** — `precision_recall.py`, 3
      queries with hand-defined ground truth: mean P@3 0.33, mean R@3 1.00, with
      one honest finding stated from my own numbers.

## Consistency with previous weeks' repo conventions

- Same skeleton: `part_*` module areas, `evidence/` with captured live outputs,
  `EVIDENCE_REPORT.md` (task → evidence), `technical_summary.md`,
  `self_review.md`, `verify_project.py`, `requirements.txt`.
- Every task script is standalone and re-runnable; none depends on a prior
  script's process state (collections are rebuilt per run).
- No API keys / hosted calls / secrets anywhere. Local model only.

## What would make it better

- A hybrid (BM25 + vector) leg so Task 5's failure is shown *fixed* too, and a
  cross-encoder re-rank pass — both covered later this week.
- Chunking tuned against an actual retrieval metric instead of the fixed 400/15%
  starting point (that is the lesson's "tuned, not guessed" directive).