# Week 08 · Day 1 — RAG Basics: the whole story, top to bottom

*You can read this end to end without opening a single script or evidence file.
Everything that matters — the choices, the numbers, and the honest failures — is
written out below.*

---

## What today was about

RAG (Retrieval-Augmented Generation) fixes the problem that a model's knowledge
is frozen at training time: it can't know your private documents, and it can't be
re-trained per document. RAG keeps your data *outside* the model, searches it at
the moment a question arrives, and hands the model just the relevant pieces. Today
was the foundation day: chunking, embeddings, a vector database, and evaluating
retrieval with precision/recall. Everything below was actually run on this
machine — no code reading required, all results inline.

**Any embedding and any sentence in Parts B/C come from one model:**
`all-MiniLM-L6-v2` (384-dimensional), running locally. No API key, no cloud.
Vector store: Qdrant in-memory mode. Python 3.14.6.

---

## Part A — chunking and embeddings, by hand first

### 1. Chunking a real document two ways

I took a real multi-paragraph document (this week's own lesson text, 1,342 words
used as token stand-ins) and split it with **fixed-size chunking at 400 tokens**
(word count as the sanctioned token proxy), once with **no overlap** and once with
**~15% overlap** (60 words).

**The awkward cut.** In the no-overlap version, the boundary after chunk 1 slices
a sentence clean in two. The sentence is from the "Chunking" section:

> …Fixed-size chunking (by token or character count) is simple but can cut a
> **sentence or** | **an idea in half** at an arbitrary boundary; overlap —
> repeating a portion of one chunk's end at the start of the next — mitigates
> that…

So a fixed boundary severing mid-sentence, exactly what the lesson said would
happen. **The fix confirmed:** in the ~15% overlap split, that entire sentence
appears contiguous and intact in one chunk. I checked this programmatically, not
just by eye — the 60-word overlap carries the cut half into the next chunk and
repeats enough context to keep the thought whole.

*Chunks produced:* no-overlap → 4 chunks (400/400/400/142 words); ~15% overlap →
4 chunks (400/400/400/322 words, the overlap carrying tail context into each next
chunk).

### 2. Cosine similarity, by hand

I embedded three real sentences with the local model:

| Sentence | Topic |
|---|---|
| Cold brew coffee is made by steeping coarse grounds in cold water for 12 to 24 hours. | coffee |
| Pour-over coffee uses a paper filter and a kettle to brew a bright, clean cup. | coffee |
| Road bikes are built for speed on paved surfaces with drop handlebars. | cycling |

Then computed pairwise cosine similarity **by hand in NumPy** —
`np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`:

| Pair | Cosine similarity |
|---|---|
| coffee × coffee (**related**) | **0.4430** |
| coffee × cycling (unrelated) | 0.0972 |
| coffee × cycling (unrelated) | 0.0548 |

The two coffee sentences — same topic, different method — score far closer to
each other than either does to the cycling sentence. That's the whole idea of
embedding-as-geometry working with a ~80 MB local model: meaning ≈ direction, and
the related pair points the same way.

---

## Part B — a real vector database, end to end

### 3. The collection

I built a local Qdrant collection in **in-memory mode** (`QdrantClient(":memory:")`
— no server to run), cosine distance, 384 dimensions to match the model. Stored
**9 real sentences across 3 topics** (coffee ×3, cycling ×3, machine-learning ×3),
each as an embedding vector with the **sentence text stored as payload** next to
it, so retrieval hands back the actual text, not just numbers. Qdrant itself
confirms 9 points. (The 8–10 range is tight, and I chose 9 so there's room for
the one sentence task 5 adds.)

### 4. Querying and verifying real retrieval

I embedded a query about one topic only — *"What kind of coffee is made by
steeping ground beans in cold water?"* — and searched. Top 3:

| rank | id | score | sentence |
|---|---|---|---|
| 1 | 2 | 0.6930 | Cold brew is made by steeping coarse coffee grounds in cold water for 12 to 24 hours. |
| 2 | 0 | 0.5941 | Espresso is a concentrated coffee brewed by forcing hot water through finely ground beans. |
| 3 | 1 | 0.4126 | Pour-over coffee uses a paper filter and a gooseneck kettle to brew a bright, clean cup. |

The top result is **the sentence a human would pick** — verified by inspection,
not just "the code ran". The other two are the right kind of neighbours (both
coffee) at clearly lower similarity.

---

## Part C — hybrid search motivation and retrieval evaluation

### 5. Breaking semantic search on purpose

I added one sentence to the collection containing a **made-up ID**, then asked for
that exact ID string:

> Ticket REF-4471 was resolved by rotating the API key.

**Honest result — query `"REF-4471"`:** pure vector search *did* return the tank
sentence at rank 1. You should read that as luck, not reliability: the corpus is
10 sentences and the token string literally appears in one of them, so leaving a
single strong token fingerprint was enough. **Honest result — query `"TICK-9981"`**
(a fabricated ID that is *not* in the corpus): the system should say "no such ID",
but pure vector search silently returned the nearest arbitrary chunks (the REF-4471
sentence at rank 1, because it's the only sentence that even smells like a ticket).
That's the real failure mode: embeddings have **no notion of exact-token match**,
so exact-ID lookup is unreliable by construction — which is precisely the concrete
case hybrid search (semantic + BM25 keyword) exists to fix.

### 6. Precision@3 and recall@3 — the honest numbers

Against the full 10-sentence collection, I defined ground truth by hand (which
sentence(s) are genuinely relevant for each query), retrieved the top 3, and
computed by hand:

| Query | Relevant ids | Retrieved top-3 | P@3 | R@3 |
|---|---|---|---|---|
| How is espresso made? | {0} | [0, 1, 2] | 1/3 = 0.33 | 1/1 = 1.00 |
| Safety equipment a cyclist should wear? | {5} | [5, 3, 4] | 1/3 = 0.33 | 1/1 = 1.00 |
| What does a vector database do with embeddings? | {7} | [7, 6, 8] | 1/3 = 0.33 | 1/1 = 1.00 |

**Mean precision@3 = 0.33, mean recall@3 = 1.00.**

**One honest finding from my own numbers** — not a hypothetical: recall is
*flawless* while precision is *capped at 1/3*, because the corpus is
topic-clustered. On every query the genuinely relevant sentence makes the top-3,
but the other two same-topic sentences come along for the ride, so exactly one of
the three retrieved chunks is actually on-point. The same finding shows up in the
near-tie in query 3 (the embedding and vector-database sentences scored 0.6599 vs
0.6579) — a slightly re-worded query could flip that order and cost recall there.
So on this corpus retrieval is *reliable but wasteful*: it finds the needle every
time while packing the prompt with two irrelevant neighbours. This is why
retrieval evaluation matters in the pipeline, not as an afterthought, and why
re-ranking (and precise retrieval generally) pays off.

---

## Files and how to reproduce

One script per task — all standalone, all re-runnable:

| Task | Script | Evidence |
|---|---|---|
| 1 chunking | `part_a/chunk_document.py` | `evidence/task1_chunking.txt` |
| 2 cosine | `part_a/cosine_similarity.py` | `evidence/task2_cosine_similarity.txt` |
| 3 collection | `part_b/setup_qdrant.py` | `evidence/task3_qdrant_setup.txt` |
| 4 query | `part_b/query_qdrant.py` | `evidence/task4_query.txt` |
| 5 break | `part_c/break_semantic_search.py` | `evidence/task5_break_semantic.txt` |
| 6 p@k/r@k | `part_c/precision_recall.py` | `evidence/task6_precision_recall.txt` |

`common.py` holds the shared model and the NumPy cosine. `part_b/` holds the
corpus, the store, and the two Part B scripts. `part_c/` holds Part C. Full audit:
`python verify_project.py` (structure + all six scripts exit 0 + no secrets).

## Honest limitations

- 10 sentences is a toy corpus — the ANN/HNSW "millions of vectors in
  milliseconds" claim is *not* exercised here.
- Chunking is fixed-size by word count; it cuts mid-sentence by design (that's
  the point being demonstrated), and the 400/15% starting values would need
  tuning against real retrieval metrics in production.
- Pure vector search only — no BM25/hybrid leg and no cross-encoder re-ranking.
  Both are later this week.