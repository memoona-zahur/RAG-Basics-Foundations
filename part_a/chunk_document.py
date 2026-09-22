"""Task 1 - chunk a real document two ways, find an awkward cut, show overlap fixes it.

Fixed-size chunking by word count (word count as a token stand-in, per the kata).
Compares chunk_tokens=400 with overlap=0 vs overlap=0.15, then locates a boundary
where a sentence is severed and proves the overlapping split keeps it intact.
"""
from __future__ import annotations

import pathlib

SOURCE = pathlib.Path(__file__).resolve().parent / "source_document.md"
CHUNK_TOKENS = 400  # words as token stand-in
OVERLAP_FRAC = 0.15
SENTENCE_ENDS = (".", "!", "?")

text = SOURCE.read_text(encoding="utf-8").strip()
words = text.split()
n_words = len(words)


def load_words() -> tuple[list[str], int]:
    """Return (word list, total word count) for the source document."""
    return words, n_words


def chunk_words(word_list: list[str], size: int, overlap: int) -> list[str]:
    """Split a token list into fixed-size chunks with the given overlap."""
    stride = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(word_list):
        end = start + size
        chunks.append(" ".join(word_list[start:end]))
        if end >= len(word_list):
            break
        start += stride
    return chunks


def sequence_contained(tokens: list[str], chunk: str) -> bool:
    """True if the ordered token sequence appears contiguously inside chunk."""
    c = chunk.split()
    for i in range(len(c) - len(tokens) + 1):
        if c[i : i + len(tokens)] == tokens:
            return True
    return False


def awkward_boundaries(chunks: list[str]) -> list[tuple[int, str, str]]:
    """Yield (chunk_index, tail_note, head_note) for boundaries that sever a sentence."""
    found = []
    for i in range(1, len(chunks)):
        tail = chunks[i - 1].split()
        head = chunks[i].split()
        last_word = tail[-1]
        first_word = head[0]
        ends_sentence = last_word.endswith(SENTENCE_ENDS)
        starts_with_continuation = first_word[0].islower()
        if not ends_sentence and starts_with_continuation:
            found.append((i, f"...{' '.join(tail[-6:])}", f"{' '.join(head[:6])}..."))
    return found


def report(title: str, chunks: list[str], overlap: int) -> list[tuple[int, str, str]]:
    print(f"\n=== {title} (chunk_size={CHUNK_TOKENS} words, overlap={overlap} words) ===")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"  chunk {idx:>2} | {len(chunk.split()):>3} words | "
              f"starts '{chunk[:38]}...' | ends '...{chunk[-38:]}'")
    return awkward_boundaries(chunks)


def severed_sentence_tokens_for(chunk_before: list[str], chunk_after: list[str]) -> list[str]:
    """Return the tokens of the sentence spanning a no-overlap boundary.

    Walks back to sentence start within the previous chunk, forward to sentence
    end within the next, and combines the pieces.
    """
    tail_end = len(chunk_before)
    sentence_start = tail_end
    while sentence_start > 0 and not chunk_before[sentence_start - 1].endswith(SENTENCE_ENDS):
        sentence_start -= 1
    head_start = 0
    sentence_end = head_start
    while sentence_end < len(chunk_after) and not chunk_after[sentence_end].endswith(SENTENCE_ENDS):
        sentence_end += 1
    return chunk_before[sentence_start:] + chunk_after[: sentence_end + 1]


def main() -> None:
    no_overlap = chunk_words(words, CHUNK_TOKENS, 0)
    overlap_words = round(CHUNK_TOKENS * OVERLAP_FRAC)
    with_overlap = chunk_words(words, CHUNK_TOKENS, overlap_words)

    print(f"Source: {SOURCE.name} | total words (token stand-ins): {n_words}")
    print(f"No-overlap chunks: {len(no_overlap)} | "
          f"~15% overlap chunks: {len(with_overlap)} (overlap = {overlap_words} words)")

    awkward = report("NO OVERLAP", no_overlap, 0)
    report("WITH ~15% OVERLAP", with_overlap, overlap_words)

    if not awkward:
        raise SystemExit("No awkward boundary found - pick a different size or document.")

    print("\n--- Awkward cut found in the no-overlap version ---")
    cut_at, tail, head = awkward[0]
    print(f"Boundary after chunk {cut_at - 1}:  ...{tail}  |  {head}")
    before_words = no_overlap[cut_at - 1].split()
    after_words = no_overlap[cut_at].split()

    severed_tokens = severed_sentence_tokens_for(before_words, after_words)
    severed_sentence = " ".join(severed_tokens)
    print(f"The severed sentence:\n    {severed_sentence}")

    holds = [
        (i, chunk)
        for i, chunk in enumerate(with_overlap, start=1)
        if sequence_contained(severed_tokens, chunk)
    ]
    print(f"\nOverlap check: is the severed sentence kept intact in the ~15% overlap split?")
    if holds:
        idx, chunk = holds[0]
        print(f"  YES - intact in chunk {idx} (overlapping split).")
        print(f"  chunk {idx} contains it as: ...{chunk[:80]}...")
    else:
        print("  NO - still severed everywhere. Try more overlap.")


if __name__ == "__main__":
    main()