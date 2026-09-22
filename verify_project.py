"""Integrity audit for the Week 08 Day 1 kata.

Verifies: expected structure exists, every task script runs clean (exit 0),
and no API keys / secrets were committed. Same role as the previous repos'
verify_project.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "common.py",
    "requirements.txt",
    "part_a/source_document.md",
    "part_a/chunk_document.py",
    "part_a/cosine_similarity.py",
    "part_b/corpus.py",
    "part_b/qdrant_store.py",
    "part_b/setup_qdrant.py",
    "part_b/query_qdrant.py",
    "part_c/break_semantic_search.py",
    "part_c/precision_recall.py",
    "EVIDENCE_REPORT.md",
    "REPORT.md",
    "technical_summary.md",
    "self_review.md",
]
TASK_SCRIPTS = [
    "part_a/chunk_document.py",
    "part_a/cosine_similarity.py",
    "part_b/setup_qdrant.py",
    "part_b/query_qdrant.py",
    "part_c/break_semantic_search.py",
    "part_c/precision_recall.py",
]
SECRET_HINTS = ("sk-", "api_key=", "HOSTED_API_KEY=", "HF_TOKEN=")


def main() -> int:
    failures = []

    print("== 1. Structure ==")
    for rel in REQUIRED_FILES:
        ok = (ROOT / rel).exists()
        print(f"  [{'ok' if ok else 'MISSING'}] {rel}")
        if not ok:
            failures.append(f"missing file: {rel}")

    print("\n== 2. Every task script runs (exit code 0) ==")
    for rel in TASK_SCRIPTS:
        proc = subprocess.run(
            [sys.executable, str(ROOT / rel)],
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        )
        ok = proc.returncode == 0
        print(f"  [{'ok' if ok else 'FAIL'}] {rel} (exit {proc.returncode})")
        if not ok:
            failures.append(f"script failed: {rel}\n{proc.stderr[-800:]}")

    print("\n== 3. No secrets committed ==")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt", ".json", ".env"}:
            if path.name == "verify_project.py":
                continue  # the scanner itself contains the hint literals
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for hint in SECRET_HINTS:
                if hint in text and path.name != ".env.example":
                    failures.append(f"possible secret '{hint}' in {path}")
    print(f"  [{'ok' if not any('possible secret' in f for f in failures) else 'FAIL'}] "
          f"scanned {sum(1 for p in ROOT.rglob('*') if p.is_file())} files")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURE(S):'}")
    for f in failures:
        print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())