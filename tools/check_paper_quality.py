"""Mandatory paper source and structural quality gate for both HTML build targets.

This script checks verifiable source properties only. Research-level correctness,
original-figure interpretation, and runtime measurement require expert review.
"""
from __future__ import annotations

import contextlib
import io
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    records = runpy.run_path(str(ROOT / "tools" / "build_site_sources.py"))["ARTICLES"]
    check = runpy.run_path(
        str(ROOT / ".agents" / "skills" / "paper-atlas-deep-dive"
            / "scripts" / "audit_article.py")
    )["check"]
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    errors: list[str] = []
    notes: list[str] = []
    for item in records:
        rel = item["path"]
        stem = Path(rel).stem
        match = re.match(r"^([ABC]\d{3})-", stem)
        if not match:
            errors.append(f"{rel}: no registered paper identifier")
            continue
        if rel in seen_paths or match.group(1) in seen_ids:
            errors.append(f"{rel}: duplicate registered paper or identifier")
        seen_paths.add(rel)
        seen_ids.add(match.group(1))
        p = (ROOT / rel).resolve()
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            failures = check(p)
        if failures:
            errors.append(f"{rel}: " + "; ".join(failures))
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8")
        if not re.search(r"https?://", body[:3000]):
            errors.append(f"{rel}: missing traceable first-party research entry")
        if not any(w in body for w in ("Figure", "Fig.", "Table", "Algorithm", "图 1", "图1", "表 1", "表1", "原文")):
            notes.append(f"{rel}: manually verify original figure/algorithm/table evidence")
    print(f"PAPER_QUALITY_STATIC_GATE: {len(records)} registered articles examined")
    print(f"unique_papers={len(seen_paths)} unique_ids={len(seen_ids)}")
    for line in notes:
        print("RESEARCH_REVIEW_NOTE:", line)
    if errors:
        for line in errors:
            print("ERROR:", line)
        print(f"PAPER_QUALITY_STATIC_GATE=FAIL issues={len(errors)}")
        return 1
    print("PAPER_QUALITY_STATIC_GATE=PASS (structural, citation-entry and link checks only)")
    print("Scientific claims, original PDF charts and experiments still require the skill's manual evidence review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
