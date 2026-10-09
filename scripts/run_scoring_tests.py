"""Check combine() against the M4 table in planning.md (no API calls).

Usage:
    python scripts/run_scoring_tests.py
"""
import sys
from pathlib import Path

# Import the real scorer from app.py so results always match the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import combine  # noqa: E402

# (stylometry, llm_judge, words, expected attribution, expected ai_likelihood,
#  expected confidence, expected penalties)
CASES = [
    (0.85, 0.90, 300, "likely_ai", 0.88, 0.88, []),
    (0.15, 0.10, 300, "likely_human", 0.12, 0.88, []),
    (0.78, 0.35, 300, "uncertain", 0.511, 0.511, ["disagreement"]),
    (0.95, 0.70, 300, "likely_ai", 0.80, 0.80, []),             # boundary
    (0.58, 0.98, 300, "uncertain", 0.82, 0.82, []),             # gap exactly 0.40
    (0.90, 0.90, 30, "uncertain", 0.70, 0.70, ["short_text"]),
    (0.90, None, 300, "uncertain", 0.70, 0.70, ["llm_unavailable"]),
    # Extra boundaries from the thresholds table
    (0.05, 0.30, 300, "likely_human", 0.20, 0.80, []),          # ai_likelihood exactly 0.20
    (0.60, 0.90, 300, "uncertain", 0.78, 0.78, []),             # both agree, blend < 0.80
    (0.40, 0.10, 300, "uncertain", 0.22, 0.78, []),             # both agree, blend > 0.20
    (0.90, None, 30, "uncertain", 0.60, 0.60, ["llm_unavailable", "short_text"]),
]

failures = 0
for sty, llm, words, attr, like, conf, pens in CASES:
    got = combine(sty, llm, words)
    ok = (got["attribution"] == attr and got["ai_likelihood"] == like
          and got["confidence"] == conf and got["penalties_applied"] == pens)
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  sty={sty} llm={llm} words={words} -> {got}")

print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
sys.exit(1 if failures else 0)
