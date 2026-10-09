"""Check how well Signal 1 (stylometry) separates labeled human and AI texts.

Prints the raw feature values and sub-scores for each group, and the AUC of
each one (chance that a random AI text scores higher than a random human
text: 1.0 = perfect, 0.5 = useless, below 0.5 = pointing the wrong way).
Use it to reset the anchors and weights in app.py.

Usage:
    python scripts/run_calibration.py HUMAN_DIR AI_DIR   # folders of .txt files
"""
import statistics
import sys
from pathlib import Path

# Import the real signal from app.py so results always match the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import _words, mattr, split_units, stylometry_score  # noqa: E402

# Pass bar from planning.md M4 step 3, applied to stylometry alone
AI_MIN = 0.6
HUMAN_MAX = 0.4


def features(text):
    counts = [n for n in (len(_words(u)) for u in split_units(text)) if n > 0]
    cv = (statistics.stdev(counts) / statistics.mean(counts)
          if len(counts) >= 2 else None)
    res = stylometry_score(text)
    return {"cv": cv, "mattr": mattr(_words(text)), **res}


def load(folder):
    return {p.name: features(p.read_text(encoding="utf-8"))
            for p in sorted(Path(folder).glob("*.txt"))}


def auc(ai_vals, human_vals):
    pairs = [(a > h) + 0.5 * (a == h) for a in ai_vals for h in human_vals]
    return sum(pairs) / len(pairs)


def column(rows, key):
    return [r[key] for r in rows.values() if r[key] is not None]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    human, ai = load(sys.argv[1]), load(sys.argv[2])

    print(f"{len(human)} human, {len(ai)} AI texts\n")
    print(f"{'':8} {'human median':>13} {'AI median':>10} {'AUC':>6}")
    for key in ["cv", "mattr", "burst", "vocab", "phrase", "score"]:
        h, a = column(human, key), column(ai, key)
        print(f"{key:8} {statistics.median(h):13.3f} "
              f"{statistics.median(a):10.3f} {auc(a, h):6.2f}")

    h, a = column(human, "score"), column(ai, "score")
    print(f"\nhuman scores <= {HUMAN_MAX}: {sum(x <= HUMAN_MAX for x in h)}/{len(h)}"
          f"  (max {max(h):.3f})")
    print(f"AI scores >= {AI_MIN}:    {sum(x >= AI_MIN for x in a)}/{len(a)}"
          f"  (min {min(a):.3f})")
    print(f"median gap: {statistics.median(a) - statistics.median(h):.3f}")
