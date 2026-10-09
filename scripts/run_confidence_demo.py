"""Run the full scoring pipeline (both signals -> combine -> label) on
clearly different inputs, to check the combined score moves and lands on
different labels. Calls Groq, but doesn't touch the audit log.

Usage:
    python scripts/run_confidence_demo.py              # built-in samples
    python scripts/run_confidence_demo.py a.txt b.txt  # your own text files
"""
import sys
from pathlib import Path

# Import the real pipeline from app.py so results always match the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import LABELS, combine, llm_judge_score, stylometry_score  # noqa: E402
from scripts.run_stylometry_tests import mixed_text  # noqa: E402

# Highly polished and uniform: even sentence lengths, generic transitions
polished_text = (
    "Effective time management is essential for achieving both personal and "
    "professional goals. By prioritizing tasks according to their importance, "
    "individuals can allocate their energy where it matters most. Additionally, "
    "setting clear deadlines helps maintain momentum and reduces the risk of "
    "procrastination. Regular breaks also play a crucial role in sustaining focus "
    "throughout the day. Furthermore, digital tools can streamline scheduling and "
    "provide valuable insights into daily habits. Ultimately, consistent practice "
    "of these strategies fosters a balanced and productive lifestyle that supports "
    "long-term success."
)

# Casual and irregular: fragments, asides, run-ons, very uneven sentences
casual_text = (
    "ok so the dishwasher died. Again. Third time since March, and this time it "
    "took a whole lasagna pan's worth of greasy water with it, which is now on "
    "the kitchen floor and also somehow in the hallway?? My roommate says it's "
    "the filter. It is not the filter. I checked the filter, I've checked it like "
    "nine times, there's a fork in there that I can see but can't reach and I'm "
    "not putting my hand in that. Called the landlord. Voicemail. Obviously. "
    "Anyway we're eating off paper plates til Thursday, if anyone has a shop vac "
    "I'll trade you a slightly soggy lasagna."
)


def run(name, text):
    sty = stylometry_score(text)
    llm = llm_judge_score(text)
    res = combine(sty["score"], llm["score"] if llm else None, sty["word_count"])
    print(f"--- {name} ({sty['word_count']} words) ---")
    print(f"stylometry {sty['score']:.3f}  llm_judge "
          f"{llm['score'] if llm else None}  -> ai_likelihood "
          f"{res['ai_likelihood']:.3f}, confidence {res['confidence']:.3f}, "
          f"penalties {res['penalties_applied']}")
    print(f"{res['attribution']}: {LABELS[res['attribution']]}\n")
    return res["attribution"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            run(path, Path(path).read_text(encoding="utf-8"))
        sys.exit(0)

    got = [run("Polished, uniform", polished_text),
           run("Mixed (polished but personal)", mixed_text),
           run("Casual, irregular", casual_text)]
    print(f"distinct labels: {len(set(got))} {got}")
