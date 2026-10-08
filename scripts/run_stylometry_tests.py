"""Run Signal 1 (stylometry) directly on sample texts.

Usage:
    python scripts/run_stylometry_tests.py              # built-in samples
    python scripts/run_stylometry_tests.py a.txt b.txt  # your own text files
"""
import sys
from pathlib import Path

# Import the real signal from app.py so results always match the app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import _words, mattr, stylometry_score  # noqa: E402


def print_result(name, text):
    res = stylometry_score(text)
    m = mattr(_words(text))
    print(f"--- {name} ---")
    print(f"word_count: {res['word_count']}, units: {res['units']}")
    print(f"mattr: {m:.3f}" if m is not None else "mattr: None")
    print(f"stylometry: {res}")
    print()


# AI-like sample (polished, many stock phrases, long sentences)
ai_text = (
    "In today's fast-paced world, organizations must navigate the complexities of modern "
    "communication, and this requires a clear strategy that allows teams to unlock the "
    "potential of their people. This synthesis of best practices delves into the tapestry "
    "of operational improvements, illustrating a multifaceted approach that seamlessly "
    "integrates stakeholder engagement, iterative feedback loops, and measurable outcomes. "
    "Ultimately, the goal is to harness the power of data-driven decision making while "
    "fostering a culture that elevates performance. Furthermore, this guide provides a "
    "testament to how organizations can embark on a journey of transformation, setting the "
    "stage for sustainable growth and a resilient future. By leveraging frameworks that "
    "unify strategy with execution, teams can cultivate scalable practices that resonate "
    "across departments and deliver clear ROI. Moreover, this approach is a beacon of "
    "clarity in an ever-evolving landscape, ensuring that leaders remain focused on what "
    "matters most: people, process, and purpose."
)


# Human-like sample (personal, details, variable sentence lengths)
human_text = (
    "I remember the first time I tried to fix the attic window. The wind was already "
    "pushing at the eaves and the ladder felt fragile under my hands. My father had left "
    "me a note about the loose panes, but he hadn't mentioned the pigeons that nested in "
    "the rafters or the way the old glass made a thin, ringing sound when you tapped it. "
    "I spent the afternoon prying old nails out, swearing at splinters, and balancing a "
    "tray of tea on the windowsill. At one point, a neighbor walked by and called hello; "
    "we chatted about the market and about the late rain. Small mundane moments like that "
    "build the tapestry of a day — the kind of detail that doesn't fit into neat lists or "
    "presentation slides. Writing this now, I can feel the grain of the wood and smell the "
    "damp paper of the instruction manual. Those sensory fragments anchor memory in a "
    "way no generic paragraph ever could, and they make the story mine rather than an "
    "example of a process."
)


# Mixed sample: somewhat polished but personal detail present
mixed_text = (
    "When preparing the report, I followed a template that my mentor had used for years, "
    "but I still found myself jotting down small asides about the team's morale and the "
    "odd schedules that had made timelines slip. The document was clean and composed, "
    "with clear headings and an executive summary, yet it also contained an anecdote about "
    "a missed dinner that stuck with me. People read the summary first, but the anecdote "
    "was the part that later sparked a productive conversation in the meeting. That mix of "
    "structure and irregular human narrative is what I aim for when drafting things: "
    "enough polish to be useful, and enough mess to be honest."
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            print_result(path, Path(path).read_text(encoding="utf-8"))
    else:
        print_result("AI-like sample", ai_text)
        print_result("Human-like sample", human_text)
        print_result("Mixed sample", mixed_text)
