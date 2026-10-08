import re
import statistics
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config (starting values; moves to config.py per the spec)
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 5000

# Signal 1: burstiness anchors (coefficient of variation of words per unit)
BURST_CV_HUMAN = 0.70
BURST_CV_AI = 0.25
BURST_MIN_UNITS = 3
BURST_NEUTRAL = 0.5

# Signal 1: vocabulary diversity anchors (MATTR)
MATTR_WINDOW = 50
MATTR_HUMAN = 0.78
MATTR_AI = 0.62

# Signal 1: stock phrases (hits per 100 words that gives the maximum score)
PHRASE_MAX_RATE = 1.0

# Signal 1: combine weights
W_BURST = 0.4
W_VOCAB = 0.3
W_PHRASE = 0.3

# Poetry mode: at least this many line breaks, and fewer sentence-ending
# marks than non-empty lines
POETRY_MIN_LINE_BREAKS = 4

# Starting list (moves to signals/stock_phrases.txt per the spec)
STOCK_PHRASES = [
    "delve", "delves", "delving", "delved",
    "tapestry", "testament to", "a symphony of",
    "it's important to note", "it is important to note",
    "it's worth noting", "it is worth noting",
    "in today's fast-paced world", "in today's digital age",
    "in the ever-evolving", "ever-evolving landscape",
    "navigate the complexities", "navigating the complexities",
    "the realm of", "in the realm of",
    "plays a crucial role", "plays a pivotal role", "a pivotal role",
    "a crucial role", "intricate interplay", "rich tapestry",
    "unlock the potential", "unleash the power", "harness the power",
    "embark on a journey", "a journey of", "multifaceted",
    "seamlessly", "furthermore", "moreover", "in conclusion",
    "in summary", "ultimately", "a myriad of", "myriad",
    "foster a sense of", "elevate your", "game-changer",
    "at the end of the day", "stands as a", "serves as a reminder",
    "a beacon of", "resonates with", "shed light on",
]

UNCERTAIN_LABEL = (
    "❔ We're not sure. Our checks found mixed signals about whether AI was "
    "used to write this text. Please use your own judgment."
)

# ---------------------------------------------------------------------------
# Signal 1: Stylometry Analyzer
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"\w+(?:'\w+)*")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+")
SENTENCE_END_RE = re.compile(r"[.!?]+")
# Quoted spans are skipped when counting stock phrases. A single-quote span
# must start at a word boundary so apostrophes ("it's") don't open one.
QUOTED_RE = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<!\w)'[^'\n]*'(?!\w)")
PHRASE_RES = [
    re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in STOCK_PHRASES
]


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _words(text):
    return WORD_RE.findall(text.lower().replace("’", "'"))


def split_units(text):
    """Split text into sentence units, or line units for poetry."""
    text = text.strip()
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    line_breaks = text.count("\n")
    end_marks = len(SENTENCE_END_RE.findall(text))
    if line_breaks >= POETRY_MIN_LINE_BREAKS and end_marks < len(lines):
        return [ln.strip() for ln in lines]
    return [u.strip() for u in SENTENCE_SPLIT_RE.split(text) if u.strip()]


def mattr(words, window=MATTR_WINDOW):
    """Moving-average type-token ratio. Falls back to plain TTR when the text
    is shorter than one window, and returns None for no words."""
    if not words:
        return None
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [
        len(set(words[i:i + window])) / window
        for i in range(len(words) - window + 1)
    ]
    return statistics.mean(ratios)


def _burst_score(units):
    counts = [n for n in (len(_words(u)) for u in units) if n > 0]
    if len(counts) < BURST_MIN_UNITS:
        return BURST_NEUTRAL
    cv = statistics.stdev(counts) / statistics.mean(counts)
    return _clip((BURST_CV_HUMAN - cv) / (BURST_CV_HUMAN - BURST_CV_AI))


def _vocab_score(words):
    m = mattr(words)
    if m is None:
        return BURST_NEUTRAL
    return _clip((MATTR_HUMAN - m) / (MATTR_HUMAN - MATTR_AI))


def _phrase_score(text, word_count):
    if word_count == 0:
        return 0.0
    unquoted = QUOTED_RE.sub(" ", text).replace("’", "'")
    hits = sum(len(rx.findall(unquoted)) for rx in PHRASE_RES)
    hits_per_100 = hits / word_count * 100
    return _clip(hits_per_100 / PHRASE_MAX_RATE)


def stylometry_score(text):
    """Signal 1. Returns a dict with the overall score in [0, 1]
    (higher = more AI-like) plus the sub-scores for the audit log:
    {"score", "burst", "vocab", "phrase", "units", "word_count"}."""
    text = text or ""
    units = split_units(text)
    words = _words(text)
    word_count = len(words)

    burst = _burst_score(units)
    vocab = _vocab_score(words)
    phrase = _phrase_score(text, word_count)
    score = W_BURST * burst + W_VOCAB * vocab + W_PHRASE * phrase

    return {
        "score": round(score, 3),
        "burst": round(burst, 3),
        "vocab": round(vocab, 3),
        "phrase": round(phrase, 3),
        "units": len(units),
        "word_count": word_count,
    }


# ---------------------------------------------------------------------------
# Content Store (in-memory for now)
# ---------------------------------------------------------------------------

CONTENT_STORE = {}


def _new_content_id():
    while True:
        cid = "c_" + uuid.uuid4().hex[:4]
        if cid not in CONTENT_STORE:
            return cid


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return "Provenance Guard is running."


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/submit")
def submit():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = body.get("text")
    creator_id = body.get("creator_id")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' is required and must be non-empty."}), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "'creator_id' is required."}), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify(
            {"error": f"'text' must be at most {MAX_TEXT_CHARS} characters."}
        ), 400

    content_id = _new_content_id()
    record = {
        "content_id": content_id,
        "creator_id": creator_id.strip(),
        "text": text,
        "submitted_at": _now(),
        "status": "pending",
    }
    CONTENT_STORE[content_id] = record

    stylometry = stylometry_score(text)
    record["signals"] = {"stylometry": stylometry, "llm_judge": None}

    # Placeholder until M4/M5: LLM judge, Confidence Scorer and Label Generator
    # aren't wired in yet, so every result is "uncertain".
    record.update({
        "attribution": "uncertain",
        "ai_likelihood": 0.5,
        "confidence": 0.5,
        "label": UNCERTAIN_LABEL,
    })

    return jsonify({
        "content_id": content_id,
        "status": record["status"],
        "attribution": record["attribution"],
        "ai_likelihood": record["ai_likelihood"],
        "confidence": record["confidence"],
        "label": record["label"],
        "word_count": stylometry["word_count"],
        "signals": record["signals"],
    }), 200


if __name__ == "__main__":
    app.run(port=5001, debug=True)
