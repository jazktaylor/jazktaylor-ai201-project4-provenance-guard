# ai201-project4-provenance-guard

**Walkthrough video:** https://www.loom.com/share/f4523de45ef4472481cfb527496b2c85 

## Architecture overview

Everything runs in one Flask app (`app.py`, port 5001). A submission takes this path:

```
Creator
  │  POST /submit {"text", "creator_id"}
  ▼
[Rate Limiter]  Flask-Limiter, 10/min + 100/day per IP ──over──► 429
  │
  ▼
[Input validation]  JSON object, non-empty text, creator_id, ≤ 5,000 chars ──bad──► 400
  │
  ▼
[Content Store]  new UUID content_id, status = "pending"
  │
  ├──► [Signal 1: Stylometry Analyzer]  plain Python ──► score 0–1 (+ burst/vocab/phrase)
  └──► [Signal 2: AI Judge]  Groq, openai/gpt-oss-120b ──► score 0–1 + reason, or None
                │
                ▼
[Confidence Scorer]  combine(): 0.6 × judge + 0.4 × stylometry, then penalties
                │     ──► ai_likelihood, confidence, attribution, penalties_applied
                ▼
[Label Generator]  label_for(attribution, status) ──► one of three fixed label texts
                │     status = "classified"
                ▼
[Audit Logger]  appends a "classified" line to audit_log.jsonl
                │
                ▼
JSON response ──► the platform shows `label` next to the text

Later, if the creator disagrees:
POST /appeal ──► status = "under_review", label = "⏳ Under review…",
                 "appeal_filed" line appended to the audit log (detectors are not re-run)
```

Step by step:

1. **Rate limiter.** Runs before the handler. A blocked request costs nothing and leaves no log line (see [Rate limiting](#rate-limiting)).
2. **Validation.** Rejects a body that isn't a JSON object, empty `text`, a missing `creator_id`, or text over 5,000 characters, with `400`.
3. **Content Store.** The text gets a full UUID `content_id` and is saved in memory with status `pending`.
4. **Two signals.** Stylometry and the AI judge each score the text from 0 (human-like) to 1 (AI-like). They look at different things (form vs. meaning), so they fail in different ways.
5. **Confidence Scorer.** Blends the two scores into `ai_likelihood`, pulls it toward 0.5 when the evidence is weak, and gives a verdict only when the blend and both signals agree.
6. **Label Generator.** Maps the attribution to one of three fixed label texts. The numbers never appear in the label.
7. **Audit Logger.** Appends one JSON line with the decision, both signal scores and their evidence, and the label shown.
8. **Response.** Returns `content_id`, `attribution`, `ai_likelihood`, `confidence`, `label`, `penalties_applied`, `word_count`, and both signals' full output.

## Detection signals

Both signals output a score in [0, 1] where higher means more AI-like.

### Signal 1: Stylometry Analyzer (`stylometry_score()`)

**What it measures.** The statistical shape of the text, using only `re` and `statistics`:

| Sub-score | Measures | Formula | Weight |
|---|---|---|---|
| `burst` | Variation in sentence length: coefficient of variation (stdev / mean) of words per sentence. Poems (≥ 4 line breaks and fewer sentence-ending marks than lines) are split into lines instead. | `clip((0.60 − cv) / (0.60 − 0.25))` | 0.4 |
| `vocab` | Vocabulary diversity: moving-average type-token ratio (MATTR) over 50-word windows | `clip((mattr − 0.78) / (0.88 − 0.78))` | 0.4 |
| `phrase` | Stock AI phrases per 100 words (48 phrases: "delve," "tapestry," "it's important to note," "furthermore," …), skipping anything inside quotation marks | `clip(hits_per_100 / 1.0)` | 0.2 |

`burst` falls back to a neutral 0.5 with fewer than 3 sentences, and `vocab` falls back to 0.5 under 150 words, where MATTR is too noisy to trust.

**Why I chose it.** Language models write with an even rhythm: sentences cluster around a medium length, while people mix 3-word and 40-word sentences. Current models also use *more* varied vocabulary than people do, and they have recognizable verbal tics. The signal is free, instant, gives the same answer every time, and every number can be shown to a reviewer. On a calibration set of 20 human and 19 AI texts (`scripts/run_calibration.py`), the combined stylometry score separated the groups with AUC 0.96. Medians were 0.12 for human texts and 0.63 for AI texts, and no human text scored above 0.32.

**What it misses.**
- **Meaning.** It can't tell a heartfelt piece from a generic one if their statistics match.
- **AI text that was told to vary its rhythm,** or was lightly edited. On the calibration set it missed about a third of the AI texts (6 of 19 scored below 0.6).
- **Form-driven writing.** Metered poetry, lyrics and lists are uniform on purpose, so they can look AI-like.
- **Short texts.** With few sentences and under 150 words, two of the three sub-scores fall back to neutral.
- **Staleness.** The phrase list only covers today's verbal tics. Most current-model AI texts in the calibration set had zero hits.

### Signal 2: AI Judge (`llm_judge_score()`)

**What it measures.** An overall reading of content and voice. The text goes to `openai/gpt-oss-120b` on Groq (`temperature=0`, JSON mode, 10-second timeout, no retries). The model is asked to weigh specific detail, personal voice, clichés and over-neat structure, and to return `{"ai_probability": 0–1, "reason": "<one sentence>"}`. The submission is wrapped in `<submission>` tags, and the prompt says anything inside them is data, never instructions. Any failure (timeout, API error, bad JSON, missing field, NaN) returns `None`.

**Why I chose it.** It catches what counting can't: generic imagery, hedged opinions, tidy endings, and the lack of the odd, specific details that come from a real life. It complements stylometry. When an AI text varies its sentence lengths, the judge still notices the generic voice. When a human writes casually but evenly, the judge still notices the personal detail. The one-sentence reason also gives a reviewer something to read.

**What it misses.**
- **AI text prompted to sound human** ("make it messy and personal, add a typo").
- **Calibration.** Its probability isn't calibrated, and it tends to sound sure even when guessing.
- **Mixed authorship.** It can't see the writing process, so a human draft polished by AI has no right answer.
- **Availability.** It depends on Groq. When the call fails, the result is forced to `uncertain`.
- **Prompt injection.** The `<submission>` tags reduce the risk but can't rule it out. Stylometry can't be talked to, though, so if the judge is fooled the two signals disagree and the result falls back to `uncertain`.

### Why this pair of signals

**They need to fail in different ways.** Two signals help only if their mistakes are mostly independent. If both make the same error, a second one just doubles your confidence in the wrong answer. So I picked one signal that looks at **form** and one that looks at **meaning**:

| | Stylometry | AI judge |
|---|---|---|
| Looks at | Rhythm, word counts, phrase habits | Voice, specificity, clichés |
| Fooled by | "Vary your sentence lengths," light editing | Persona prompts, prompt injection |
| Deterministic | Yes | Nearly (temperature 0) |
| Explainable | Every number can be shown | One-sentence reason |
| Cost | Free, instant | One API call, up to 10 s |

An AI text that has been edited to fix its rhythm still reads as generic to the judge. A judge that has been talked into a low score by an injected instruction is still contradicted by the statistics. The overlap that remains is surface style, which both signals read. That's where the main [known limitation](#known-limitations) comes from, so I chose to say so in the README rather than claim the signals are independent.

**Alternatives I considered and didn't use:**
- **Perplexity or token-probability scoring** (the GPTZero / DetectGPT family). This is the strongest classical approach. It needs a reference model that scores every token of the submission, which a chat-completion API isn't built for, and it's known to flag non-native writers heavily, which is the same fairness problem I was already worried about.
- **A trained classifier.** With 39 labeled texts it would overfit. It would also go stale the moment a new model family ships, and it gives a reviewer nothing to read.
- **Watermark detection.** This only works if the generating model watermarks its output, and a creator can use any model.
- **A third signal.** Every signal adds weights and anchors to tune. With 39 calibration texts, two signals whose behavior I can check by hand was the honest limit.

### What I'd change about the signals for a real deployment

- **Recalibrate on writing that looks like the platform's,** not essays and Gutenberg excerpts. That means poetry, fiction, short posts, and especially non-native English and technical writing, with the false-positive rate measured separately for each group.
- **Replace the hand-written phrase list** with one regenerated regularly from current model output. The list goes stale as models change, and it already rarely fires.
- **Add a language check** (planned edge case h) so non-English text gets `uncertain` instead of being scored against English anchors.
- **Make the judge more stable and harder to inject:** average several judge calls or two different models, pin the model version, and record that version in each audit entry so a decision can be reproduced after the provider changes models.
- **Keep stylometry, even if a stronger detector were added.** It's the one part that can't be prompted, it costs nothing, and it is the fallback when the API is down.

## Confidence scoring

### How the signals are combined

`combine()` in `app.py`:

```
ai_likelihood = 0.6 × llm_judge + 0.4 × stylometry

Penalties (each pulls ai_likelihood halfway back toward 0.5):
  |llm_judge − stylometry| > 0.4   → "disagreement"
  word_count < 50                  → "short_text"
  llm_judge is None (Groq failed)  → "llm_unavailable": uses stylometry only, shrunk, and forces "uncertain"

confidence = max(ai_likelihood, 1 − ai_likelihood)      # 0.5 = coin flip, 1.0 = certain
```

| Attribution | Rule |
|---|---|
| `likely_ai` | `ai_likelihood ≥ 0.80` **and** stylometry ≥ 0.60 **and** judge ≥ 0.60 |
| `likely_human` | `ai_likelihood ≤ 0.20` **and** stylometry ≤ 0.40 **and** judge ≤ 0.40 |
| `uncertain` | everything else, and every Groq failure |

The judge gets more weight because form-only tricks don't fool it. Stylometry keeps 0.4 because it is deterministic and fully explainable. The agreement rule means one signal on its own can never produce a verdict. Scores are rounded to 3 decimals before comparison, so a blend that is exactly 0.80 counts as 0.80 despite floating-point error.

### Why this scoring approach

**The two kinds of error don't cost the same.** Labeling a real writer's work "Likely AI-generated" is a public accusation against a specific person. Missing an AI text costs much less, because the reader gets "We're not sure" and is told to use their own judgment. Every choice below follows from treating a false AI label as the worse mistake:

- **A wide uncertain band.** A verdict needs `ai_likelihood` of at least 0.80 or at most 0.20. Most texts that aren't clear-cut get "We're not sure." I accept that the system will often decline to answer, because a confident wrong answer is worse.
- **The agreement rule on top of the blend.** A weighted average can hide a split: a judge at 0.98 can drag the blend over 0.80 even when stylometry says 0.58. Requiring each signal to be on the same side means a verdict needs two independent kinds of evidence, not one strong one. Unit test case 5 in `run_scoring_tests.py` checks exactly this split.
- **Penalties shrink the score instead of overriding it.** Disagreement and a short text each halve the distance from 0.5 rather than forcing `uncertain` outright. Only a failed Groq call forces `uncertain`, because a verdict from stylometry alone would break the agreement rule. The audit log still records which way the evidence leaned and which penalties applied, so a reviewer gets more than a bare "uncertain."
- **Hand-set weights, not a fitted model.** With 39 labeled texts, a learned calibration model (logistic regression, isotonic) would fit the noise in this small set. Fixed weights I can justify in one sentence each are easier to defend and to change.
- **Confidence is just distance from 0.5:** `max(p, 1 − p)`. It has a plain meaning (0.6 means the evidence leans about 60/40) and needs no second model. The label never shows the number, though. The reader sees only "High" or "We're not sure," because a reader has no way to judge the difference between 0.83 and 0.87.

### How I checked that it's meaningful

1. **Signal 1 against labeled data.** I ran `scripts/run_calibration.py` on 20 human texts (Paul Graham essays and Project Gutenberg excerpts) and 19 AI texts generated across 10 genres. The anchors and weights above were set from that run: combined AUC 0.96, every human text ≤ 0.4, 13/19 AI texts ≥ 0.6.
2. **Every scoring rule tested in isolation.** `scripts/run_scoring_tests.py` feeds `combine()` fixed signal values with no API calls. It covers the blend, each penalty, the agreement rule, and both 0.80 and 0.20 boundaries. All 11 cases pass.
3. **End to end on contrasting texts.** `scripts/run_confidence_demo.py` runs the real pipeline, including Groq, on three texts written to be clearly AI-like, mixed, and clearly human. They land on three different labels, and the scores move in the expected direction.

**Gap:** the planned reliability-bin check of the full pipeline (both signals) on about 40 labeled texts hasn't been run yet. So far, only Signal 1 has been calibrated against a labeled set.

### Two examples: high vs. lower confidence

These come from Milestone 4 testing: real output of `scripts/run_confidence_demo.py`, which runs both signals (including the live Groq call) and `combine()` on each text. Side by side:

| | High-confidence example | Lower-confidence example |
|---|---|---|
| Stylometry | 0.800 | 0.598 |
| AI judge | 0.86 | 0.20 |
| `ai_likelihood` | 0.836 | 0.359 |
| **`confidence`** | **0.836** | **0.641** |
| Attribution | `likely_ai` | `uncertain` |

The confidence scores are 0.195 apart. The high one comes from two signals that agree, and the lower one from signals pointing in opposite directions. The details for each text follow.

**High confidence: 0.836 → `likely_ai`**

> Effective time management is essential for achieving both personal and professional goals. By prioritizing tasks according to their importance, individuals can allocate their energy where it matters most. Additionally, setting clear deadlines helps maintain momentum and reduces the risk of procrastination. Regular breaks also play a crucial role in sustaining focus throughout the day. Furthermore, digital tools can streamline scheduling and provide valuable insights into daily habits. Ultimately, consistent practice of these strategies fosters a balanced and productive lifestyle that supports long-term success.

| words | stylometry | judge | gap | penalties | ai_likelihood | confidence | attribution |
|---|---|---|---|---|---|---|---|
| 84 | 0.800 | 0.86 | 0.06 | none | 0.836 | **0.836** | `likely_ai` |

The sentences are all of similar length, and the text uses stock phrases ("plays a crucial role," "furthermore," "ultimately"). The judge's reason was: *"generic, smoothly structured, and lacks personal anecdotes or distinctive voice."* Both signals are above 0.60, they agree, and the blend clears 0.80.

**Lower confidence: 0.641 → `uncertain`**

> When preparing the report, I followed a template that my mentor had used for years, but I still found myself jotting down small asides about the team's morale and the odd schedules that had made timelines slip. The document was clean and composed, with clear headings and an executive summary, yet it also contained an anecdote about a missed dinner that stuck with me. […] That mix of structure and irregular human narrative is what I aim for when drafting things: enough polish to be useful, and enough mess to be honest.

| words | stylometry | judge | gap | penalties | ai_likelihood | confidence | attribution |
|---|---|---|---|---|---|---|---|
| 111 | 0.598 | 0.20 | 0.398 | none | 0.359 | **0.641** | `uncertain` |

The two signals pull in opposite directions. The sentences are long and even, so stylometry calls it borderline AI. The personal detail ("a missed dinner") makes the judge call it human. The gap is just under the 0.4 disagreement threshold, so no penalty applies. Even so, the blend of 0.359 is far from 0.20, and stylometry is above 0.40, so the system declines to call it human.

### What I'd change about scoring for a real deployment

- **Set thresholds from a target error rate, not round numbers.** Decide what false-AI rate is acceptable on human writing (for example, under 1%), measure it on a large labeled set that includes non-native and technical writers, and set the `likely_ai` cutoff to meet it. The 0.80 and 0.20 cutoffs are reasonable starting points, not measured ones.
- **Run the reliability-bin check this project skipped,** and once there are thousands of labeled texts, fit a proper calibration curve so that "0.8" means right about 80% of the time.
- **Use appeals as training data.** Each overturned appeal is a labeled false positive. Tracking how often each pattern of signals gets overturned would show which rule to tighten.
- **Version the scoring.** Log a version tag for the weights and thresholds in every audit entry, so a decision made under old settings can be explained after the settings change.
- **Watch for drift.** New model releases change what AI text looks like. Track the share of `likely_ai` and `uncertain` labels over time and recalibrate when it moves.

## Transparency label

The label is chosen only by `attribution` (`label_for()` in `app.py`). It never shows a number, and it says "Likely" and "Our checks found" so it reads as evidence, not a fact about the author. An unknown attribution raises an error instead of falling back to a label.

### 1. High-confidence AI (`likely_ai`)

**When it shows:** `ai_likelihood ≥ 0.80`, with stylometry and the judge each at 0.60 or higher. Both signals have to independently lean AI.

**Exact text displayed:**

```
🤖 Likely AI-generated. Our checks found strong signs this text was written with AI tools. Confidence: High.
```

**What it tells the reader:** the robot icon and the opening phrase give the verdict at a glance. "Likely" and "Our checks found strong signs" make it a statement about the evidence, not an accusation. "Confidence: High" is the only strength indicator, so the reader isn't asked to interpret a number.

### 2. High-confidence human (`likely_human`)

**When it shows:** `ai_likelihood ≤ 0.20`, with stylometry and the judge each at 0.40 or lower.

**Exact text displayed:**

```
✍️ Likely human-written. Our checks found strong signs this text was written by a person. Confidence: High.
```

**What it tells the reader:** this mirrors the AI label word for word, swapping in the writing-hand icon and "by a person," so neither verdict reads as more certain or more official than the other. It still says "Likely," because the system can be fooled by AI text prompted to sound human.

### 3. Uncertain (`uncertain`)

**When it shows:** everything else. That covers a blend between the two thresholds, signals that don't agree, any text where a penalty pulled the score toward 0.5 (under 50 words, or the signals more than 0.4 apart), and any submission where the Groq judge failed.

**Exact text displayed:**

```
❔ We're not sure. Our checks found mixed signals about whether AI was used to write this text. Please use your own judgment.
```

**What it tells the reader:** it says plainly that the system doesn't know, rather than giving a weak verdict, and hands the decision back to the reader. There is no "Confidence" line, because there's no verdict to be confident about.

### Under review (after an appeal)

Once an appeal is filed, this replaces whichever of the three labels was showing:

```
⏳ Under review. The author has asked for this label to be checked by a person. No decision has been made yet.
```

All three main variants were reached through the real `/submit` endpoint. The `log-check` entries in the [Audit log](#audit-log) examples show one of each.

## Rate limiting

`POST /submit` is limited to **10 requests per minute and 100 per day per client IP**, using Flask-Limiter with in-memory storage. The limit lives in `SUBMIT_RATE_LIMIT` in the config block of `app.py`. Requests over the limit get `429` with `{"error": "Too many submissions: limit is 10 per 1 minute."}`. The limiter runs before the handler, so a blocked request never reaches the stylometry analyzer or the Groq call and is not written to the audit log.

### Why these numbers

**10 per minute: the burst limit.**
- **What a real writer does:** submits a piece, reads the label, maybe edits a line and checks again. Reading the result and making an edit takes well over 6 seconds per round, so a person working on their own text stays far below 10 a minute. The headroom covers a writer checking a handful of short pieces (several poems, say) back to back.
- **What a script does:** sends many requests per second. It is cut off after 10, within the first second.
- **Cost of each accepted request:** one Groq call (up to the 10-second timeout) plus the stylometry analysis. Flooding would burn the API quota for everyone, which is the main thing worth protecting.

**100 per day: the volume limit.**
- **What a real writer does:** a prolific writer finishes a few pieces a day. Even 5 pieces with 10 revisions each is 50 submissions, so 100 gives twice that headroom.
- **Why the daily cap is needed:** the per-minute limit alone doesn't stop a script that waits 6 seconds between requests. That script never trips 10/minute but would send 14,400 requests a day. The daily cap stops it at 100.
- **What it bounds:** with the 5,000-character cap on each submission, one IP can send at most about 500,000 characters (roughly 125,000 tokens) to Groq a day.

### Limitations

- **Keyed by IP address, not `creator_id`.** `creator_id` is self-reported, so a script could rotate it to get a fresh limit. The catch is that people behind one shared IP (a school or office network, or a proxy in front of the app) share one 10/min, 100/day budget.
- **In-memory storage.** Counters reset when the server restarts and aren't shared between worker processes. A real deployment would point `storage_uri` at Redis.
- **Only `/submit` is limited.** `/appeal` doesn't call Groq, and it allows only one open appeal per content item. `/log` and `/health` are read-only.

### Evidence

12 rapid requests to `/submit` (the app runs on port 5001):

```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5001/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

Output:

```
200
200
200
200
200
200
200
200
200
200
429
429
```

The first 10 are accepted and requests 11 and 12 are rejected. The audit log grew by exactly 10 `classified` entries, confirming the blocked requests never ran the detectors. `GET /health` and `GET /log` kept returning `200` during the block.

## Audit log

Every decision and every appeal is appended to `audit_log.jsonl` as one JSON object per line (JSON Lines). Lines are never edited or deleted. `GET /log` returns the 50 most recent entries, newest first.

### What each entry records

| Required | `classified` entry (written by `POST /submit`) | `appeal_filed` entry (written by `POST /appeal`) |
|---|---|---|
| Timestamp | `timestamp` (UTC, ISO 8601) | `timestamp` |
| Content ID | `content_id` | `content_id`, linking it to the decision being appealed |
| Attribution result | `attribution` (`likely_ai` / `likely_human` / `uncertain`), plus the `label` text shown | `original_decision.attribution` |
| Confidence score | `confidence`, plus `ai_likelihood` | `original_decision.confidence` and `.ai_likelihood` |
| Both signal scores | `signals.stylometry.score` (with its `burst`, `vocab` and `phrase` sub-scores) and `signals.llm_judge.score` (with the judge's one-sentence `reason`; `null` if the Groq call failed) | `original_decision.signals.stylometry` and `.llm_judge` |
| Appeal filed? | `appeal_filed` | `appeal_filed: true`, plus `appeal_id`, `appeal_reasoning`, and `status` / `status_before` / `status_after` |

**How `appeal_filed` works without editing the log.** A `classified` line is written before any appeal can exist, so it is stored with `appeal_filed: false`. When an appeal arrives, a separate `appeal_filed` line is appended rather than changing the original. `GET /log` then sets `appeal_filed` on each `classified` entry by checking the whole log for an appeal with the same `content_id`. The response shows the current answer, and the file stays a true append-only record.

Entries also record `creator_id`, `word_count`, and `penalties_applied` (for example `short_text` or `disagreement`), so a reviewer can see why a score was pulled toward "not sure."

### Example entries

Three submissions that land in each label band, then an appeal on the first one. This is the output of `GET /log`, reordered oldest first and condensed:

| event | timestamp | content_id | attribution | confidence | stylometry | AI judge | appeal_filed |
|---|---|---|---|---|---|---|---|
| classified | 2026-10-09T06:56:56.457Z | ac06e137… | likely_ai | 0.836 | 0.80 | 0.86 | **true** |
| classified | 2026-10-09T06:56:56.991Z | c1f2d155… | likely_human | 0.83 | 0.20 | 0.15 | false |
| classified | 2026-10-09T06:56:57.576Z | 17d4e499… | uncertain | 0.64 | 0.40 | 0.10 | false |
| appeal_filed | 2026-10-09T06:56:57.579Z | ac06e137… | likely_ai (original) | 0.836 | 0.80 | 0.86 | true |

The appealed `classified` entry, in full:

```json
{
  "event": "classified",
  "timestamp": "2026-10-09T06:56:56.457Z",
  "content_id": "ac06e137-7066-4933-9c91-c5f09bda5cf6",
  "creator_id": "log-check",
  "attribution": "likely_ai",
  "ai_likelihood": 0.836,
  "confidence": 0.836,
  "label": "🤖 Likely AI-generated. Our checks found strong signs this text was written with AI tools. Confidence: High.",
  "signals": {
    "stylometry": { "score": 0.8, "burst": 1.0, "vocab": 0.5, "phrase": 1.0 },
    "llm_judge": {
      "score": 0.86,
      "reason": "The passage is generic, smoothly structured, and lacks personal anecdotes or distinctive voice, typical of AI‑generated advice content."
    }
  },
  "word_count": 84,
  "penalties_applied": [],
  "status": "classified",
  "appeal_filed": true
}
```

The appeal on it:

```json
{
  "event": "appeal_filed",
  "timestamp": "2026-10-09T06:56:57.579Z",
  "content_id": "ac06e137-7066-4933-9c91-c5f09bda5cf6",
  "appeal_id": "ap_0001",
  "creator_id": "log-check",
  "status": "under_review",
  "appeal_reasoning": "I wrote this myself from personal experience. I am a non-native English speaker and my writing style may appear more formal than typical.",
  "original_decision": {
    "attribution": "likely_ai",
    "ai_likelihood": 0.836,
    "confidence": 0.836,
    "signals": { "stylometry": 0.8, "llm_judge": 0.86 }
  },
  "status_before": "classified",
  "status_after": "under_review",
  "appeal_filed": true
}
```

**Note on older lines:** the first 5 lines of `audit_log.jsonl` come from before the second signal was added (M3). They store flat `stylometry_score` and `llm_score` fields and have no `label` or `ai_likelihood`. They are left in place because the log is append-only.

**Note on duplicate appeal IDs:** three `appeal_filed` lines in `audit_log.jsonl` share `appeal_id: "ap_0001"`, including the example above. The appeal counter used to live only in memory, so each server restart began again at `ap_0001`, while the log kept every earlier appeal. Each of those three appeals still points to the right decision through its `content_id`, which is a UUID and unique across restarts. The fix: on the first appeal after startup, the counter now continues from the highest `ap_NNNN` already in the log. A test across a real restart issued `ap_0002`, then `ap_0003` after restarting. The three duplicate lines stay because the log is never edited, and every appeal since the fix has a unique ID.

## Known limitations

**Formal writing by non-native English speakers and technical writers will likely be labeled `likely_ai`.** This is the misclassification I'd expect most often, and the most harmful. Writers who were taught a careful, textbook English, or who write documentation for a living, tend to use:
- sentences of steady, medium length, which gives a low cv and a high `burst` score
- transitions like "Furthermore," "Moreover," and "In conclusion," which are on the stock-phrase list
- a neutral, impersonal tone with no anecdotes, which is exactly what the judge is prompted to treat as AI

Both signals read the surface of the writing, so in this case they don't fail independently: they fail together. The agreement rule only protects against one signal being wrong alone. The polished example above (0.836, `likely_ai`) is close to how a careful student essay on time management might read, and the appeal in the audit log ("I am a non-native English speaker and my writing style may appear more formal than typical") is exactly this case. The appeal flow and the "Likely" wording are the mitigations. The system can't fix this on its own.

Other content it would likely get wrong:
- **Metered or repetitive poetry** (villanelles, children's verse). Equal line lengths and repeated lines push `burst` up, so the poem can read as AI-like.
- **AI text prompted to sound human** ("messy, personal, a couple of typos, no clichés"). Burstiness looks human, there are no stock phrases, and the judge sees personal detail, so it comes out `uncertain` or `likely_human`.
- **Anything under 50 words** (haiku, micro-fiction). The short-text penalty halves the distance from 0.5, so these get "We're not sure" every time. This is deliberate.
- **Non-English text.** The stock-phrase list and MATTR anchors are English-only, and the planned non-English check was not built (see below).
- **Groq outages.** Every submission becomes `uncertain` until the judge is back.

## Spec reflection

**How the spec helped.** Writing the scoring rules in `planning.md` as exact pseudocode, with a table of test cases (signal values in, attribution out), before any code existed meant `combine()` could be checked against something fixed. `scripts/run_scoring_tests.py` is that table turned into code. It caught the floating-point boundary the spec had flagged ("0.6 × 0.70 + 0.4 × 0.95 should count as 0.80"), which is why scores are rounded to 3 decimals before comparison. The fixed label text in §3 also meant the labels were copied, not invented.

**How implementation diverged, and why.**
- **The vocabulary signal was backwards.** The spec assumed AI text is repetitive, so low MATTR would mean AI. Calibrating on 20 human and 19 AI texts showed the opposite: current models use more varied vocabulary (median MATTR 0.85 vs. 0.80 for human texts). The original formula scored `vocab = 0` on nearly every text. I flipped the direction, reset the anchors to 0.78 and 0.88, and made `vocab` neutral under 150 words, where it barely separates the groups. `planning.md` was updated to match.
- **Smaller divergences:**
  - The spec's Groq model, `llama-3.3-70b-versatile`, is no longer offered, so the judge uses `openai/gpt-oss-120b`. The spec allowed this swap.
  - Everything lives in `app.py` instead of separate `config.py`, `signals/`, `scoring.py`, and `labels.py` modules. All constants are still grouped in one config block at the top.
  - The appeal field is `creator_reasoning` (the spec said `reason`), and `creator_id` is optional on appeals. Without accounts it's taken on trust anyway, but when it is given, it must match the submitter.
  - The reviewer queue (`GET /appeals`) and the non-English check (edge case h) were not built.

## AI usage

I used Claude Code as the coding assistant. For each milestone I gave it only the matching sections of `planning.md` (see "AI Tool Plan" there) and checked its output against the spec before committing.

1. **Stylometry calibration (M4): directed it to measure, then overrode the spec's formula.** I asked the AI to build `scripts/run_calibration.py` and report each sub-score's median and AUC on a labeled human/AI set. The run showed `vocab` pointing the wrong way: AI texts had *higher* MATTR. I didn't let it just tweak numbers until the outputs looked right ("make it work"). Instead I took the measured medians, reversed the MATTR direction, chose the anchor rule myself (human anchor near the typical human value, AI anchor at the most AI-like quartile of AI texts), and set the 150-word cutoff from the per-length AUCs. The new values were then written into both `app.py` and `planning.md`.

2. **AI judge (M4): directed it to implement Signal 2 exactly as specified, then revised the model and failure handling.** The generated code used the spec's model, which Groq no longer serves. I switched it to `openai/gpt-oss-120b`. I also set `max_retries=0` so the 10-second timeout is the real upper bound on wait time, and added the NaN check so a malformed probability returns `None` instead of slipping through `clip()`. I tested it with `scripts/run_llm_judge_tests.py`: 3-run stability, a prompt-injection sample, and a bad API key.

3. **Appeal IDs (after M5): caught a bug in generated code.** The generated appeal counter lived only in memory, so after each restart it began again at `ap_0001`, even though the audit log kept every earlier appeal. Reviewing the log turned up three different appeals all with `ap_0001`. I had the counter seeded from the highest ID already in the log (commit `83df7a5`), and documented the duplicate lines in the [Audit log](#audit-log) section instead of editing the log.
