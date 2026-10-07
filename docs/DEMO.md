# ReviewMind — Live Demo Script

A ~7-minute guided demo for one presenter and one browser. The audience
needs no setup; everything happens in the Streamlit app at
`http://localhost:8501`.

## Before the audience arrives

1. `pip install -r requirements.txt`
2. `streamlit run app/main.py` — keep this tab open.
3. Open `data/sample_reviews/sample_reviews.txt` in an editor (you will
   paste from it, or just upload it — both paths are part of the demo).
4. Decide live vs. demo mode:
   - **Live** (recommended): `.env` contains `LLM_API_KEY`. Analysis takes
     20–60 seconds — that pause is scripted into Act 2, not dead air.
   - **Offline backup**: rename `.env` away and restart. The app shows a
     demo-mode banner and answers instantly with the local mock client.

## Act 1 — The problem (1 min)

> "A product launch gets hundreds of reviews. Star ratings tell you
> *how much* people like it, never *why*. Reading them all doesn't
> scale, Ctrl+F doesn't understand opinions. ReviewMind turns a pile
> of reviews into a dashboard: sentiment, aspects, complaints, and
> answers — in one click."

Point at the empty app: input on top, **Analyze Reviews** button, nothing
else yet. "No dashboard until there is data — everything below this
button is computed, not hardcoded."

## Act 2 — Analyze (2 min, includes the model wait)

1. Upload `data/sample_reviews/sample_reviews.txt` via **Upload .txt / .csv**
   ("61 real-ish phone reviews — mixed, the way real feedback is").
2. Click **Analyze Reviews**.
3. While the spinner runs, walk the pipeline — this is the architecture
   slide, spoken:
   > "Reviews go through traditional NLP first — cleaning, splitting,
   > statistics — then a prompt is built and sent to an LLM API. The
   > JSON that comes back is *validated* against a schema before
   > anything renders. If the model returns garbage, you get a friendly
   > error, never a broken dashboard."

## Act 3 — The dashboard (2 min)

Tour top to bottom, one sentence each:

- **Overall Sentiment + aspect percentages** — "Mixed, and the four
  buckets always sum to 100%."
- **Charts** — bar chart of aspect sentiment, donut of the mix.
- **Aspect table + positive/negative points** — "Camera praised, battery
  criticized — the classic split."
- **Common complaints** (yellow, not red — "a finding, not a crash")
  **+ AI summary**.
- **NLP statistics expander** — "This part never touched the model:
  sentence counts, tokens, top keywords like battery(9), camera(8).
  Deterministic code does structure; the model does judgment."

## Act 4 — Ask the reviews (1.5 min)

1. The question is prefilled: *"What is the biggest problem reported by
   customers?"* Click **Ask**. ("Battery — and it quotes the reviews,
   it doesn't invent.")
2. Then type something unanswerable, e.g. *"What is the price in EUR of
   the third unit shipped to Berlin?"* Click **Ask**.
   > "It refuses: *cannot be determined from the provided reviews*.
   > Grounded Q&A or nothing — that refusal is a tested behavior,
   > `tests/e2e_live.py` asserts it against the live model."

## Act 5 — Close (30 s)

> "Reviews → NLP → prompt → LLM API → validated JSON → dashboard.
> 21 unit tests, 18 live end-to-end checks, and a demo mode that runs
> with no key at all. Questions?"

## If something goes wrong

| Symptom | Say / do |
|---|---|
| Spinner runs > 2 min | "Free-tier models queue — talk through Act 3 from memory, it always lands." |
| HTTP 429 | Wait 30 s, retry. The client backs off automatically. |
| HTTP 401/403 | Key rejected or model gated — switch `LLM_MODEL` in `.env`, restart. |
| No network at all | Restart without `.env` → demo-mode backup, instant answers. |
| Empty-input / oversized errors | Intentional guards — click through them, they are part of the story. |
