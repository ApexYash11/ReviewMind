# ReviewMind — Testing Guide

How to verify the app works, from fastest to slowest. Start at the top;
each layer covers what the previous one cannot.

## 1. Unit tests (20 s, no key, no network)

```bash
pytest -v
```

21 tests across three files:

| File | What it proves |
|---|---|
| `tests/test_nlp.py` | Cleaning, sentence splitting, tokenization, normalization, review splitting, statistics |
| `tests/test_analyzer.py` | Valid payloads accepted; bad sentiment / bad shapes rejected; empty input fails fast |
| `tests/test_mock_llm.py` | Demo client returns schema-valid JSON, detects the battery complaint, answers QA, admits ignorance |

All must pass. A failure here means the pipeline contract broke — do not
proceed to live testing until it is green.

## 2. Live end-to-end script (2–4 min, costs API calls)

```bash
python -m tests.e2e_live
```

18 checks against the real model using
`data/sample_reviews/sample_reviews.txt` (61 reviews). It asserts the NLP
stage, the LLM analysis, schema validation, keyword extraction, the QA
path, both input guards, and that an unanswerable question is refused
rather than hallucinated. Exit `0` = all green, `1` = something failed.
Not collected by `pytest` on purpose.

## 3. Manual UI checklist (5–10 min, in the browser)

Run `streamlit run app/main.py` and work through this table in order.
Every row has an exact expected result — anything else is a bug.

| # | Action | Expected |
|---|---|---|
| 1 | Open the app | Title REVIEWMIND, prefilled example reviews, uploader, Analyze + Clear buttons, no dashboard, no Q&A section |
| 2 | Clear the text, click **Analyze Reviews** | Error: "Please paste or upload at least one review" — no spinner, no crash |
| 3 | Paste 25,000+ characters, click **Analyze** | Error: "Input too large … Limit is 20000" |
| 4 | Upload `data/sample_reviews/sample_reviews.txt` (or `.pdf`) | "Loaded …" info; text area fills |
| 5 | Upload a `.csv` with a `review` column | Only that column's rows are used |
| 6 | Upload a `.csv` without a review-like column | All fields joined, no crash |
| 7 | Upload an empty `.csv`, an empty `.txt`, or a `.json` file | Clean error, no crash, previous state kept |
| 7b | Upload a `.pdf`, `.docx`, and `.xlsx` with review text | Text extracted; spreadsheet prefers a `review` column like CSV |
| 8 | Click **Analyze Reviews** on the sample file | Spinner 10–60 s, then: 5 metric columns, 2 charts, aspect table, points, complaints in yellow, AI summary, NLP stats expander with 61 rows |
| 9 | Check the four aspect percentages | They sum to exactly 100% |
| 10 | Click **Clear** | Text box empties, dashboard and Q&A section disappear |
| 11 | Re-analyze, keep the prefilled question, click **Ask** | An answer quoting review content appears, no error box |
| 12 | Ask "What is the price in EUR of the third unit shipped to Berlin?" | "The answer cannot be determined from the provided reviews." |
| 13 | Click **Ask** with an empty question | Error: "Please type a question first." |
| 14 | Stop the app, rename `.env` away, restart | Demo-mode banner; analysis and Q&A answer instantly with no key |
| 15 | Set a bogus `LLM_API_KEY`, restart, analyze | Friendly "API key rejected (HTTP 401)" — never a stack trace |

## 4. What "done" looks like

- `pytest`: 37 passed.
- `python -m tests.e2e_live`: 18/18 passed.
- All 15 manual rows behave as written, in both live and demo mode.
- `git status` is clean and every fix is a separate commit.
