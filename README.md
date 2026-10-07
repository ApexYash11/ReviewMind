# ReviewMind

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](requirements.txt) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Tests](https://img.shields.io/badge/tests-21_passing-brightgreen.svg)](#tests)

AI product review analyzer: traditional NLP preprocessing plus an LLM API, turning unstructured customer feedback into a structured dashboard.

Paste reviews (or upload a `.txt`, `.csv`, `.pdf`, `.docx` or `.xlsx` file), and ReviewMind returns overall sentiment, aspect-based sentiment (Camera, Battery, Display, …), positive/negative points, common complaints, an AI summary — and answers follow-up questions strictly from the reviews.

> **Status: working.** The NLP pipeline, prompt builders, LLM client with retries, response validation, mock demo client, Streamlit dashboard, and Q&A path are all implemented and tested (**37 unit tests passing**, plus a live end-to-end script with **18 checks** against a real model). No API key is required to try it: without one the app runs in **demo mode** with a local rule-based client that returns the same structured schema.

## Why this exists

ReviewMind is not a chatbot wrapped around reviews. It is a small pipeline with a strict division of labor: cheap deterministic code does structure, the model does judgment.

Three rules follow from that, and they shape the whole codebase:

1. **Traditional NLP does what is deterministic.** Cleaning, sentence splitting, tokenization, normalization, review splitting, and statistics never touch the model.
2. **The LLM does what needs judgment.** Aspect extraction, aspect sentiment, summarization, and question answering — always returned as validated JSON, never trusted raw.
3. **No silent failures, no hallucinated answers.** Every LLM response is validated against the schema before rendering; the UI shows friendly messages instead of stack traces; Q&A answers come strictly from the reviews or say they cannot be determined.

| Traditional NLP | LLM |
|---|---|
| Text cleaning | Understands opinions |
| Sentence segmentation | Aspect extraction |
| Tokenization | Aspect-based sentiment |
| Normalization | Summarization |
| Review splitting | Question answering |
| Text statistics | |

## Architecture

```
Streamlit UI → Review Input → NLP Preprocessor → Prompt Builder
            → LLM Client → LLM API → Structured JSON
            → Response Validator → Streamlit Renderer → Dashboard
```

## What you get

| Feature | Where it lives |
|---|---|
| Overall sentiment + per-aspect sentiment bars and donut | `app/main.py` → `aspect_chart`, `donut_chart` |
| Aspect table, positive/negative points, complaints, AI summary | `app/main.py` dashboard |
| Per-review NLP statistics + top keywords | `app/nlp_processor.py` |
| Grounded Q&A over the reviews | `app/review_analyzer.py` → `answer_question` |
| Demo mode with zero setup (no key, no network) | `app/mock_llm.py` |

## Setup

Requires Python 3.12+.

```bash
git clone https://github.com/ApexYash11/ReviewMind.git
cd ReviewMind
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

Then edit `.env` and set `LLM_API_KEY` (any OpenAI-compatible provider works), and run:

```bash
streamlit run app/main.py
```

### Demo mode (no API key)

If `LLM_API_KEY` is not set, the app automatically starts in **demo mode**: a banner is shown and a rule-based `MockLLMClient` (`app/mock_llm.py`) simulates the LLM locally with the same structured JSON output. This lets the project be demonstrated without any key or internet connection. Adding the key switches back to real LLM analysis with no code changes.

Demo mode is a keyword-based simulation, so its judgments are coarser than a real model's — it exists for UI/offline demonstration, not for analysis quality claims.

### Supported providers (OpenAI-compatible)

| Provider | LLM_BASE_URL | Example LLM_MODEL |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Local (Ollama/LM Studio) | `http://localhost:11434/v1` | `llama3.1` |

Set these via `.env` or `config/config.yaml`.

## Demo flow

1. Open the Streamlit app.
2. Paste 5–10 sample reviews (or upload a `.txt`, `.csv`, `.pdf`, `.docx` or `.xlsx` file — or use `data/sample_reviews/sample_reviews.txt`).
3. Click **Analyze Reviews**.
4. Review the dashboard: sentiment metrics, aspect charts and table, positive/negative points, complaints, AI summary, and NLP statistics.
5. Ask a question, e.g. *"What is the biggest problem reported by customers?"*
6. Explain the flow: **Reviews → NLP → Prompt → LLM API → JSON → Streamlit**.

## Tests

```bash
pytest -v        # 37 tests, no API key or network access required
```

### End-to-end test against a live LLM

`tests/e2e_live.py` exercises the real pipeline (NLP → prompt → API → JSON → validation → QA) using `data/sample_reviews/sample_reviews.txt`:

```bash
python -m tests.e2e_live     # 18 checks; needs LLM_API_KEY / LLM_MODEL / LLM_BASE_URL
```

It asserts the NLP stage, the LLM analysis, response validation, keyword extraction, the QA path, the input guards, and that an unanswerable question is not hallucinated. It costs API calls, so it is **not** collected by `pytest`. Exit code is `0` when all checks pass, `1` otherwise.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LLM returned no content: the token budget was consumed by reasoning` | Reasoning models spent all of `max_tokens` thinking | Raise `LLM_MAX_TOKENS` (try 6000+) or cap with `LLM_MAX_REASONING_TOKENS` |
| HTTP 403 with *"only available on agentic harnesses"* | The chosen `:free` model is gated to coding agents | Pick another model in `.env` |
| HTTP 429 | Free-tier rate limit | Wait and retry; the client backs off automatically |

## Error handling

The app shows friendly Streamlit messages (never stack traces) for: empty input, empty files, unsupported files, oversized input, missing/invalid API key, timeouts, rate limits, network errors, invalid JSON and missing response fields.

## Repository layout

```
reviewmind/
├── app/
│   ├── main.py            # Streamlit frontend
│   ├── llm_client.py      # Configurable OpenAI-compatible API client
│   ├── mock_llm.py        # Rule-based mock LLM for demo mode
│   ├── nlp_processor.py   # Traditional NLP preprocessing
│   ├── review_analyzer.py # Orchestration + response validation
│   └── models.py          # Dataclasses for validated results
├── prompts/
│   ├── review_analysis.txt
│   └── review_qa.txt
├── config/config.yaml     # Provider, model, timeouts, input limits
├── tests/
│   ├── test_nlp.py        # Preprocessor unit tests
│   ├── test_analyzer.py   # Validation + orchestration tests
│   ├── test_mock_llm.py   # Demo-mode client tests
│   ├── test_uploads.py    # File-upload parsing tests (txt/csv/pdf/docx/xlsx)
│   └── e2e_live.py        # Live-model end-to-end checks (not run by pytest)
├── data/sample_reviews/   # Sample corpora (.txt + .pdf)
├── .env.example
└── requirements.txt
```

## Limitations

- Free-tier models are slow (tens of seconds per analysis) and rate-limited; that is the provider, not the app.
- Demo-mode judgments are keyword-based and coarser than a real model's.
- The sample corpus is small and hand-written; no accuracy claim is made beyond "the pipeline returns validated, schema-conformant output."

## License

Apache-2.0. See [LICENSE](LICENSE).
