# ReviewMind — AI Product Review Analyzer

ReviewMind is a Streamlit-based NLP application that analyzes customer product
reviews using **traditional NLP preprocessing** plus an **LLM API**, turning
unstructured feedback into structured insights.

## Features

- Overall sentiment detection
- Aspect extraction and aspect-based sentiment (Camera, Battery, Display, ...)
- Positive / negative point extraction
- Common complaint detection
- AI review summary
- Question answering over the reviews

## Architecture

```
Streamlit UI → Review Input → NLP Preprocessor → Prompt Builder
            → LLM Client → LLM API → Structured JSON
            → Response Validator → Streamlit Renderer → Dashboard
```

**Division of labor**

| Traditional NLP | LLM |
|---|---|
| Text cleaning | Understands opinions |
| Sentence segmentation | Aspect extraction |
| Tokenization | Aspect-based sentiment |
| Normalization | Summarization |
| Review splitting | Question answering |
| Text statistics | |

## Project Structure

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
├── config/config.yaml
├── tests/
├── data/sample_reviews/
├── .env.example
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your LLM API key
cp .env.example .env
#    then edit .env and set LLM_API_KEY (any OpenAI-compatible provider works)

# 3. Run the app
streamlit run app/main.py
```

### Demo mode (no API key)

If `LLM_API_KEY` is not set, the app automatically starts in **demo mode**:
a banner is shown and a rule-based `MockLLMClient` (`app/mock_llm.py`)
simulates the LLM locally with the same structured JSON output. This lets
the project be demonstrated without any API key or internet connection —
useful for classroom demos and testing. Adding the key switches back to
real LLM analysis with no code changes.

### Supported providers (OpenAI-compatible)

| Provider | LLM_BASE_URL | Example LLM_MODEL |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Local (Ollama/LM Studio) | `http://localhost:11434/v1` | `llama3.1` |

Set these via `.env` or `config/config.yaml`.

## Demo Flow

1. Open the Streamlit app.
2. Paste 5–10 sample reviews (or upload a `.txt`/`.csv`, or use
   `data/sample_reviews/sample_reviews.txt`).
3. Click **Analyze Reviews**.
4. Review the dashboard: sentiment metrics, aspect table, positive/negative
   points, complaints, AI summary and NLP statistics.
5. Ask a question, e.g. *"What is the biggest problem reported by customers?"*
6. Explain the flow: **Reviews → NLP → Prompt → LLM API → JSON → Streamlit**.

## Running Tests

```bash
pytest -v
```

### End-to-end test against a live LLM

`tests/e2e_live.py` exercises the real pipeline (NLP → prompt → API → JSON →
validation → QA) using `data/sample_reviews/sample_reviews.txt`:

```bash
python tests/e2e_live.py
```

It asserts the NLP stage, the LLM analysis, response validation, keyword
extraction, the QA path, the input guards, and that an unanswerable question
is not hallucinated. It costs API calls, so it is **not** collected by `pytest`
(prefixed to avoid auto-discovery).

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LLM returned no content: the token budget was consumed by reasoning` | Reasoning models spent all of `max_tokens` thinking | Raise `LLM_MAX_TOKENS` (try 6000+) or cap with `LLM_MAX_REASONING_TOKENS` |
| HTTP 403 with *"only available on agentic harnesses"* | The chosen `:free` model is gated to coding agents | Pick another model in `.env` |
| HTTP 429 | Free-tier rate limit | Wait and retry; the client now backs off automatically |

## Error Handling

The app shows friendly Streamlit messages (never stack traces) for: empty
input, unsupported files, oversized input, missing/invalid API key, timeouts,
rate limits, network errors, invalid JSON and missing response fields.
