"""Streamlit frontend for ReviewMind.

Renders the review input, the analytics dashboard and the QA section.
All heavy lifting lives in the app/ modules; this file only orchestrates
and displays.
"""

import io
import sys
from pathlib import Path

# `streamlit run app/main.py` puts the app/ directory on sys.path, not the
# project root, so `app.*` imports would fail without this.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.llm_client import LLMError, load_config
from app.review_analyzer import ReviewAnalyzer
from app.models import ValidationError

st.set_page_config(page_title="ReviewMind", page_icon=":material/sentiment_satisfied:", layout="wide")

config = load_config()


def _api_key_configured() -> bool:
    """True when a real LLM API key is available in the environment."""
    import os

    return bool(os.getenv("LLM_API_KEY", "").strip())


DEMO_MODE = not _api_key_configured()

DEFAULT_REVIEWS_TEXT = (
    "Review 1: I love the camera on this phone, photos are sharp and detailed.\n"
    "Review 2: Battery life is disappointing, I have to charge it twice a day.\n"
    "Review 3: The display is excellent, very bright and colorful."
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "reviews_text" not in st.session_state:
    st.session_state.reviews_text = DEFAULT_REVIEWS_TEXT
if "raw_reviews" not in st.session_state:
    st.session_state.raw_reviews = ""
if "stats" not in st.session_state:
    st.session_state.stats = []
if "keywords" not in st.session_state:
    st.session_state.keywords = []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".txt", ".csv", ".pdf", ".docx", ".xlsx")

REVIEW_COLUMNS = ("review", "reviews", "text", "comment", "body")


def _pick_review_column(header: list[str]) -> int | None:
    """Index of the review-like column, or None when there isn't one."""
    lowered = [h.strip().lower() for h in header]
    for candidate in REVIEW_COLUMNS:
        if candidate in lowered:
            return lowered.index(candidate)
    return None


def _join_table_rows(header: list[str], rows: list[list[str]]) -> str | None:
    """Turn table rows into review text, preferring a review-like column."""
    col = _pick_review_column(header)
    if col is None:
        joined = "\n\n".join(" ".join(filter(None, row)) for row in rows)
    else:
        joined = "\n\n".join(
            row[col] for row in rows if len(row) > col and row[col].strip()
        )
    if not joined.strip():
        st.error("No review text found in the uploaded file.")
        return None
    return joined


def read_uploaded_file(uploaded) -> str | None:
    """Read an uploaded review file and return its text content."""
    name = uploaded.name.lower()
    if name.endswith(".txt"):
        text = uploaded.read().decode("utf-8", errors="replace")
        if not text.strip():
            st.error("The text file is empty.")
            return None
        return text
    if name.endswith(".csv"):
        import csv

        raw = uploaded.read().decode("utf-8", errors="replace")
        try:
            rows = list(csv.reader(io.StringIO(raw)))
        except csv.Error:
            st.error("Could not parse the CSV file. Please check its format.")
            return None
        if not rows:
            st.error("The CSV file is empty.")
            return None
        return _join_table_rows(rows[0], rows[1:])
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(uploaded.read()))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            st.error("Could not read the PDF file. Is it a valid PDF?")
            return None
        if not text.strip():
            st.error("No readable text found in the PDF file.")
            return None
        return text
    if name.endswith(".docx"):
        from docx import Document

        try:
            document = Document(io.BytesIO(uploaded.read()))
            text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
        except Exception:
            st.error("Could not read the Word file. Is it a valid .docx?")
            return None
        if not text.strip():
            st.error("No readable text found in the Word file.")
            return None
        return text
    if name.endswith(".xlsx"):
        from openpyxl import load_workbook

        try:
            workbook = load_workbook(io.BytesIO(uploaded.read()), read_only=True)
            sheet = workbook.active
            table = [
                ["" if cell is None else str(cell) for cell in row]
                for row in sheet.iter_rows(values_only=True)
            ]
        except Exception:
            st.error("Could not read the Excel file. Is it a valid .xlsx?")
            return None
        if not table:
            st.error("The Excel file is empty.")
            return None
        return _join_table_rows(table[0], table[1:])
    st.error(
        "Unsupported file type. Please upload a .txt, .csv, .pdf, .docx or .xlsx file."
    )
    return None


SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "negative": "#e74c3c",
    "mixed": "#f39c12",
}


def aspect_chart(analysis) -> None:
    """Horizontal bar chart of aspect counts per sentiment."""
    import altair as alt
    import pandas as pd

    rows = [
        {"Aspect": a.aspect, "Sentiment": a.sentiment, "Count": 1}
        for a in analysis.aspects
    ]
    if not rows:
        st.info("No aspects to chart.")
        return
    df = pd.DataFrame(rows)
    grouped = df.groupby(["Aspect", "Sentiment"], as_index=False).sum()

    order = list(pd.unique(df["Aspect"]))
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("Count:Q", title="Mentions"),
            y=alt.Y("Aspect:N", sort=order, title=None),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS),
                    range=list(SENTIMENT_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=["Aspect", "Sentiment", "Count"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def donut_chart(analysis) -> None:
    """Donut chart of the aspect sentiment breakdown."""
    import altair as alt
    import pandas as pd

    sentiments = [a.sentiment for a in analysis.aspects] or [analysis.overall_sentiment]
    counts = pd.Series(sentiments).value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    chart = (
        alt.Chart(counts)
        .mark_arc(innerRadius=50)
        .encode(
            theta="Count:Q",
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS),
                    range=list(SENTIMENT_COLORS.values()),
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["Sentiment", "Count"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def run_analysis() -> None:
    """Execute the full pipeline and store results in session state."""
    raw = st.session_state.reviews_text.strip()
    if not raw:
        st.error("Please paste or upload at least one review before analyzing.")
        return
    max_chars = config.get("app", {}).get("max_input_chars", 20000)
    if len(raw) > max_chars:
        st.error(f"Input too large ({len(raw)} characters). Limit is {max_chars}.")
        return
    try:
        analyzer = ReviewAnalyzer(config, demo_mode=DEMO_MODE)
        with st.spinner(
            "Processing reviews (NLP + LLM)... this can take 10-60s "
            "depending on the model. Please keep this tab open."
        ):
            result, stats, keywords = analyzer.analyze_reviews(raw)
        st.session_state.analysis = result
        st.session_state.stats = stats
        st.session_state.keywords = keywords
        st.session_state.raw_reviews = raw
    except ValidationError as exc:
        st.error(str(exc))
    except LLMError as exc:
        st.error(f"LLM error: {exc}")
    except Exception as exc:  # keep stack traces away from the user
        st.error(f"Something went wrong: {exc}")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("REVIEWMIND")
st.subheader("AI Product Review Analyzer")
st.caption("Analyze customer opinions using NLP + LLM")

if DEMO_MODE:
    st.info(
        "🧪 **Demo mode** — no `LLM_API_KEY` found, so analysis is simulated "
        "locally. Add your key to `.env` (see `.env.example`) for real LLM output."
    )

st.divider()

# ----------------------------------------------------------------------
# Review input
# ----------------------------------------------------------------------
st.subheader("Review Input")

upload = st.file_uploader(
    "Upload .txt / .csv / .pdf / .docx / .xlsx",
    type=["txt", "csv", "pdf", "docx", "xlsx"],
)
if upload is not None:
    content = read_uploaded_file(upload)
    if content:
        st.session_state.reviews_text = content
        st.info(f"Loaded {upload.name} ({len(content)} characters).")

st.session_state.reviews_text = st.text_area(
    "Paste customer reviews here (one or more, optionally prefixed with 'Review N:')",
    value=st.session_state.reviews_text,
    height=180,
)

col_btn, col_clear = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("Analyze Reviews", type="primary")
with col_clear:
    if st.button("Clear"):
        st.session_state.reviews_text = ""
        st.session_state.analysis = None
        st.session_state.raw_reviews = ""
        st.session_state.stats = []
        st.session_state.keywords = []
        st.rerun()

if analyze_clicked:
    run_analysis()

# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
analysis = st.session_state.analysis
if analysis is not None:
    st.divider()
    st.subheader("Review Analytics")

    # Overall sentiment metrics
    st.markdown("**Overall Sentiment**")
    sentiment = analysis.overall_sentiment
    metric_cols = st.columns(5)
    metric_cols[0].metric("Detected", sentiment.capitalize())
    # Sentiment distribution across aspects for a quick visual.
    # All four sentiment classes are shown so the percentages sum to 100%.
    aspect_sentiments = [a.sentiment for a in analysis.aspects] or [sentiment]
    total = len(aspect_sentiments)
    # Round all but the last bucket; the last takes the remainder so the
    # displayed percentages always sum to exactly 100%.
    pct_positive = round(100 * aspect_sentiments.count("positive") / total)
    pct_neutral = round(100 * aspect_sentiments.count("neutral") / total)
    pct_negative = round(100 * aspect_sentiments.count("negative") / total)
    pct_mixed = 100 - pct_positive - pct_neutral - pct_negative
    metric_cols[1].metric("Positive aspects", f"{pct_positive}%")
    metric_cols[2].metric("Neutral aspects", f"{pct_neutral}%")
    metric_cols[3].metric("Negative aspects", f"{pct_negative}%")
    metric_cols[4].metric("Mixed aspects", f"{pct_mixed}%")

    st.divider()

    # Sentiment distribution chart
    st.markdown("**Sentiment Distribution**")
    chart_cols = st.columns([2, 1])
    with chart_cols[0]:
        aspect_chart(analysis)
    with chart_cols[1]:
        donut_chart(analysis)

    st.divider()

    # Aspect analysis table
    left, right = st.columns(2)
    with left:
        st.markdown("**Aspect Analysis**")
        if analysis.aspects:
            import pandas as pd

            df = pd.DataFrame(
                [{"Aspect": a.aspect, "Sentiment": a.sentiment.capitalize()} for a in analysis.aspects]
            )
            st.table(df)
        else:
            st.info("No product aspects were detected in these reviews.")

    with right:
        st.markdown("**Positive Points**")
        for point in analysis.positive_points:
            st.markdown(f"- {point}")
        st.markdown("**Negative Points**")
        for point in analysis.negative_points:
            st.markdown(f"- {point}")

    st.divider()

    # Complaints, summary and NLP statistics
    st.markdown("**Common Complaints**")
    if analysis.common_complaints:
        st.warning(", ".join(analysis.common_complaints))
    else:
        st.info("No recurring complaints found.")

    st.markdown("**AI Summary**")
    st.success(analysis.summary or "No summary available.")

    if st.session_state.stats:
        with st.expander("Traditional NLP statistics (preprocessing)", expanded=False):
            import pandas as pd

            stats_df = pd.DataFrame(
                [
                    {
                        "Review": s.index,
                        "Sentences": s.sentence_count,
                        "Tokens": s.token_count,
                        "Characters": s.char_count,
                        "Avg sentence length": s.avg_sentence_length,
                    }
                    for s in st.session_state.stats
                ]
            )
            st.dataframe(stats_df, width="stretch", hide_index=True)
            if st.session_state.keywords:
                kw = ", ".join(f"{word} ({count})" for word, count in st.session_state.keywords)
                st.caption(f"Top keywords after normalization: {kw}")

# ----------------------------------------------------------------------
# Ask the reviews
# ----------------------------------------------------------------------
if st.session_state.get("raw_reviews"):
    st.divider()
    st.subheader("Ask the Reviews")
    question = st.text_input(
        "Question",
        value="What is the biggest problem reported by customers?",
        key="qa_question",
    )
    if st.button("Ask"):
        if not question.strip():
            st.error("Please type a question first.")
        else:
            try:
                analyzer = ReviewAnalyzer(config, demo_mode=DEMO_MODE)
                with st.spinner("Thinking... (this usually takes 5-30 seconds)"):
                    answer = analyzer.answer_question(st.session_state.raw_reviews, question)
                st.markdown("**Answer:**")
                st.info(answer)
            except ValidationError as exc:
                st.error(str(exc))
            except LLMError as exc:
                st.error(f"LLM error: {exc}")
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
