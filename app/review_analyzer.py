"""Review analysis orchestration for ReviewMind.

Wires together the traditional NLP preprocessor, the prompt templates and
the LLM client. Validates every LLM response against the expected schema
before handing it to the Streamlit renderer.
"""

from pathlib import Path

from app.llm_client import LLMClient, LLMError, load_config
from app.models import (
    VALID_SENTIMENTS,
    AnalysisResult,
    AspectSentiment,
    ValidationError,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _validate_analysis(data: object) -> AnalysisResult:
    """Validate raw LLM JSON and coerce it into an AnalysisResult."""
    if not isinstance(data, dict):
        raise ValidationError("Expected a JSON object from the LLM.")

    sentiment = str(data.get("overall_sentiment", "")).strip().lower()
    if sentiment not in VALID_SENTIMENTS:
        raise ValidationError("Missing or invalid 'overall_sentiment' field.")

    aspects_raw = data.get("aspects", [])
    if not isinstance(aspects_raw, list):
        raise ValidationError("'aspects' must be a list.")

    aspects: list[AspectSentiment] = []
    for item in aspects_raw:
        if isinstance(item, dict) and item.get("aspect") and item.get("sentiment"):
            aspects.append(
                AspectSentiment(aspect=str(item["aspect"]), sentiment=str(item["sentiment"]))
            )

    def _string_list(key: str) -> list[str]:
        values = data.get(key, [])
        if not isinstance(values, list):
            raise ValidationError(f"'{key}' must be a list.")
        return [str(v).strip() for v in values if str(v).strip()]

    return AnalysisResult(
        overall_sentiment=sentiment,
        aspects=aspects,
        positive_points=_string_list("positive_points"),
        negative_points=_string_list("negative_points"),
        common_complaints=_string_list("common_complaints"),
        summary=str(data.get("summary", "")).strip(),
    )


class ReviewAnalyzer:
    """High-level facade: preprocess -> prompt -> LLM -> validate."""

    def __init__(self, config: dict | None = None, demo_mode: bool = False) -> None:
        from app.nlp_processor import shared_processor  # local import avoids cycles

        self.config = config or load_config()
        self.processor = shared_processor()
        self.demo_mode = demo_mode
        if demo_mode:
            from app.mock_llm import MockLLMClient

            self.client = MockLLMClient(self.config)
        else:
            self.client = LLMClient(self.config)
        self._analysis_template = load_prompt("review_analysis")
        self._qa_template = load_prompt("review_qa")

    # ------------------------------------------------------------------
    def analyze_reviews(self, raw_text: str) -> tuple[AnalysisResult, list, list[str]]:
        """Run the full analysis pipeline.

        Returns (validated result, per-review stats, top keywords).
        """
        if not raw_text or not raw_text.strip():
            raise ValidationError("No reviews found. Please paste or upload some reviews.")

        reviews, stats = self.processor.process(raw_text)
        if not reviews:
            raise ValidationError("No reviews found. Please paste or upload some reviews.")

        joined = "\n\n".join(f"Review {i}: {review}" for i, review in enumerate(reviews, 1))
        max_chars = self.config.get("app", {}).get("max_input_chars", 20000)
        if len(joined) > max_chars:
            raise ValidationError(
                f"Input too large ({len(joined)} characters). Limit is {max_chars}."
            )

        prompt = self._analysis_template.replace("{reviews}", joined)
        raw = self.client.chat(prompt, system="You analyze product reviews and return only valid JSON.")
        data = self.client.extract_json(raw)
        result = _validate_analysis(data)
        keywords = top_keywords_safe(reviews)
        return result, stats, keywords

    # ------------------------------------------------------------------
    def answer_question(self, reviews_text: str, question: str) -> str:
        """Answer a question strictly from the supplied reviews."""
        if not reviews_text or not reviews_text.strip():
            raise ValidationError("No reviews found. Please paste or upload some reviews.")
        if not question or not question.strip():
            raise ValidationError("Please type a question first.")
        # Normalize through the NLP splitter and number the reviews, exactly
        # like analyze_reviews does. The mock LLM (demo mode) extracts
        # "Review N:" lines, so passing raw unnumbered text would collapse
        # everything into one review and poison its answers.
        reviews = self.processor.split_reviews(reviews_text)
        if not reviews:
            raise ValidationError("No reviews found. Please paste or upload some reviews.")
        joined = "\n\n".join(f"Review {i}: {review}" for i, review in enumerate(reviews, 1))
        max_chars = self.config.get("app", {}).get("max_input_chars", 20000)
        if len(joined) > max_chars:
            raise ValidationError(
                f"Input too large ({len(joined)} characters). Limit is {max_chars}."
            )
        prompt = (
            self._qa_template.replace("{reviews}", joined)
            .replace("{question}", question.strip())
        )
        answer = self.client.chat(
            prompt,
            system="Answer strictly from the provided reviews. If the answer is not in them, say so.",
        )
        return answer.strip()


def top_keywords_safe(reviews: list[str]) -> list[tuple[str, int]]:
    """Wrapper around nlp_processor.top_keywords that never crashes the UI."""
    try:
        from app.nlp_processor import top_keywords

        return top_keywords(reviews)
    except Exception:
        return []
