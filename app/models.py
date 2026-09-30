"""Data models for ReviewMind.

These dataclasses represent the validated output of the LLM and the
statistics computed by the traditional NLP preprocessor.
"""

from dataclasses import dataclass, field


VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}


@dataclass
class ReviewStats:
    """Basic text statistics for a single review (traditional NLP)."""

    index: int
    text: str
    sentence_count: int
    token_count: int
    char_count: int
    avg_sentence_length: float


@dataclass
class AspectSentiment:
    """Sentiment associated with one product aspect."""

    aspect: str
    sentiment: str

    def __post_init__(self) -> None:
        self.aspect = str(self.aspect).strip().title()
        sentiment = str(self.sentiment).strip().lower()
        self.sentiment = sentiment if sentiment in VALID_SENTIMENTS else "neutral"


@dataclass
class AnalysisResult:
    """Validated, structured result of the LLM review analysis."""

    overall_sentiment: str
    aspects: list[AspectSentiment] = field(default_factory=list)
    positive_points: list[str] = field(default_factory=list)
    negative_points: list[str] = field(default_factory=list)
    common_complaints: list[str] = field(default_factory=list)
    summary: str = ""


class ValidationError(Exception):
    """Raised when the LLM response does not match the expected schema."""
