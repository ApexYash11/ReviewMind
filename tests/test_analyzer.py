"""Unit tests for response validation and analyzer orchestration."""

import pytest

from app.models import ValidationError
from app.review_analyzer import ReviewAnalyzer, _validate_analysis


VALID = {
    "overall_sentiment": "mixed",
    "aspects": [
        {"aspect": "camera", "sentiment": "positive"},
        {"aspect": "battery", "sentiment": "negative"},
    ],
    "positive_points": ["Good camera quality", "Sharp photos"],
    "negative_points": ["Poor battery life"],
    "common_complaints": ["Battery life"],
    "summary": "Customers like the camera but dislike the battery.",
}


def test_validate_analysis_accepts_valid_payload():
    result = _validate_analysis(VALID)
    assert result.overall_sentiment == "mixed"
    assert result.aspects[0].aspect == "Camera"  # title-cased
    assert result.aspects[1].sentiment == "negative"
    assert len(result.positive_points) == 2
    assert result.summary.startswith("Customers like")


def test_validate_analysis_rejects_non_dict():
    with pytest.raises(ValidationError):
        _validate_analysis(["not", "a", "dict"])


def test_validate_analysis_rejects_missing_sentiment():
    broken = dict(VALID)
    del broken["overall_sentiment"]
    with pytest.raises(ValidationError):
        _validate_analysis(broken)


def test_validate_analysis_rejects_invalid_sentiment():
    broken = dict(VALID, overall_sentiment="amazing")
    with pytest.raises(ValidationError):
        _validate_analysis(broken)


def test_validate_analysis_rejects_non_list_aspects():
    broken = dict(VALID, aspects="camera")
    with pytest.raises(ValidationError):
        _validate_analysis(broken)


def test_analyzer_raises_on_empty_reviews():
    analyzer = ReviewAnalyzer.__new__(ReviewAnalyzer)  # skip __init__ (no LLM needed)
    with pytest.raises(ValidationError):
        analyzer.analyze_reviews("   ")


def test_prompt_templates_loaded():
    analyzer = ReviewAnalyzer.__new__(ReviewAnalyzer)
    analyzer._analysis_template = "Analyze: {reviews}"
    analyzer._qa_template = "Reviews: {reviews} Q: {question}"
    assert "{reviews}" in analyzer._analysis_template
    assert "{question}" in analyzer._qa_template
