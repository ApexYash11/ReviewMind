"""Unit tests for the mock LLM client used in demo mode."""

from app.mock_llm import MockLLMClient


client = MockLLMClient()

ANALYSIS_PROMPT = (
    "Analyze the reviews.\n"
    "Review 1: The camera is great and photos are sharp.\n"
    "Review 2: Battery life is disappointing, it drains fast.\n"
    "Review 3: The display is excellent and bright.\n"
)

QA_PROMPT = (
    "Reviews:\n"
    "Review 1: The camera is great and photos are sharp.\n"
    "Review 2: Battery life is disappointing, it drains fast.\n"
    "\n"
    "Question: What do customers complain about most?"
)


def test_chat_returns_valid_json_for_analysis():
    import json

    raw = client.chat(ANALYSIS_PROMPT)
    data = json.loads(raw)  # must not raise
    assert isinstance(data, dict)


def test_analysis_has_required_schema_fields():
    import json

    data = json.loads(client.chat(ANALYSIS_PROMPT))
    for key in (
        "overall_sentiment",
        "aspects",
        "positive_points",
        "negative_points",
        "common_complaints",
        "summary",
    ):
        assert key in data
    assert data["overall_sentiment"] in {"positive", "negative", "neutral", "mixed"}
    assert isinstance(data["aspects"], list)


def test_mock_detects_battery_complaint():
    import json

    data = json.loads(client.chat(ANALYSIS_PROMPT))
    aspect_names = {a["aspect"] for a in data["aspects"]}
    assert "Battery" in aspect_names
    battery = next(a for a in data["aspects"] if a["aspect"] == "Battery")
    assert battery["sentiment"] == "negative"
    assert data["common_complaints"]


def test_mock_detects_camera_and_display_positive():
    import json

    data = json.loads(client.chat(ANALYSIS_PROMPT))
    by_name = {a["aspect"]: a["sentiment"] for a in data["aspects"]}
    assert by_name.get("Camera") == "positive"
    assert by_name.get("Display") == "positive"


def test_qa_complaint_question():
    answer = client.chat(QA_PROMPT)
    assert isinstance(answer, str)
    assert "battery" in answer.lower()


def test_qa_insufficient_information():
    answer = client.chat(
        "Reviews:\nReview 1: Nice product.\n\nQuestion: How many languages does it support?"
    )
    assert answer == "The answer cannot be determined from the provided reviews."


def test_analyzer_uses_mock_in_demo_mode():
    """ReviewAnalyzer(demo_mode=True) must not require an API key."""
    from app.review_analyzer import ReviewAnalyzer

    analyzer = ReviewAnalyzer(config={}, demo_mode=True)
    result, stats, keywords = analyzer.analyze_reviews(ANALYSIS_PROMPT)
    assert result.overall_sentiment in {"positive", "negative", "neutral", "mixed"}
    assert len(stats) == 3
