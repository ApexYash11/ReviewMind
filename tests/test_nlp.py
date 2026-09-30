"""Unit tests for the traditional NLP preprocessor."""

from app.nlp_processor import NlpProcessor


processor = NlpProcessor()


def test_clean_text_strips_urls_and_html():
    dirty = "Great phone <b>really</b>! Visit https://example.com now."
    cleaned = processor.clean_text(dirty)
    assert "https://" not in cleaned
    assert "<b>" not in cleaned
    assert "Great phone" in cleaned


def test_split_sentences_basic():
    text = "I love this phone. The camera is great! Battery is poor."
    sentences = processor.split_sentences(text)
    assert len(sentences) == 3


def test_tokenize_returns_words():
    tokens = processor.tokenize("Battery life is disappointing.")
    assert "Battery" in tokens
    assert "life" in tokens


def test_normalize_removes_stopwords_and_punct():
    normalized = processor.normalize_tokens(processor.tokenize("The battery is not charging!"))
    assert "the" not in normalized
    assert "is" not in normalized
    assert "battery" in normalized
    assert "charge" in normalized  # "charging" lemmatized to "charge"


def test_split_reviews_with_headers():
    raw = "Review 1: Great camera.\nReview 2: Poor battery.\nReview 3: Lovely display."
    reviews = processor.split_reviews(raw)
    assert len(reviews) == 3
    assert reviews[0].startswith("Great camera")
    assert reviews[2].endswith("Lovely display.")


def test_split_reviews_blank_lines():
    raw = "Great camera overall.\n\nBattery drains way too fast."
    reviews = processor.split_reviews(raw)
    assert len(reviews) == 2


def test_compute_statistics():
    reviews = ["Great camera. Sharp photos.", "Poor battery life."]
    stats = processor.compute_statistics(reviews)
    assert len(stats) == 2
    assert stats[0].sentence_count == 2
    assert stats[0].token_count > 0
    assert stats[1].char_count == len("Poor battery life.")
