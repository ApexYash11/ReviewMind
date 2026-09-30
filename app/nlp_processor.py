"""Traditional NLP preprocessing for ReviewMind.

This module handles text cleaning, sentence segmentation, tokenization,
normalization, review splitting and basic statistics. It deliberately
does NOT use any LLM: its job is to prepare and organize the review text
before the LLM is called.
"""

import re
import string
import unicodedata

from app.models import ReviewStats

try:  # pragma: no cover - exercised via integration, not unit tests
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NLTK_AVAILABLE = False

# Review separator: "Review 1:", "review 2:", "REVIEW 10:" ...
_REVIEW_HEADER_RE = re.compile(r"^\s*review\s*\d+\s*[:.\-]\s*", re.IGNORECASE)


def ensure_nltk_data() -> None:
    """Download the small NLTK models needed for tokenization/normalization."""
    if not _NLTK_AVAILABLE:
        return
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for name in packages:
        try:
            nltk.data.find(_nltk_path(name))
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def _nltk_path(package: str) -> str:
    """Return the lookup path used by nltk.data.find for a package."""
    mapping = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    return mapping.get(package, f"corpora/{package}")


class NlpProcessor:
    """Traditional NLP pipeline: clean -> split -> tokenize -> normalize."""

    def __init__(self) -> None:
        ensure_nltk_data()
        if _NLTK_AVAILABLE:
            try:
                self._stopwords = set(stopwords.words("english"))
            except LookupError:
                self._stopwords = set()
            self._lemmatizer = WordNetLemmatizer()
        else:
            self._stopwords = set()
            self._lemmatizer = None
        self._punct_table = str.maketrans("", "", string.punctuation)

    # ------------------------------------------------------------------
    # Step 1: text cleaning
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Normalize whitespace and unicode, strip HTML and control chars."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
        text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
        text = re.sub(r"[^\S\r\n]+", " ", text)        # collapse spaces/tabs
        text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank lines
        return text.strip()

    # ------------------------------------------------------------------
    # Step 2: sentence segmentation
    # ------------------------------------------------------------------
    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences (NLTK punkt, with regex fallback)."""
        text = self.clean_text(text)
        if not text:
            return []
        if _NLTK_AVAILABLE:
            try:
                return [s for s in sent_tokenize(text) if s.strip()]
            except LookupError:
                pass
        fallback = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in fallback if s.strip()]

    # ------------------------------------------------------------------
    # Step 3: tokenization
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        """Split text into word tokens (NLTK, with regex fallback)."""
        text = self.clean_text(text)
        if not text:
            return []
        if _NLTK_AVAILABLE:
            try:
                return [t for t in word_tokenize(text) if t.strip()]
            except LookupError:
                pass
        return re.findall(r"[A-Za-z0-9']+", text)

    # ------------------------------------------------------------------
    # Step 4: normalization
    # ------------------------------------------------------------------
    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        """Lowercase, remove punctuation/stopwords, lemmatize."""
        normalized = []
        for token in tokens:
            word = token.translate(self._punct_table).lower()
            if not word or word in self._stopwords or word.isdigit():
                continue
            if self._lemmatizer is not None:
                word = self._lemmatizer.lemmatize(word, pos="v")
                word = self._lemmatizer.lemmatize(word)
            normalized.append(word)
        return normalized

    # ------------------------------------------------------------------
    # Step 5: review splitting
    # ------------------------------------------------------------------
    def split_reviews(self, raw_text: str) -> list[str]:
        """Split pasted input into individual reviews.

        Supports explicit headers ("Review 1: ..."), blank-line blocks,
        or one review per line.
        """
        text = self.clean_text(raw_text)
        if not text:
            return []

        lines = text.splitlines()
        headers = [i for i, line in enumerate(lines) if _REVIEW_HEADER_RE.match(line)]
        if len(headers) >= 2:
            reviews: list[str] = []
            for start, end in zip(headers, [*headers[1:], None]):
                chunk = " ".join(
                    _REVIEW_HEADER_RE.sub("", line).strip()
                    for line in lines[start:end]
                    if line.strip()
                )
                if chunk.strip():
                    reviews.append(chunk.strip())
            return reviews

        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) >= 2:
            return blocks

        return [line.strip() for line in lines if line.strip()]

    # ------------------------------------------------------------------
    # Step 6: basic text statistics
    # ------------------------------------------------------------------
    def compute_statistics(self, reviews: list[str]) -> list[ReviewStats]:
        """Compute per-review statistics for the dashboard."""
        stats: list[ReviewStats] = []
        for i, review in enumerate(reviews, start=1):
            sentences = self.split_sentences(review)
            tokens = self.tokenize(review)
            stats.append(
                ReviewStats(
                    index=i,
                    text=review,
                    sentence_count=len(sentences),
                    token_count=len(tokens),
                    char_count=len(review),
                    avg_sentence_length=round(len(tokens) / len(sentences), 1)
                    if sentences
                    else 0.0,
                )
            )
        return stats

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self, raw_text: str) -> tuple[list[str], list[ReviewStats]]:
        """Run the full preprocessing pipeline and return reviews + stats."""
        reviews = self.split_reviews(raw_text)
        return reviews, self.compute_statistics(reviews)


def top_keywords(reviews: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """Most frequent normalized keywords across all reviews."""
    processor = NlpProcessor()
    counts: dict[str, int] = {}
    for review in reviews:
        tokens = processor.normalize_tokens(processor.tokenize(review))
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
