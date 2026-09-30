"""Mock LLM client for ReviewMind demo mode.

Simulates the OpenAI-compatible chat API so the app can be demonstrated
without an API key or internet connection. It performs a lightweight
rule-based analysis of the preprocessed reviews and returns the same
structured JSON / plain-text answers the real LLM would produce.
"""

import json
import re
import time

from app.llm_client import LLMClient

# Words the mock analyzer uses to score sentences/aspects.
POSITIVE_WORDS = {
    "love", "loved", "great", "excellent", "amazing", "good", "sharp",
    "bright", "colorful", "premium", "solid", "impressive",
    "surprisingly", "natural", "detailed", "recommend", "beautiful",
    "smooth", "worth", "easy", "clear", "loud",
}
NEGATIVE_WORDS = {
    "disappointing", "poor", "bad", "worst", "slow", "drain",
    "heat", "hot", "overheat", "annoying", "fails", "fail", "broken",
    "weak", "expensive", "overpriced", "lag", "crash", "crashes",
    "problem", "issue", "terrible", "horrible", "waste",
}

# Common product aspects to look for in the text.
ASPECT_KEYWORDS = {
    "Camera": ["camera", "photo", "photos", "picture", "pictures", "lens", "video"],
    "Battery": ["battery", "charge", "charging", "charger", "drain", "drains"],
    "Display": ["display", "screen", "brightness", "colors", "colour", "color"],
    "Sound": ["sound", "speaker", "speakers", "audio", "volume", "call", "calls"],
    "Performance": ["performance", "speed", "fast", "slow", "lag", "gaming", "heat", "hot", "overheat"],
    "Design": ["design", "build", "look", "looks", "premium", "feel", "feels", "weight"],
    "Price": ["price", "cost", "expensive", "cheap", "worth", "value"],
    "Setup": ["setup", "install", "installation", "pairing"],
}


class MockLLMClient(LLMClient):
    """Drop-in replacement for LLMClient that never calls the network."""

    def __init__(self, config: dict | None = None) -> None:
        # Intentionally skip LLMClient.__init__: no config/API key needed.
        self.model = "mock-analyzer"

    # ------------------------------------------------------------------
    def chat(self, prompt: str, system: str | None = None) -> str:
        """Return a simulated API response with realistic latency."""
        time.sleep(1.2)  # small pause so the spinner is visible in the demo
        if "Question:" in prompt:
            return self._answer_question(prompt)
        return json.dumps(self._analyze(prompt))

    # ------------------------------------------------------------------
    # Analysis simulation
    # ------------------------------------------------------------------
    def _analyze(self, prompt: str) -> dict:
        reviews = self._extract_reviews(prompt)
        aspect_scores = self._score_aspects(reviews)

        aspects = [
            {"aspect": name, "sentiment": self._label(score)}
            for name, score in sorted(aspect_scores.items(), key=lambda kv: -abs(kv[1]))
        ]

        positive_points, negative_points, complaints = self._collect_points(reviews, aspect_scores)
        overall = self._overall_label(aspect_scores, positive_points, negative_points)
        summary = self._build_summary(overall, aspects, complaints)

        return {
            "overall_sentiment": overall,
            "aspects": aspects,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "common_complaints": complaints,
            "summary": summary,
        }

    def _extract_reviews(self, prompt: str) -> list[str]:
        """Pull the 'Review N: ...' lines out of the analysis prompt."""
        found = re.findall(r"Review \d+: (.+)", prompt)
        return [text.strip() for text in found if text.strip()] or [prompt.strip()]

    @staticmethod
    def _word_variants(word: str) -> list[str]:
        """Cheap stemming: also try the word without common suffixes."""
        variants = [word]
        if len(word) > 3 and word.endswith("s"):
            variants.append(word[:-1])
        if len(word) > 4 and word.endswith("es"):
            variants.append(word[:-2])
        if len(word) > 5 and word.endswith("ing"):
            variants.append(word[:-3])
        if len(word) > 4 and word.endswith("ed"):
            variants.append(word[:-2])
        return variants

    def _score_text(self, text: str) -> int:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        score = 0
        for token in tokens:
            variants = self._word_variants(token)
            if any(v in POSITIVE_WORDS for v in variants):
                score += 1
            elif any(v in NEGATIVE_WORDS for v in variants):
                score -= 1
        return score

    def _score_aspects(self, reviews: list[str]) -> dict[str, float]:
        """Net sentiment score per aspect (positive > 0, negative < 0)."""
        scores: dict[str, float] = {}
        mentions: dict[str, int] = {}
        for review in reviews:
            lowered = review.lower()
            score = self._score_text(review)
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(k in lowered for k in keywords):
                    scores[aspect] = scores.get(aspect, 0) + score
                    mentions[aspect] = mentions.get(aspect, 0) + 1
        # Normalize by mention count so one strongly-worded review
        # does not dominate; keep sign.
        return {
            aspect: scores[aspect] / max(1, mentions[aspect] ** 0.5)
            for aspect in scores
        }

    @staticmethod
    def _label(score: float) -> str:
        if score > 0.3:
            return "positive"
        if score < -0.3:
            return "negative"
        return "neutral"

    def _collect_points(
        self, reviews: list[str], aspect_scores: dict[str, float]
    ) -> tuple[list[str], list[str], list[str]]:
        positive: list[str] = []
        negative: list[str] = []
        for review in reviews:
            score = self._score_text(review)
            clean = review[0].upper() + review[1:] if review else review
            if score > 0:
                positive.append(clean)
            elif score < 0:
                negative.append(clean)

        complaints = [
            aspect
            for aspect, score in sorted(aspect_scores.items(), key=lambda kv: kv[1])
            if score < -0.3
        ]
        return positive[:6], negative[:6], complaints[:3]

    @staticmethod
    def _overall_label(
        aspect_scores: dict[str, float],
        positive: list[str],
        negative: list[str],
    ) -> str:
        if not aspect_scores:
            return "neutral"
        avg = sum(aspect_scores.values()) / len(aspect_scores)
        if avg > 0.3 and not negative:
            return "positive"
        if avg < -0.3 and not positive:
            return "negative"
        if abs(avg) <= 0.3:
            return "mixed" if positive and negative else "neutral"
        return "mixed"

    @staticmethod
    def _build_summary(overall: str, aspects: list[dict], complaints: list[str]) -> str:
        liked = [a["aspect"] for a in aspects if a["sentiment"] == "positive"]
        disliked = [a["aspect"] for a in aspects if a["sentiment"] == "negative"]
        parts = []
        if liked:
            parts.append(f"Customers generally like the {', '.join(liked[:3])}")
        if disliked:
            parts.append(f"while {', '.join(disliked[:3])} draw criticism")
        if complaints and complaints[0] not in disliked:
            parts.append(f"The most common complaint is {complaints[0].lower()}")
        text = ", ".join(parts) if parts else "Feedback is largely neutral."
        return text.rstrip(",") + f". Overall sentiment: {overall}."

    # ------------------------------------------------------------------
    # QA simulation
    # ------------------------------------------------------------------
    def _answer_question(self, prompt: str) -> str:
        question = prompt.split("Question:")[-1].strip().rstrip()
        reviews = self._extract_reviews(prompt.split("Question:")[0])
        lowered = question.lower()

        if not reviews or not any(self._score_text(r) != 0 or r for r in reviews):
            return "The answer cannot be determined from the provided reviews."

        aspect_scores = self._score_aspects(reviews)
        positive, negative, complaints = self._collect_points(reviews, aspect_scores)

        if any(w in lowered for w in ("complain", "problem", "issue", "worst", "biggest")):
            if not complaints and not negative:
                return "No clear complaints appear in the provided reviews."
            focus = complaints[0] if complaints else "overall quality"
            detail = negative[0] if negative else f"Issues with {focus.lower()}"
            return f"The most common complaint is {focus.lower()}. Example: \"{detail}\""

        if any(w in lowered for w in ("like", "love", "best", "good", "praise", "positive", "strength")):
            if not positive:
                return "No clearly positive feedback appears in the provided reviews."
            return f"Customers like: {'; '.join(positive[:3])}"

        # Aspect-specific question: does any aspect name appear in the question?
        for aspect in ASPECT_KEYWORDS:
            if aspect.lower() in lowered:
                score = aspect_scores.get(aspect)
                if score is None:
                    return f"The answer cannot be determined from the provided reviews."
                label = self._label(score)
                return f"Sentiment about the {aspect.lower()} is {label} (mentioned in the reviews)."

        return "The answer cannot be determined from the provided reviews."
