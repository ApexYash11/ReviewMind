"""End-to-end smoke test against the live LLM API using sample data.

Run:  python -m tests.e2e_live
Set   LLM_API_KEY / LLM_MODEL / LLM_BASE_URL in .env before running.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm_client import LLMError, load_config  # noqa: E402
from app.models import ValidationError  # noqa: E402
from app.review_analyzer import ReviewAnalyzer  # noqa: E402

SAMPLE = ROOT / "data" / "sample_reviews" / "sample_reviews.txt"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" -> {detail}" if detail else ""))


def main() -> int:
    config = load_config()
    llm = config["llm"]
    print(f"provider={llm['provider']} model={llm['model']} base_url={llm['base_url']}")
    print("-" * 60)

    if not llm.get("model") or not llm.get("base_url"):
        check("config: model + base_url present", False, "missing LLM_MODEL/LLM_BASE_URL")
        return 1

    text = SAMPLE.read_text(encoding="utf-8")
    check("data: sample file loaded", len(text) > 0, f"{len(text)} chars")

    analyzer = ReviewAnalyzer(config, demo_mode=False)
    check("app: analyzer built in live mode", analyzer.client.__class__.__name__ == "LLMClient")

    # 1. NLP preprocessing (offline stage)
    reviews, stats = analyzer.processor.process(text)
    check("nlp: reviews split", len(reviews) > 10, f"{len(reviews)} reviews")
    check("nlp: per-review stats", len(stats) == len(reviews), f"{len(stats)} stat rows")
    check(
        "nlp: stats fields populated",
        all(s.token_count > 0 and s.char_count > 0 for s in stats),
    )

    # 2. LLM analysis
    t0 = time.time()
    try:
        result, stats, keywords = analyzer.analyze_reviews(text)
    except (LLMError, ValidationError) as exc:
        check("llm: analyze_reviews", False, f"{type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - t0
    check("llm: analyze_reviews", True, f"{elapsed:.1f}s")

    check("validate: overall sentiment valid", result.overall_sentiment in
          {"positive", "negative", "neutral", "mixed"}, result.overall_sentiment)
    check("validate: aspects returned", len(result.aspects) >= 3, f"{len(result.aspects)} aspects")
    check(
        "validate: aspect sentiments valid",
        all(a.sentiment in {"positive", "negative", "neutral", "mixed"} for a in result.aspects),
    )
    check("validate: positive points", len(result.positive_points) >= 1, f"{len(result.positive_points)}")
    check("validate: negative points", len(result.negative_points) >= 1, f"{len(result.negative_points)}")
    check("validate: complaints", len(result.common_complaints) >= 1, f"{len(result.common_complaints)}")
    check("validate: summary non-empty", len(result.summary) > 20, f"{len(result.summary)} chars")
    check("nlp: top keywords", len(keywords) >= 3, ", ".join(f"{w}({c})" for w, c in keywords[:5]))

    # 3. QA stage
    t1 = time.time()
    try:
        answer = analyzer.answer_question(text, "What is the biggest problem reported by customers?")
    except LLMError as exc:
        check("llm: answer_question", False, str(exc))
        answer = ""
    else:
        check("llm: answer_question", len(answer) > 20, f"{time.time() - t1:.1f}s")

    # 4. Negative path: empty input must fail fast without an API call
    try:
        analyzer.analyze_reviews("   ")
        check("guard: empty input rejected", False, "no error raised")
    except ValidationError as exc:
        check("guard: empty input rejected", True, str(exc))

    # 5. Negative path: oversized input
    try:
        analyzer.analyze_reviews("x" * (config["app"]["max_input_chars"] + 100))
        check("guard: oversized input rejected", False, "no error raised")
    except ValidationError as exc:
        check("guard: oversized input rejected", True, str(exc)[:60])

    # 6. QA grounding: unanswerable question
    try:
        unans = analyzer.answer_question(text, "What is the price in EUR of the third unit shipped to Berlin?")
        grounded = "cannot be determined" in unans.lower()
        check("guard: unanswerable question not hallucinated", grounded, unans[:90])
    except LLMError as exc:
        check("guard: unanswerable question not hallucinated", False, str(exc))

    print("-" * 60)
    print("=== RESULT ===")
    print(f"Overall sentiment : {result.overall_sentiment}")
    print("Aspects           :")
    for a in result.aspects:
        print(f"   - {a.aspect:<18} {a.sentiment}")
    print(f"Positive points   : {result.positive_points[:5]}")
    print(f"Negative points   : {result.negative_points[:5]}")
    print(f"Complaints        : {result.common_complaints[:5]}")
    print(f"Summary           : {result.summary}")
    print(f"QA answer         : {answer[:300]}")
    print("-" * 60)

    failed = [r for r in results if r[0] == FAIL]
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
