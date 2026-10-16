"""Real training lines (training/vi_ocr/doc_lines.py): the parts that need no documents."""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "vi_ocr"))

import doc_lines  # noqa: E402


def test_every_language_gets_the_same_share_of_pages():
    # the corpus: far more Japanese documents than English or Vietnamese ones
    docs = [(f"ja{i}.pdf", 10, "ja") for i in range(160)] + [(f"en{i}.pdf", 10, "en") for i in range(24)]
    docs += [(f"vi{i}.pdf", 10, "vi") for i in range(10)]
    draws = doc_lines.page_draws(docs, 3000, random.Random(1))
    shares = Counter(path[:2] for path, _, _ in draws)
    assert all(abs(shares[lang] / 3000 - 1 / 3) < 0.04 for lang in ("ja", "en", "vi"))
    assert all(0 <= page < 10 for _, page, _ in draws)
    assert draws == doc_lines.page_draws(docs, 3000, random.Random(1))  # reproducible from the seed
