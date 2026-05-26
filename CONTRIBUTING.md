# Contributing to cavaquinho

Contributions are welcome — bug fixes, new features, documentation improvements, and benchmark additions.

## Before you start

Open an issue before submitting a pull request for significant changes. This avoids duplicated effort and lets us align on the approach before implementation.

For small fixes (typos, docs, one-line bugs), a PR without a prior issue is fine.

## Setup

```bash
git clone https://github.com/felipetp-ctrl/cavaquinho
cd cavaquinho
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                        # all tests
pytest --cov=cavaquinho       # with coverage report
```

Coverage must stay at or above **90%** for a PR to be accepted.

## Project structure

```
cavaquinho/
├── __init__.py               # public API surface
├── core.py                   # caco validator — main entry point
├── models.py                 # ClaimResult, ValidationResult, Labels
├── config.py                 # constants: thresholds, supported languages
├── aggregator.py             # weighted score aggregation
├── extractor/
│   ├── base.py               # ExtractorContract interface
│   ├── rule_extractor.py     # NLTK sentence splitter (default)
│   └── llm_extractor.py      # LLM-based claim decomposer
├── classifier/
│   ├── base.py               # ClassifierContract interface
│   └── deberta.py            # DeBERTa NLI classifier (default)
└── tests/
    └── unit/                 # unit tests (all model calls mocked)
```

## Design principles

- **Each component is independently replaceable.** `ExtractorContract` and `ClassifierContract` are the extension points. Custom implementations should require no changes to any other file.
- **No breaking changes to the public API** (`caco`, `validate`, `ClaimResult`, `ValidationResult`, `Labels`, `THRESHOLDS`) without a major version bump.
- **New optional parameters** must have a default that preserves current behaviour.
- **No new required dependencies.** `torch` and `transformers` are optional extras (`[nli]`).

## Adding a new component

Implement the relevant abstract base class:

```python
# custom extractor
from cavaquinho.extractor.base import ExtractorContract

class MyExtractor(ExtractorContract):
    def extract(self, response: str, context: str, prompt: str | None = None) -> list[str]:
        ...

# custom classifier — implement both methods for batching support
from cavaquinho.classifier.base import ClassifierContract
from cavaquinho.models import ClaimResult

class MyClassifier(ClassifierContract):
    def classify(self, claim: str, context: str) -> ClaimResult:
        ...

    def classify_batch(self, claims: list[str], context: str) -> list[ClaimResult]:
        # optional but recommended for performance
        ...
```

## Commit style

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`.

## Reporting bugs

Please include:
- Python version and OS
- cavaquinho version (`pip show cavaquinho`)
- Minimal reproducible example
- Expected vs. actual output
