# Changelog

All notable changes to cavaquinho are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.2.4] — 2026-05-26

### Added
- **CLI** — `cavaquinho validate` command for one-off validation from the terminal or CI pipelines. `--response`/`--response-file`, `--context`/`--context-file` (accepts `-` for stdin), `--json`, `--threshold`, `--language`. Exit codes: `0` = ok, `1` = hallucination, `2` = error.
- **`Validator` class** — PascalCase name for `caco`. `caco` preserved as backwards-compatible alias.
- **`validate_batch()`** — validate multiple responses against a shared or per-response context in one call.
- **`classify_batch()`** on `DeBERTaClassifier` — all (sentence, claim) pairs submitted in a single batched pipeline call, maximising GPU/MPS throughput.
- **`preload()`** on `Validator` — no-op warm-up hook for explicit server initialisation.
- **`__version__`** exported from `cavaquinho`.
- **`__repr__`** on `ClaimResult` and `ValidationResult` — compact single-line output for REPL use.
- **`__str__`** on `ValidationResult` — returns `result.summary`.
- **`MiniCheckClassifier.neutral_band`** — maps borderline P(supported) scores to `VALUE_NEUTRAL`. Default: `(0.4, 0.6)`.
- Subpackage re-exports: `from cavaquinho.extractor import LLMExtractor` and `from cavaquinho.classifier import DeBERTaClassifier` now work as documented.

### Fixed
- **`DeBERTaClassifier` label selection** — a high-confidence `neutral` score from one context sentence could previously suppress a lower-score `contradiction` from another. Now any detected contradiction wins.
- **`LLMExtractor` JSON parser** — fence stripping now handles ` ```json`, ` ```python`, leading spaces, and other fence variants that caused `JSONDecodeError`.

### Changed
- `Validator.validate()` uses `classify_batch()` instead of `ThreadPoolExecutor` + `classify()`. Faster on GPU/MPS; no thread overhead on CPU.

---

## [0.2.3] — 2026-05-26

### Added
- `Validator` class (PascalCase) with `caco` alias.
- `preload()` on `Validator`.
- Subpackage re-exports for `extractor` and `classifier`.

### Fixed
- `score=1.0` example in README corrected to `score=0.9998`.

### Changed
- README: "Detection limits and sensitivity" section with threshold tuning guide.
- All README examples updated to `from cavaquinho import Validator`.

---

## [0.2.2] — 2026-05-25

### Fixed
- Empty wheel on PyPI — source files moved into `cavaquinho/` package directory.

---

## [0.2.1] — 2026-05-20

### Changed
- Version bump to publish updated README to PyPI.

---

## [0.2.0] — 2026-05-25

### Fixed

- **Threshold propagation** — `caco(threshold=X)` now correctly forwards the
  configured value to `Aggregator`.  Previously the threshold was stored on the
  `caco` instance but the default `Aggregator` always used `DEFAULT_THRESHOLD`,
  silently ignoring the user's setting.

- **Evidence selection** — `DeBERTaClassifier.classify` now correctly returns
  the best available result when no contradiction is found.  The previous
  implementation tracked only the highest contradiction score, so claims with
  strong entailment evidence were returned as `neutral` with `score=0.0`.

- **NLI input format** — changed from raw string concatenation
  (`f"{sentence} [SEP] {claim}"`) to the canonical `{"text": ..., "text_pair":
  ...}` dict accepted by the HuggingFace cross-encoder pipeline.

- **`reason` field now surfaces evidence** — the `ClaimResult.reason` field for
  contradictions now contains the contradicting context sentence instead of
  repeating the label string.  Summary messages were updated accordingly.

- **Silent exception swallowing in `LLMExtractor`** — failures in the LLM call
  now emit a `logging.WARNING` with the exception type and message before
  falling back to `RuleExtractor`.

- **`max_workers=0` crash when no claims are extracted** — `ThreadPoolExecutor`
  received `max_workers=0` when the extractor returned an empty list, raising a
  `ValueError`.  The value is now floored at 1.

### Changed

- **`ValidationResult.claims` is now `tuple[ClaimResult, ...]`** — previously
  `list[ClaimResult]`, which allowed mutation despite the dataclass being
  `frozen=True`.  This is a breaking change for code that mutated the list;
  iteration and indexing are unaffected.

- **`torch` moved to optional dependency** — install with
  `pip install "cavaquinho[nli]"` to include `transformers` and `torch`.  The
  default install no longer pulls ~2 GB of PyTorch for users who supply a
  custom classifier.

- **Apple Silicon (MPS) device support** — `DeBERTaClassifier` now detects
  `torch.backends.mps.is_available()` and uses `"mps"` automatically when
  neither CUDA nor explicit device is configured.

- **`ThreadPoolExecutor` worker count bounded** — capped at
  `min(n_claims, cpu_count)` to avoid unbounded thread creation on long
  responses.

- **`LLMExtractor.max_claims` default raised from 10 to 20** — the previous
  limit could silently under-report contradictions in longer responses.

- **Prompt example typo corrected** — "Spiders has 8 legs" →
  "Spiders have 8 legs."

### Added

- **Unit test suite** — 65 tests covering `models`, `aggregator`, `extractor`,
  and `core` without requiring a real NLI model.

- **CI workflow** (`ci.yml`) — runs the test suite on Python 3.10–3.13 via
  GitHub Actions, with coverage enforcement (≥ 80 %).

- **Publish workflow** (`publish.yml`) — builds and publishes to PyPI on GitHub
  release using trusted publishing (no stored API token required).

- **Benchmark script** (`benchmarks/nli_benchmark.py`) — evaluates NLI
  classifier models on the ASSIN2 Portuguese validation split (500 pairs).
  Benchmark results for `cross-encoder/nli-deberta-v3-base` vs.
  `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` are documented in the README.

- **Docstrings** — all public classes, methods, and module-level constants now
  have full docstrings.

---

## [0.1.0] — 2025-03-01

Initial release.

- Claim extraction via NLTK sentence tokenisation
- NLI classification via `cross-encoder/nli-deberta-v3-base`, fully local
- Label-weighted score aggregation with configurable threshold
- English and Portuguese language support
- Pluggable extractor and classifier interfaces
- `LLMExtractor` for LLM-based atomic claim decomposition
