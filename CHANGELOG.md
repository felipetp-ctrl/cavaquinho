# Changelog

All notable changes to cavaquinho are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.2.5] — 2026-05-27

### Added
- `ruff` as mandatory CI gate (zero violations enforced on every push)
- `TestAggregateDocumentedLimitations`: pins edge-case behaviour (0 claims, 100 % contradictions, threshold boundary)
- `TestDocumentedLimitations` in extractor: regression tests for limitations listed in README
- `TestCacoAlias`: verifies `DeprecationWarning` on legacy alias
- `benchmarks/faithfulness_benchmark.py`: evaluates full pipeline on HaluEval QA dataset
- `docs/evaluation.md`: evaluation methodology, metrics, and local reproduction instructions
- `docs/PRD.md`: product roadmap v0.3 → v1.0

### Changed
- Renamed class `caco` → `Caco` (PEP 8); `caco` kept as alias with `DeprecationWarning`
- Refactored `Aggregator`: replaced `_build_summary_en` / `_build_summary_pt` with a single method backed by a language-keyed template dict
- Documented score semantics explicitly (weighted mean; dilution on sparse contradiction is intentional behaviour)
- Pinned dependency upper bounds: `transformers<5`, `torch<3`, `nltk<4`
- Added author email and `Changelog` URL to PyPI metadata

### Fixed
- CI coverage configuration: omit CLI and live-model classifiers; add previously missing unit tests

---

## [0.2.4] — 2026-05-26

### Added
- **CLI** — `cavaquinho validate` command. Options: `--response`/`--response-file`, `--context`/`--context-file` (accepts `-` for stdin), `--json`, `--threshold`, `--language`. Exit codes: `0` = ok, `1` = hallucination, `2` = error.
- **`Validator` class** — PascalCase name for `caco`; `caco` preserved as backwards-compatible alias.
- **`validate_batch()`** — validate multiple responses against a shared or per-response context in one call.
- **`classify_batch()`** on `DeBERTaClassifier` — all (sentence, claim) pairs submitted in a single batched pipeline call.
- **`preload()`** on `Validator` — warm-up hook for explicit server initialisation.
- **`__version__`** exported from `cavaquinho`.
- **`__repr__`** on `ClaimResult` and `ValidationResult` — compact single-line output.
- **`__str__`** on `ValidationResult` — returns `result.summary`.
- **`MiniCheckClassifier.neutral_band`** — maps borderline P(supported) scores to `VALUE_NEUTRAL`. Default: `(0.4, 0.6)`.
- Subpackage re-exports: `from cavaquinho.extractor import LLMExtractor` and `from cavaquinho.classifier import DeBERTaClassifier` now work as documented.

### Fixed
- **`DeBERTaClassifier` label selection** — any detected contradiction now wins over a higher-confidence neutral from another sentence.
- **`LLMExtractor` JSON parser** — fence stripping handles ` ```json`, ` ```python`, leading spaces, and other variants.

### Changed
- `Validator.validate()` uses `classify_batch()` instead of `ThreadPoolExecutor` + `classify()`.

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

---

## [0.2.2] — 2026-05-25

### Fixed
- Empty wheel on PyPI — source files moved into `cavaquinho/` package directory.

---

## [0.2.1] — 2026-05-20

### Changed
- Version bump to publish updated README to PyPI (no code changes).

---

## [0.2.0] — 2026-05-25

### Added
- 65 unit tests covering `models`, `aggregator`, `extractor`, and `core` without requiring a real NLI model. Coverage: 92.8 %.
- GitHub Actions CI: test matrix Python 3.10–3.13, coverage gate ≥ 80 %.
- GitHub Actions publish: PyPI trusted publishing on release tag.
- `benchmarks/nli_benchmark.py`: evaluates NLI classifiers on ASSIN2-PT validation split.
- Full docstrings on all public APIs.

### Changed
- `ValidationResult.claims` is now `tuple[ClaimResult, ...]` (immutable).
- `torch` moved to optional dependency — `pip install "cavaquinho[nli]"`.
- `DeBERTaClassifier` detects MPS and uses it automatically on Apple Silicon.
- `ThreadPoolExecutor` worker count bounded at `min(n_claims, cpu_count)`.
- `LLMExtractor.max_claims` default raised from 10 to 20.

### Fixed
- `threshold` now correctly forwarded from `caco()` to `Aggregator`.
- Evidence selection returns best overall result when no contradiction is found.
- NLI input uses `{"text": ..., "text_pair": ...}` dict instead of raw `[SEP]` concatenation.
- `reason` field surfaces the contradicting sentence, not the label string.
- `LLMExtractor` logs a warning on fallback instead of silently swallowing the error.
- `max_workers` floored at 1 to avoid `ValueError` on empty claim list.

---

## [0.1.0] — 2026-05-23

Initial release.

- Claim extraction via NLTK sentence tokenisation (`RuleExtractor`) and LLM-based atomic decomposition (`LLMExtractor`)
- NLI classification via `cross-encoder/nli-deberta-v3-base`, fully local
- Label-weighted score aggregation with configurable threshold
- English and Portuguese language support
- Pluggable extractor and classifier interfaces (`BaseExtractor`, `BaseClassifier`, `BaseAggregator`)

[Unreleased]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/felipetp-ctrl/cavaquinho/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/felipetp-ctrl/cavaquinho/releases/tag/v0.1.0
