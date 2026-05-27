<h1 align="center">cavaquinho</h1>

<p align="center">
  <b>Faithfulness hallucination detection for LLM responses</b>
</p>

<p align="center">
  <a href="https://github.com/felipetp-ctrl/cavaquinho/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/cavaquinho" alt="License"></a>
  <a href="https://pypi.org/project/cavaquinho/"><img src="https://img.shields.io/pypi/v/cavaquinho?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://github.com/felipetp-ctrl/cavaquinho/actions/workflows/ci.yml"><img src="https://github.com/felipetp-ctrl/cavaquinho/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://arxiv.org/abs/2502.08109"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="HuDEx paper"></a>
</p>

---

Most hallucination detectors return a single score. Cavaquinho tells you **which claim contradicts which sentence** — and runs fully local, with no external API required.

Cavaquinho decomposes a response into atomic claims, verifies each one against a provided context using Natural Language Inference, and returns a structured result with per-claim labels, scores, and evidence.

Inspired by [HuDEx: Integrating Hallucination Detection and Explainability](https://arxiv.org/abs/2502.08109).

## Installation

```bash
# Default — no torch required (bring your own classifier)
pip install cavaquinho

# With local DeBERTa NLI classifier
pip install "cavaquinho[nli]"
```

## Quickstart

```python
from cavaquinho import Validator

# The NLI model (~500 MB) loads here — instantiate once and reuse.
validator = Validator()

result = validator.validate(
    response="The LGPD was created in 2015 during Dilma Rousseff's government.",
    context="The LGPD was enacted on August 14, 2018, by President Michel Temer."
)

print(result.score)             # 0.9998  (≈ 1.0 for strong contradictions)
print(result.is_hallucination)  # True
print(result.summary)
# 1 of 1 claim(s) contradict the provided context.
# Main conflict: 'The LGPD was created in 2015...'
# contradicting evidence: 'The LGPD was enacted on August 14, 2018...'. Score: 1.00.
```

## CLI

```bash
cavaquinho validate \
  --response "The LGPD was created in 2015." \
  --context "The LGPD was enacted in 2018 by Michel Temer."

# JSON output for pipelines
cavaquinho validate --response "..." --context "..." --json

# Files and stdin
cavaquinho validate --response-file resp.txt --context-file ctx.txt
cavaquinho validate --response "..." --context-file - < ctx.txt
```

Exit codes: `0` = no hallucination, `1` = hallucination detected, `2` = error.

## Claim-level inspection

Each claim in the response is verified independently. The result exposes the full trace:

```python
for claim in result.claims:
    print(claim.text)      # "The LGPD was created in 2015..."
    print(claim.label)     # Labels.VALUE_CONTRADICTION
    print(claim.score)     # 0.9998  (model confidence, rounded to 4 decimal places)
    print(claim.evidence)  # "The LGPD was enacted on August 14, 2018..."
    print(claim.reason)    # "The LGPD was enacted on August 14, 2018..."
```

## Using the result

```python
from cavaquinho import Validator, Labels

validator = Validator()
result = validator.validate(response=response, context=context)

# threshold-based decision
if result.is_hallucination:
    response = retry_generation()

# per-claim decision
for claim in result.claims:
    if claim.label == Labels.VALUE_CONTRADICTION and claim.score > 0.8:
        logger.warning(f"Conflicting claim: {claim.text}")
        logger.warning(f"Context evidence: {claim.evidence}")

# score-based soft warning
if result.score > 0.3:
    ui.show_disclaimer("This response may contain inaccurate information.")
```

## Batch validation

```python
validator = Validator()

results = validator.validate_batch(
    responses=["Response A.", "Response B.", "Response C."],
    context="The shared retrieved context.",
)

# Or with per-response contexts
results = validator.validate_batch(
    responses=responses,
    contexts=contexts,  # list[str], same length as responses
)
```

## Detection limits and sensitivity

Cavaquinho verifies *faithfulness to the provided context*, not factual accuracy against external knowledge.

**When detection works best:**
- Claims with specific details that directly contradict facts stated in the context.
- `"The LGPD was created in 2015 during Dilma Rousseff's government."` detects better than `"The LGPD was created in 2015."` — more detail gives the NLI model stronger signal.

**Known false-negative patterns:**
- **Short or vague claims** — less context for the model to infer a contradiction. Score may sit just below the threshold.
- **Factual errors absent from context** — if the context doesn't contradict the claim, the result will be `NEUTRAL`, not `CONTRADICTION`. This is correct faithfulness behaviour, not a miss.
- **True claims near the threshold** — a faithful claim may score `0.4999` instead of `ENTAILMENT`; inspect `result.score` numerically for critical pipelines.

**Adjusting sensitivity with `threshold`:**

```python
# More sensitive — catches more hallucinations, more false positives
validator = Validator(threshold=0.3)

# More conservative — fewer false positives, may miss weak contradictions
validator = Validator(threshold=0.7)
```

For critical pipelines, inspect `result.score` directly instead of relying solely on `result.is_hallucination`.

## RAG integration

The context parameter accepts the documents your retriever already returned. No additional steps are required.

### LangChain

```python
from cavaquinho import Validator

source_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in source_docs])
response = llm.invoke(query)

validator = Validator()
result = validator.validate(response=response, context=context)
```

### LlamaIndex

```python
from cavaquinho import Validator

response = query_engine.query("What is the LGPD?")
context = "\n".join([node.text for node in response.source_nodes])

validator = Validator()
result = validator.validate(response=str(response), context=context)
```

### Direct LLM call

```python
from cavaquinho import Validator
from openai import OpenAI

client = OpenAI()
docs = retriever.search(query)
context = "\n".join(docs)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Context: {context}"},
        {"role": "user", "content": query}
    ]
).choices[0].message.content

validator = Validator()
result = validator.validate(response=response, context=context)
```

## Configuration

All components have sensible defaults. Each can be replaced independently.

```python
from cavaquinho import Validator
from cavaquinho.extractor import LLMExtractor
from cavaquinho.classifier import DeBERTaClassifier
from cavaquinho.aggregator import Aggregator
from openai import OpenAI

validator = Validator(
    extractor=LLMExtractor(
        llm_fn=lambda p: OpenAI().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": p}]
        ).choices[0].message.content
    ),
    classifier=DeBERTaClassifier(
        model_name="cross-encoder/nli-deberta-v3-base"
    ),
    aggregator=Aggregator(
        threshold=0.6,
        language="english"
    ),
    language="english"
)
```

### Portuguese

```python
validator = Validator(language="portuguese")

result = validator.validate(
    response="A LGPD foi criada em 2015.",
    context="A LGPD foi sancionada em 14 de agosto de 2018."
)
```

## How it works

Three components run in sequence:

**1. Claim extraction.** The response is split into atomic sentences using NLTK. Each sentence is treated as an independent, verifiable assertion. An `LLMExtractor` is available for higher-precision decomposition when a language model client is available.

**2. NLI classification.** For each claim, the context is split into sentences and the claim is compared against each one using a fine-tuned DeBERTa NLI model (`cross-encoder/nli-deberta-v3-base`). All (sentence, claim) pairs across all claims are submitted in a single batched model call. The sentence with the highest contradiction score is used as the `evidence` field — if any sentence contradicts the claim, the result is `CONTRADICTION` regardless of other scores.

**3. Weighted aggregation.** Scores are combined using label-weighted averaging. Contradiction labels carry full weight (1.0), neutral labels carry partial weight (0.5), and entailment labels carry no weight (0.0). The aggregated score is compared against a configurable threshold to produce the final `is_hallucination` decision.

```
response
    │
    ▼
[ RuleExtractor / LLMExtractor ]   →   ["claim 1", "claim 2", ...]
    │
    ▼ (batched)
[ DeBERTaClassifier            ]   →   [ClaimResult, ClaimResult, ...]
    │
    ▼
[ Aggregator                   ]   →   ValidationResult
```

## Custom components

Every component implements an abstract contract. Custom implementations are supported without modifying any other part of the pipeline:

```python
from cavaquinho.extractor.base import ExtractorContract
from cavaquinho.classifier.base import ClassifierContract
from cavaquinho.models import ClaimResult

class MyExtractor(ExtractorContract):
    def extract(self, response: str, context: str, prompt: str | None = None) -> list[str]:
        ...

class MyClassifier(ClassifierContract):
    def classify_batch(self, claims: list[str], context: str) -> list[ClaimResult]:
        ...

validator = Validator(extractor=MyExtractor(), classifier=MyClassifier())
```

## Output schema

```python
@dataclass(frozen=True)
class ClaimResult:
    text: str            # extracted claim
    evidence: str        # context sentence used in comparison
    label: Labels        # VALUE_ENTAILMENT | VALUE_NEUTRAL | VALUE_CONTRADICTION
    score: float         # model confidence 0.0–1.0, rounded to 4 decimal places
    reason: str | None   # contradicting evidence sentence when label is VALUE_CONTRADICTION

@dataclass(frozen=True)
class ValidationResult:
    score: float                    # weighted aggregate score
    is_hallucination: bool          # score > threshold
    claims: tuple[ClaimResult, ...] # per-claim breakdown (immutable)
    summary: str                    # natural language description of the result
```

## Benchmarks

### Faithfulness — HaluEval QA (English)

Evaluated on [HaluEval QA](https://github.com/RUCAIBox/HaluEval) (Zhang et al., 2023).
Each sample produces two response/context pairs: one hallucinated, one faithful.
Full pipeline: `RuleExtractor → DeBERTaClassifier → Aggregator` (threshold 0.5).

> **Results pending first run.** Run the benchmark locally and the table below will be populated.

| Model | Accuracy | Precision | Recall | F1 | FNR |
|-------|----------|-----------|--------|----|-----|
| `cross-encoder/nli-deberta-v3-base` *(default)* | — | — | — | — | — |

Reproduce with: `python -m benchmarks.faithfulness_benchmark --n 500 --save results/faithfulness.json`

See [`docs/evaluation.md`](docs/evaluation.md) for methodology, metric definitions, and known limitations.

### NLI Component — ASSIN2 Portuguese

Evaluated on the ASSIN2 Brazilian Portuguese validation split (500 sentence pairs, balanced). Binary task: entailment detection vs. non-entailment.

| Model | Accuracy | F1-entailment | F1-none | ms/sample |
|-------|----------|---------------|---------|-----------|
| Majority baseline | 0.500 | 0.667 | 0.000 | — |
| `cross-encoder/nli-deberta-v3-base` *(default)* | **0.882** | **0.885** | **0.879** | 29.5 |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 0.876 | 0.884 | 0.866 | 28.9 |

Reproduce with: `python -m benchmarks.nli_benchmark --n 500`

## Limitations

- **Faithfulness scope.** Cavaquinho verifies whether the response contradicts the provided context. It does not verify factual accuracy against external knowledge — that requires a separate retrieval or knowledge-base step.
- **Context dependency.** Detection quality is proportional to context quality. Incomplete or irrelevant retrieved documents will reduce accuracy.
- **False positives on ambiguous contexts.** When the context contains multiple facts about overlapping subjects, the NLI model may classify semantically consistent claims as contradictions.
- **Python 3.10+.** Tested on Python 3.11 and 3.12. Python 3.14 is functional but produces deprecation warnings from PyTorch.

## Roadmap

### v0.2 — Foundation fixes ✅
See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

### v0.3 — Confidence
- Self-consistency detection via response sampling
- Consistency scoring across multiple sampled outputs

### v0.4 — Factual
- Optional external search integration (Tavily, Wikipedia API)
- Factual verification without a pre-supplied context

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

## Named after Caco, the cat

<p align="center">
  <img src="docs/caco.jpg" alt="Caco, chief hallucination auditor" width="420">
  <br>
  <sub>Caco — chief hallucination auditor</sub>
</p>

## References

- HuDEx: [arXiv 2502.08109](https://arxiv.org/abs/2502.08109)
- DeBERTa NLI: [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
- FaithDial: Dziri et al., 2022
- HaluEval: Li et al., 2023

## License

MIT
