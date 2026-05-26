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
import cavaquinho as caco

validator = caco.caco()

result = validator.validate(
    response="The LGPD was created in 2015 during Dilma Rousseff's government.",
    context="The LGPD was enacted on August 14, 2018, by President Michel Temer."
)

print(result.score)             # 1.0
print(result.is_hallucination)  # True
print(result.summary)
# 1 of 1 claim(s) contradict the provided context.
# Main conflict: 'The LGPD was created in 2015...'
# contradicting evidence: 'The LGPD was enacted on August 14, 2018...'. Score: 1.00.
```

## Claim-level inspection

Each claim in the response is verified independently. The result exposes the full trace:

```python
for claim in result.claims:
    print(claim.text)      # "The LGPD was created in 2015..."
    print(claim.label)     # Labels.VALUE_CONTRADICTION
    print(claim.score)     # 1.0
    print(claim.evidence)  # "The LGPD was enacted on August 14, 2018..."
    print(claim.reason)    # "The LGPD was enacted on August 14, 2018..."
```

## Using the result

```python
result = validator.validate(response=response, context=context)

# threshold-based decision
if result.is_hallucination:
    response = retry_generation()

# per-claim decision
for claim in result.claims:
    if claim.label == caco.Labels.VALUE_CONTRADICTION and claim.score > 0.8:
        logger.warning(f"Conflicting claim: {claim.text}")
        logger.warning(f"Context evidence: {claim.evidence}")

# score-based soft warning
if result.score > 0.3:
    ui.show_disclaimer("This response may contain inaccurate information.")
```

## RAG integration

The context parameter accepts the documents your retriever already returned. No additional steps are required.

### LangChain

```python
import cavaquinho as caco

source_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in source_docs])
response = llm.invoke(query)

validator = caco.caco()
result = validator.validate(response=response, context=context)
```

### LlamaIndex

```python
import cavaquinho as caco

response = query_engine.query("What is the LGPD?")
context = "\n".join([node.text for node in response.source_nodes])

validator = caco.caco()
result = validator.validate(response=str(response), context=context)
```

### Direct LLM call

```python
import cavaquinho as caco
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

validator = caco.caco()
result = validator.validate(response=response, context=context)
```

## Configuration

All components have sensible defaults. Each can be replaced independently.

```python
from cavaquinho import caco
from cavaquinho.extractor import LLMExtractor
from cavaquinho.classifier import DeBERTaClassifier
from cavaquinho.aggregator import Aggregator
from openai import OpenAI

validator = caco(
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
validator = caco.caco(language="portuguese")

result = validator.validate(
    response="A LGPD foi criada em 2015.",
    context="A LGPD foi sancionada em 14 de agosto de 2018."
)
```

## How it works

Three components run in sequence:

**1. Claim extraction.** The response is split into atomic sentences using NLTK. Each sentence is treated as an independent, verifiable assertion. An `LLMExtractor` is available for higher-precision decomposition when a language model client is available.

**2. NLI classification.** For each claim, the context is split into sentences and the claim is compared against each one using a fine-tuned DeBERTa NLI model (`cross-encoder/nli-deberta-v3-base`). The sentence with the highest contradiction score is used as the `evidence` field. All (sentence, claim) pairs across all claims are batched into a single model call via `classify_batch()`, maximising GPU/MPS throughput.

**3. Weighted aggregation.** Scores are combined using label-weighted averaging. Contradiction labels carry full weight (1.0), neutral labels carry partial weight (0.5), and entailment labels carry no weight (0.0). The aggregated score is compared against a configurable threshold to produce the final `is_hallucination` decision.

```
response
    │
    ▼
[ RuleExtractor / LLMExtractor ]   →   ["claim 1", "claim 2", ...]
    │
    ▼ (concurrent)
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
    def classify(self, claim: str, context: str) -> ClaimResult:
        ...

validator = caco.caco(extractor=MyExtractor(), classifier=MyClassifier())
```

## Output schema

```python
@dataclass(frozen=True)
class ClaimResult:
    text: str            # extracted claim
    evidence: str        # context sentence used in comparison
    label: Labels        # VALUE_ENTAILMENT | VALUE_NEUTRAL | VALUE_CONTRADICTION
    score: float         # model confidence 0.0–1.0
    reason: str | None   # contradicting evidence sentence when label is VALUE_CONTRADICTION

@dataclass(frozen=True)
class ValidationResult:
    score: float                    # weighted aggregate score
    is_hallucination: bool          # score > threshold
    claims: tuple[ClaimResult, ...] # per-claim breakdown (immutable)
    summary: str                    # natural language description of the result
    threshold: float                # decision threshold used (default 0.5)
```

## Benchmark — ASSIN2 Portuguese NLI

Evaluated on the ASSIN2 Brazilian Portuguese validation split (500 sentence pairs, balanced). Binary task: entailment detection vs. non-entailment.

| Model | Accuracy | F1-entailment | F1-none | ms/sample |
|-------|----------|---------------|---------|-----------|
| Majority baseline | 0.500 | 0.667 | 0.000 | — |
| `cross-encoder/nli-deberta-v3-base` *(default)* | **0.882** | **0.885** | **0.879** | 29.5 |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 0.876 | 0.884 | 0.866 | 28.9 |

Reproduce with: `python -m benchmarks.nli_benchmark --n 500`

## Benchmark — HaluEval Hallucination Detection

Evaluated on [HaluEval](https://github.com/RUCKBReasoning/HaluEval) (Li et al., 2023), the standard hallucination detection benchmark.
Binary task: given a knowledge snippet (context) and an answer/summary, detect whether it is hallucinated (`yes`) or faithful (`no`).

> Results below use `cross-encoder/nli-deberta-v3-base`, n=1000, seed=42. QA results pass the question as `query=` to enable short-claim expansion.

| Subset | Threshold | neutral weight | Accuracy | F1-hal | F1-faith | Macro F1 |
|--------|-----------|----------------|----------|--------|----------|----------|
| QA | 0.3 | 0.0 | **0.647** | 0.571 | **0.700** | **0.636** |
| QA | 0.3 | 0.5 *(default)* | 0.551 | 0.660 | 0.341 | 0.500 |
| Majority baseline | — | — | 0.512 | 0.677 | 0.000 | 0.338 |
| Summarization | 0.7 | 0.5 | ~0.45¹ | ~0.41 | ~0.49 | ~0.45 |

> ¹ Summarization results from n=500 calibration sweep. Reproduce with:

```bash
python -m benchmarks.halueval_benchmark --subset qa --n 1000 --threshold 0.3 --workers 1
python -m benchmarks.halueval_benchmark --subset summarization --n 1000 --threshold 0.7 --workers 1
```

**Interpretation:**
- **F1-hal** — how well cavaquinho catches hallucinated responses (recall matters most)
- **F1-faith** — how well cavaquinho avoids false positives on faithful responses
- **Macro F1** — balanced average; most informative metric when the majority baseline dominates F1-hal

With `neutral_weight=0.0` and `query` expansion: accuracy +13.5pp, macro F1 +29.8pp, and precision on hallucinations rises from 0.54 to 0.76 — meaning alerts are far more reliable. The trade-off is lower recall (0.46 vs 0.85), so choose based on whether missing a hallucination or issuing a false alarm is more costly.

### Calibration curve — QA subset (n=1000, neutral weight=0.5)

Sweep of thresholds from 0.1 to 0.9. Use `--calibrate` to reproduce.

| Threshold | Precision | Recall | F1-hal | F1-faith |
|-----------|-----------|--------|--------|----------|
| 0.1 | 0.543 | 0.850 | 0.663 | 0.355 |
| 0.2 | 0.543 | 0.850 | 0.663 | 0.355 |
| 0.3 | 0.544 | 0.844 | **0.662** | 0.363 |
| 0.4 | 0.547 | 0.836 | 0.662 | 0.380 |
| 0.5 *(default)* | 0.614 | 0.467 | 0.531 | 0.615 |
| 0.6 | 0.620 | 0.465 | 0.531 | 0.620 |
| 0.7 | 0.623 | 0.459 | 0.529 | 0.623 |
| 0.8 | 0.621 | 0.451 | 0.523 | 0.622 |
| 0.9 | 0.628 | 0.445 | 0.521 | **0.628** |

Key finding: thresholds 0.1–0.4 maximise hallucination recall (0.84–0.85) at the cost of more false positives. The default 0.5 favours precision (+14pp) but misses ~53% of real hallucinations. Choose based on your application's cost of a missed hallucination vs. a false alarm.

### Calibration curve — Summarization subset (n=500)

| Threshold | Precision | Recall | F1-hal | F1-faith |
|-----------|-----------|--------|--------|----------|
| 0.1–0.4 | 0.506 | 1.000 | 0.672 | 0.000 |
| 0.5 | 0.491 | 0.715 | 0.582 | 0.312 |
| 0.6 | 0.478 | 0.652 | 0.552 | 0.333 |
| 0.7 *(recommended)* | 0.452 | 0.372 | **0.408** | **0.494** |
| 0.8 | 0.449 | 0.190 | 0.267 | 0.588 |
| 0.9 | 0.468 | 0.087 | 0.147 | 0.634 |

Key finding: for summarization, thresholds ≤ 0.4 collapse to majority-class prediction (all hallucinated). The NLI model tends to find spurious contradictions in long documents, inflating the aggregated score. Threshold 0.7 gives the best macro F1 (~0.45). This is a known limitation of sentence-level NLI applied to long documents — see [Limitations](#limitations).

## Limitations

- **Faithfulness scope.** Cavaquinho verifies whether the response contradicts the provided context. It does not verify factual accuracy against external knowledge — that requires a separate retrieval or knowledge-base step.
- **Context dependency.** Detection quality is proportional to context quality. Incomplete or irrelevant retrieved documents will reduce accuracy.
- **False positives on ambiguous contexts.** When the context contains multiple facts about overlapping subjects, the NLI model may classify semantically consistent claims as contradictions.
- **Long documents inflate scores.** For summarization tasks, sentence-level NLI over long source documents tends to find spurious contradictions, pushing the aggregated score up. Use a higher threshold (`THRESHOLDS["summarization"] = 0.7`) or truncate the context to the most relevant passages before calling `validate()`.
- **Python 3.10+.** Tested on Python 3.11 and 3.12. Python 3.14 is functional but produces deprecation warnings from PyTorch.

## Roadmap

### v0.2 — Foundation ✅
Pipeline estável, 65 testes unitários, 92.8% cobertura, CI/CD, benchmark ASSIN2-PT, publicação no PyPI.

### v0.3 — Calibration ✅
- `THRESHOLDS` dict por task type (`qa=0.3`, `summarization=0.4`, `default=0.5`)
- `query` param em `validate()` — expande claims curtos (<5 tokens) com contexto da pergunta
- `ValidationResult.threshold` exposto no resultado
- HaluEval benchmark integrado (`--subset`, `--n`, `--calibrate`)
- Batching NLI: todos os pares (sentença × claim) em uma única chamada ao modelo

### v0.4 — Factual
- Verificação factual com busca externa (Tavily, Wikipedia API)
- Detecção sem contexto pré-fornecido

### v0.5 — Self-consistency
- Detecção via response sampling
- Score de consistência entre múltiplas saídas amostradas

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
