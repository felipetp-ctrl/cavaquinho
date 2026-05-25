<p align="center">
  <img src="docs/header.png" alt="cavaquinho — faithfulness hallucination detection for LLM responses" width="100%">
</p>

<p align="center">
  <a href="https://github.com/felipetp-ctrl/cavaquinho/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/cavaquinho" alt="License"></a>
  <a href="https://pypi.org/project/cavaquinho/"><img src="https://img.shields.io/pypi/v/cavaquinho?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://arxiv.org/abs/2502.08109"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="HuDEx paper"></a>
</p>

---

Cavaquinho is a Python library for faithfulness hallucination detection in LLM responses. It decomposes a response into atomic claims, verifies each one against a provided context using Natural Language Inference, and returns a structured result with per-claim labels, scores, and evidence — not a single opaque verdict.

Inspired by [HuDEx: Integrating Hallucination Detection and Explainability](https://arxiv.org/abs/2502.08109).

Named after Caco, the cat.

## Installation

```bash
pip install cavaquinho
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
# contradicts evidence: 'The LGPD was enacted on August 14, 2018...'. Score: 1.00.
```

## Claim-level inspection

Each claim in the response is verified independently. The result exposes the full trace:

```python
for claim in result.claims:
    print(claim.text)      # "The LGPD was created in 2015..."
    print(claim.label)     # Labels.VALUE_CONTRADICTION
    print(claim.score)     # 1.0
    print(claim.evidence)  # "The LGPD was enacted on August 14, 2018..."
    print(claim.reason)    # "contradiction"
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

**2. NLI classification.** For each claim, the context is split into sentences and the claim is compared against each one using a fine-tuned DeBERTa NLI model (`cross-encoder/nli-deberta-v3-base`). The sentence with the highest contradiction score is used as the `evidence` field. Classification runs concurrently across all claims via `ThreadPoolExecutor`.

**3. Weighted aggregation.** Scores are combined using label-weighted averaging. Contradiction labels carry full weight (1.0), neutral labels carry partial weight (0.5), and entailment labels carry no weight (0.0). The aggregated score is compared against a configurable threshold to produce the final `is_hallucination` decision.

```
response
    │
    ▼
[ ClaimExtractor ]   →   ["claim 1", "claim 2", ...]
    │
    ▼ (concurrent)
[ NLIClassifier  ]   →   [ClaimResult, ClaimResult, ...]
    │
    ▼
[ Aggregator     ]   →   ValidationResult
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
    reason: str | None   # populated when label is VALUE_CONTRADICTION

@dataclass(frozen=True)
class ValidationResult:
    score: float              # weighted aggregate score
    is_hallucination: bool    # score > threshold
    claims: list[ClaimResult] # per-claim breakdown
    summary: str              # natural language description of the result
```

## Limitations

- **Faithfulness scope.** Cavaquinho verifies whether the response contradicts the provided context. It does not verify factual accuracy against external knowledge — that requires a separate retrieval or knowledge-base step.
- **Context dependency.** Detection quality is proportional to context quality. Incomplete or irrelevant retrieved documents will reduce accuracy.
- **False positives on ambiguous contexts.** When the context contains multiple facts about overlapping subjects (e.g., two different dates for the same entity), the NLI model may classify semantically consistent claims as contradictions.
- **Python 3.10+.** Tested on Python 3.11 and 3.12. Python 3.14 is functional but produces deprecation warnings from PyTorch (`torch.jit.script` not supported in 3.14+).

## Roadmap

### v0.1 — Faithfulness ✅
- Claim extraction via NLTK (no external dependencies)
- NLI classification via DeBERTa, runs fully local
- Label-weighted score aggregation
- English and Portuguese support
- Pluggable extractor and classifier interfaces
- LLMExtractor for LLM-based claim decomposition

### v0.2 — Confidence
- Self-consistency detection via response sampling
- Consistency scoring across multiple sampled outputs

### v0.3 — Factual
- Optional external search integration (Tavily, Wikipedia API)
- Factual verification without a pre-supplied context

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

## References

- HuDEx: [arXiv 2502.08109](https://arxiv.org/abs/2502.08109)
- DeBERTa NLI: [`cross-encoder/nli-deberta-v3-base`](https://huggingface.co/cross-encoder/nli-deberta-v3-base)
- FaithDial: Dziri et al., 2022
- HaluEval: Li et al., 2023

## License

MIT