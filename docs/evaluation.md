# Evaluation Methodology

## Benchmark: HaluEval QA

### Dataset

[HaluEval](https://github.com/RUCAIBox/HaluEval) (Zhang et al., 2023) contains
LLM-generated responses labelled as hallucinated (`yes`) or faithful (`no`) against
a gold knowledge passage.  We use the `qa_samples` split (10 000 samples).

Each sample produces two evaluation pairs:

| Response type       | Gold label   |
|---------------------|--------------|
| `hallucinated_answer` | hallucination |
| `right_answer`        | faithful      |

This yields a balanced binary classification problem at the response level.

### Pipeline

The full `cavaquinho` pipeline is evaluated end-to-end:

```
RuleExtractor → DeBERTaClassifier (cross-encoder/nli-deberta-v3-base) → Aggregator
```

Default threshold: **0.5**.

### Metrics

| Metric | Why it matters |
|--------|---------------|
| **Accuracy** | Overall correctness |
| **Precision** (hallucination) | How many flagged responses are true hallucinations |
| **Recall** (hallucination) | How many hallucinations were caught — the primary safety metric |
| **F1** (hallucination) | Harmonic mean of precision and recall |
| **False-negative rate** | Fraction of hallucinations silently passed — the key risk metric |
| **Mean score (hallucination)** | Expected score for truly hallucinated responses |
| **Mean score (faithful)** | Expected score for truly faithful responses |

The **false-negative rate** is the most important metric for production use: a missed
hallucination that reaches the user is more harmful than a false alarm.

### Reproducing results

```bash
pip install "cavaquinho[nli]" datasets scikit-learn

# 500 samples (default)
python -m benchmarks.faithfulness_benchmark

# Full dataset
python -m benchmarks.faithfulness_benchmark --n 5000 --save results/faithfulness.json
```

### Known limitations

- **Threshold sensitivity**: results depend on the configured threshold. The default
  0.5 is a conservative starting point; tune for your precision/recall tradeoff.
- **Claim granularity vs. response-level labels**: HaluEval labels whole responses.
  A response with one incorrect sentence surrounded by correct ones may score below
  threshold — see the [scoring semantics](../cavaquinho/aggregator.py) for the
  documented dilution behaviour.
- **Single language**: HaluEval QA is English only. For Portuguese evaluation, see
  the ASSIN2 NLI benchmark at `benchmarks/nli_benchmark.py`.

## References

- Zhang, Y. et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark
  for Large Language Models.* arXiv:2305.11747.
- He, P. et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training
  with Gradient-Disentangled Embedding Sharing.* arXiv:2111.09543.
