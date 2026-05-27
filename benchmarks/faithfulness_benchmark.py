"""
Faithfulness benchmark — evaluates the full cavaquinho pipeline on HaluEval QA.

HaluEval (Zhang et al., 2023) provides LLM-generated QA responses labelled as
hallucinated (yes) or faithful (no) against a gold passage.  We use the "qa"
split (10 000 samples) and evaluate at the response level: does Validator flag
the response as a hallucination when the gold label is "yes"?

Metrics reported:
  - Accuracy
  - Precision / Recall / F1 for the "hallucination" class
  - False-negative rate (missed hallucinations — the safety-critical metric)
  - Mean score for true-hallucination vs true-faithful samples

Usage:
    # Run with defaults (500 samples, DeBERTa classifier)
    python -m benchmarks.faithfulness_benchmark

    # Run with more samples
    python -m benchmarks.faithfulness_benchmark --n 1000

    # Save results to JSON for README integration
    python -m benchmarks.faithfulness_benchmark --save results/faithfulness.json

Requirements:
    pip install "cavaquinho[nli]" datasets scikit-learn
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from cavaquinho import Validator
from cavaquinho.classifier.deberta import DeBERTaClassifier


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_halueval_qa(n: int) -> list[dict]:
    """Load up to *n* samples from the HaluEval QA split."""
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "knowledge": row["knowledge"],
            "question": row["question"],
            "hallucinated_answer": row["hallucinated_answer"],
            "right_answer": row["right_answer"],
        }
        for row in ds
    ]


def build_pairs(samples: list[dict]) -> tuple[list[dict], list[int]]:
    """
    For each sample produce two evaluation pairs:
      - (hallucinated_answer, knowledge) → gold label 1 (hallucination)
      - (right_answer, knowledge)        → gold label 0 (faithful)

    Returns (pairs, gold_labels).
    """
    pairs: list[dict] = []
    gold: list[int] = []
    for s in samples:
        pairs.append({"response": s["hallucinated_answer"], "context": s["knowledge"]})
        gold.append(1)
        pairs.append({"response": s["right_answer"], "context": s["knowledge"]})
        gold.append(0)
    return pairs, gold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model_name: str
    n_samples: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    mean_score_hallucination: float
    mean_score_faithful: float
    total_latency_s: float
    predictions: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


def run_benchmark(
    validator: Validator,
    pairs: list[dict],
    gold: list[int],
    model_name: str,
) -> BenchmarkResult:
    predictions: list[int] = []
    scores: list[float] = []

    t0 = time.perf_counter()
    for pair in pairs:
        try:
            result = validator.validate(
                response=pair["response"],
                context=pair["context"],
            )
            predictions.append(1 if result.is_hallucination else 0)
            scores.append(result.score)
        except ValueError:
            # Empty response or context — treat as faithful (conservative)
            predictions.append(0)
            scores.append(0.0)
    elapsed = time.perf_counter() - t0

    hal_scores = [s for s, g in zip(scores, gold) if g == 1]
    faithful_scores = [s for s, g in zip(scores, gold) if g == 0]

    tp = sum(1 for p, g in zip(predictions, gold) if p == 1 and g == 1)
    fn = sum(1 for p, g in zip(predictions, gold) if p == 0 and g == 1)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        n_samples=len(pairs),
        threshold=validator.threshold,
        accuracy=accuracy_score(gold, predictions),
        precision=precision_score(gold, predictions, zero_division=0),
        recall=recall_score(gold, predictions, zero_division=0),
        f1=f1_score(gold, predictions, zero_division=0),
        false_negative_rate=fnr,
        mean_score_hallucination=sum(hal_scores) / len(hal_scores) if hal_scores else 0.0,
        mean_score_faithful=sum(faithful_scores) / len(faithful_scores) if faithful_scores else 0.0,
        total_latency_s=elapsed,
        predictions=predictions,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(r: BenchmarkResult, gold: list[int]) -> None:
    print(f"\n{'━' * 70}")
    print(f"Model    : {r.model_name}")
    print(f"Samples  : {r.n_samples} ({r.n_samples // 2} HaluEval QA pairs)")
    print(f"Threshold: {r.threshold}")
    print(f"{'━' * 70}")
    print(classification_report(
        gold,
        r.predictions,
        target_names=["faithful", "hallucination"],
        zero_division=0,
    ))
    cm = confusion_matrix(gold, r.predictions)
    print("  Confusion matrix (rows=gold, cols=predicted):")
    print(f"  {'':16} {'faithful':>10} {'hallucination':>14}")
    labels = ["faithful", "hallucination"]
    for label, row in zip(labels, cm):
        print(f"  gold={label:<12} {row[0]:>10}   {row[1]:>12}")
    print(f"\n  False-negative rate (missed hallucinations): {r.false_negative_rate:.3f}")
    print(f"  Mean score — hallucination samples: {r.mean_score_hallucination:.3f}")
    print(f"  Mean score — faithful samples     : {r.mean_score_faithful:.3f}")
    print(f"  Total inference time              : {r.total_latency_s:.1f} s")
    print(f"  Mean latency per pair             : {r.total_latency_s / (r.n_samples or 1) * 1000:.1f} ms")


def save_results(results: list[BenchmarkResult], path: str) -> None:
    out = []
    for r in results:
        d = asdict(r)
        del d["predictions"]  # large list — not useful in summary JSON
        del d["scores"]
        out.append(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithfulness benchmark on HaluEval QA using the cavaquinho pipeline"
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of HaluEval QA samples (each yields 2 eval pairs; default: 500)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Hallucination threshold passed to Validator (default: 0.5)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If provided, save JSON summary to this path"
    )
    args = parser.parse_args()

    print(f"\nDataset  : HaluEval QA (Zhang et al., 2023)")
    print(f"Samples  : {args.n} QA pairs → {args.n * 2} response evaluations")
    print(f"Task     : Binary faithfulness — does cavaquinho flag the hallucinated answer?")
    print(f"Threshold: {args.threshold}\n")

    print("Loading dataset…")
    samples = load_halueval_qa(args.n)
    pairs, gold = build_pairs(samples)

    print("Loading Validator (DeBERTa NLI)…")
    classifier = DeBERTaClassifier()
    validator = Validator(classifier=classifier, threshold=args.threshold)

    result = run_benchmark(validator, pairs, gold, model_name="cross-encoder/nli-deberta-v3-base")
    print_report(result, gold)

    print(f"\n{'━' * 70}")
    print("SUMMARY")
    print(f"{'━' * 70}")
    print(f"{'Metric':<40} {'Value':>10}")
    print("-" * 55)
    print(f"{'Accuracy':<40} {result.accuracy:>10.3f}")
    print(f"{'Precision (hallucination)':<40} {result.precision:>10.3f}")
    print(f"{'Recall (hallucination)':<40} {result.recall:>10.3f}")
    print(f"{'F1 (hallucination)':<40} {result.f1:>10.3f}")
    print(f"{'False-negative rate (missed)':<40} {result.false_negative_rate:>10.3f}")
    print(f"{'Mean score — hallucinations':<40} {result.mean_score_hallucination:>10.3f}")
    print(f"{'Mean score — faithful':<40} {result.mean_score_faithful:>10.3f}")
    print(f"{'━' * 70}")

    print("\nNote: recall (hallucination) = sensitivity. The false-negative rate")
    print("is the safety-critical metric — missed hallucinations that reach users.")

    if args.save:
        save_results([result], args.save)


if __name__ == "__main__":
    main()
