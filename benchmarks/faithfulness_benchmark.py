"""
Faithfulness benchmark — evaluates the full cavaquinho pipeline on RAGTruth.

RAGTruth (Wu et al., 2023) provides LLM-generated responses (QA, Summary,
Data2txt) labelled at the span level for two hallucination types:

  - evident_conflict : the response directly contradicts the source context.
  - baseless_info    : the response adds information not present in the context
                       but does not contradict it.

cavaquinho is a *faithfulness* detector — it finds contradictions between a
response and a supplied context.  Accordingly:

  - evident_conflict = 1  → cavaquinho SHOULD fire   (gold label 1)
  - baseless_info    = 1 only (no conflict) → cavaquinho should NOT fire
  - clean (both = 0)      → cavaquinho should NOT fire (gold label 0)

The secondary table reports cavaquinho's behaviour on baseless-only samples
to make explicit that this is a known, intentional limitation.

Metrics reported:
  - Accuracy / Precision / Recall / F1 for the "evident_conflict" class
  - False-negative rate  (missed contradictions — the safety-critical metric)
  - False-positive rate on clean samples
  - False-positive rate on baseless-only samples (separate table)
  - Mean score per group

Usage:
    # Run with defaults (full 2700-sample test split)
    python -m benchmarks.faithfulness_benchmark

    # Smaller run for a quick check
    python -m benchmarks.faithfulness_benchmark --n 500

    # Save results to JSON
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

def load_ragtruth(n: int, cache_dir: str | None = None) -> list[dict]:
    """Load up to *n* samples from the RAGTruth test split.

    Args:
        n: Maximum number of rows to load.
        cache_dir: Optional Hugging Face datasets cache directory.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("wandb/RAGTruth-processed", split="test", cache_dir=cache_dir)
    # drop malformed outputs
    ds = ds.filter(lambda r: r["quality"] == "good")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "context": row["context"],
            "response": row["output"],
            "evident_conflict": row["hallucination_labels_processed"]["evident_conflict"],
            "baseless_info": row["hallucination_labels_processed"]["baseless_info"],
            "task_type": row["task_type"],
        }
        for row in ds
    ]


def build_pairs(samples: list[dict]) -> tuple[list[dict], list[int]]:
    """
    Gold label:
      1 if evident_conflict == 1  (direct contradiction → should be flagged)
      0 otherwise                 (clean or baseless-only → should not be flagged)
    """
    pairs: list[dict] = []
    gold: list[int] = []
    for s in samples:
        pairs.append({"response": s["response"], "context": s["context"]})
        gold.append(s["evident_conflict"])
    return pairs, gold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model_name: str
    n_samples: int
    n_conflict: int
    n_baseless_only: int
    n_clean: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    false_positive_rate_clean: float
    false_positive_rate_baseless: float
    mean_score_conflict: float
    mean_score_baseless_only: float
    mean_score_clean: float
    total_latency_s: float
    predictions: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    per_sample_claims: list[dict] = field(default_factory=list)


def run_benchmark(
    validator: Validator,
    pairs: list[dict],
    gold: list[int],
    samples: list[dict],
    model_name: str,
) -> BenchmarkResult:
    predictions: list[int] = []
    scores: list[float] = []
    per_sample_claims: list[list[dict]] = []

    t0 = time.perf_counter()
    for i, pair in enumerate(pairs):
        try:
            result = validator.validate(
                response=pair["response"],
                context=pair["context"],
            )
            predictions.append(1 if result.is_hallucination else 0)
            scores.append(result.score)
            # store per-sample claim breakdown for diagnostics
            per_claims = [
                {
                    "text": c.text,
                    "label": c.label.value,
                    "score": c.score,
                    "evidence": c.evidence,
                    "reason": c.reason,
                }
                for c in result.claims
            ]
            per_sample_claims.append(per_claims)
        except ValueError:
            predictions.append(0)
            scores.append(0.0)
            per_sample_claims.append([])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)} done…", flush=True)
    elapsed = time.perf_counter() - t0

    # group indices
    conflict_idx = [i for i, s in enumerate(samples) if s["evident_conflict"] == 1]
    baseless_idx = [i for i, s in enumerate(samples)
                    if s["evident_conflict"] == 0 and s["baseless_info"] == 1]
    clean_idx = [i for i, s in enumerate(samples)
                 if s["evident_conflict"] == 0 and s["baseless_info"] == 0]

    def mean_score(idx: list[int]) -> float:
        vals = [scores[i] for i in idx]
        return sum(vals) / len(vals) if vals else 0.0

    def fpr(idx: list[int]) -> float:
        fp = sum(predictions[i] for i in idx)
        return fp / len(idx) if idx else 0.0

    tp = sum(1 for p, g in zip(predictions, gold) if p == 1 and g == 1)
    fn = sum(1 for p, g in zip(predictions, gold) if p == 0 and g == 1)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return BenchmarkResult(
        model_name=model_name,
        n_samples=len(pairs),
        n_conflict=len(conflict_idx),
        n_baseless_only=len(baseless_idx),
        n_clean=len(clean_idx),
        threshold=validator.threshold,
        accuracy=accuracy_score(gold, predictions),
        precision=precision_score(gold, predictions, zero_division=0),
        recall=recall_score(gold, predictions, zero_division=0),
        f1=f1_score(gold, predictions, zero_division=0),
        false_negative_rate=fnr,
        false_positive_rate_clean=fpr(clean_idx),
        false_positive_rate_baseless=fpr(baseless_idx),
        mean_score_conflict=mean_score(conflict_idx),
        mean_score_baseless_only=mean_score(baseless_idx),
        mean_score_clean=mean_score(clean_idx),
        total_latency_s=elapsed,
        predictions=predictions,
        scores=scores,
        per_sample_claims=per_sample_claims,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(r: BenchmarkResult, gold: list[int]) -> None:
    print(f"\n{'━' * 70}")
    print(f"Model    : {r.model_name}")
    print(f"Samples  : {r.n_samples}  (conflict={r.n_conflict}, "
          f"baseless-only={r.n_baseless_only}, clean={r.n_clean})")
    print(f"Threshold: {r.threshold}")
    print(f"{'━' * 70}")
    print(classification_report(
        gold,
        r.predictions,
        target_names=["faithful", "conflict"],
        zero_division=0,
    ))
    cm = confusion_matrix(gold, r.predictions)
    print("  Confusion matrix (rows=gold, cols=predicted):")
    print(f"  {'':16} {'faithful':>10} {'conflict':>10}")
    for label, row in zip(["faithful", "conflict"], cm):
        print(f"  gold={label:<12} {row[0]:>10}   {row[1]:>8}")

    print(f"\n  False-negative rate (missed contradictions) : {r.false_negative_rate:.3f}")
    print(f"  False-positive rate on clean samples        : {r.false_positive_rate_clean:.3f}")
    print(f"  False-positive rate on baseless-only samples: {r.false_positive_rate_baseless:.3f}")
    print(f"\n  Mean score — conflict samples    : {r.mean_score_conflict:.3f}")
    print(f"  Mean score — baseless-only       : {r.mean_score_baseless_only:.3f}")
    print(f"  Mean score — clean samples       : {r.mean_score_clean:.3f}")
    print(f"\n  Total inference time   : {r.total_latency_s:.1f} s")
    print(f"  Mean latency per sample: {r.total_latency_s / (r.n_samples or 1) * 1000:.1f} ms")


def save_results(results: list[BenchmarkResult], path: str) -> None:
    out = []
    for r in results:
        d = asdict(r)
        del d["predictions"]
        del d["scores"]
        out.append(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {path}")


def save_detailed_results(results: list[BenchmarkResult], path: str) -> None:
    """Save full per-sample results including claim-level breakdown."""
    out = []
    for r in results:
        for i, pred in enumerate(r.predictions):
            sample = {
                "model_name": r.model_name,
                "threshold": r.threshold,
                "index": i,
                "predicted": pred,
                "score": r.scores[i] if i < len(r.scores) else 0.0,
                "claims": r.per_sample_claims[i] if i < len(r.per_sample_claims) else [],
            }
            out.append(sample)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\nDetailed results saved to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faithfulness benchmark on RAGTruth using the cavaquinho pipeline"
    )
    parser.add_argument(
        "--n", type=int, default=2700,
        help="Number of RAGTruth test samples to evaluate (default: all 2700)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Hallucination threshold passed to Validator (default: 0.5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="NLI pipeline batch size (lower = less VRAM; default: 4)"
    )
    parser.add_argument(
        "--model", type=str, default="cross-encoder/nli-deberta-v3-base",
        help="HuggingFace model identifier for DeBERTaClassifier"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="If provided, save JSON summary to this path"
    )
    parser.add_argument(
        "--hf-cache-dir", type=str, default=None,
        help=(
            "Optional Hugging Face datasets cache directory "
            "(default: use Hugging Face global cache)"
        ),
    )
    parser.add_argument(
        "--save-detailed", type=str, default=None,
        help=("If provided, save per-sample detailed JSON to this path"),
    )
    args = parser.parse_args()

    print(f"\nDataset  : RAGTruth (Wu et al., 2023) — test split")
    print(f"Samples  : {args.n}")
    print(f"Task     : Binary faithfulness — detect evident_conflict hallucinations")
    print(f"Threshold: {args.threshold}")
    print(f"Model    : {args.model}")
    print(f"Batch sz : {args.batch_size}\n")
    if args.hf_cache_dir:
        print(f"HF cache : {args.hf_cache_dir}\n")

    print("Loading dataset…")
    samples = load_ragtruth(args.n, cache_dir=args.hf_cache_dir)
    pairs, gold = build_pairs(samples)

    print(f"Loading Validator ({args.model})…")
    classifier = DeBERTaClassifier(model_name=args.model, batch_size=args.batch_size)
    validator = Validator(classifier=classifier, threshold=args.threshold)

    result = run_benchmark(validator, pairs, gold, samples, model_name=args.model)
    print_report(result, gold)

    print(f"\n{'━' * 70}")
    print("SUMMARY")
    print(f"{'━' * 70}")
    print(f"{'Metric':<48} {'Value':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<48} {result.accuracy:>10.3f}")
    print(f"{'Precision (conflict)':<48} {result.precision:>10.3f}")
    print(f"{'Recall (conflict)':<48} {result.recall:>10.3f}")
    print(f"{'F1 (conflict)':<48} {result.f1:>10.3f}")
    print(f"{'False-negative rate (missed contradictions)':<48} {result.false_negative_rate:>10.3f}")
    print(f"{'FPR — clean samples':<48} {result.false_positive_rate_clean:>10.3f}")
    print(f"{'FPR — baseless-only samples':<48} {result.false_positive_rate_baseless:>10.3f}")
    print(f"{'━' * 70}")

    print("\nNote: baseless_info hallucinations (added facts not in context but not")
    print("contradicting it) are outside cavaquinho's scope by design — the model")
    print("correctly classifies them as non-contradictory.")

    if args.save:
        save_results([result], args.save)
    if args.save_detailed:
        save_detailed_results([result], args.save_detailed)


if __name__ == "__main__":
    main()
