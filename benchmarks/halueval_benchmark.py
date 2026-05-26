"""
HaluEval benchmark — evaluates cavaquinho against the HaluEval dataset.

HaluEval (Li et al., 2023) contains human-annotated hallucination labels for
QA and summarization tasks. Each sample provides a knowledge snippet (context),
a response, and a binary label (yes = hallucinated, no = faithful).

This benchmark maps directly to cavaquinho's use case: given a context and a
response, does the pipeline correctly flag hallucinated responses?

Subsets evaluated:
  - qa_samples        (10 000 samples, balanced)
  - summarization     (10 000 samples, balanced)

Usage:
    python -m benchmarks.halueval_benchmark [--subset qa|summarization|all]
                                            [--n N]
                                            [--threshold FLOAT]
                                            [--language english]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import cavaquinho as caco


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_qa(n: int) -> list[dict]:
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "context": row["knowledge"],  # type: ignore[index]
            "response": row["answer"],  # type: ignore[index]
            "gold": row["hallucination"],  # type: ignore[index]  # "yes" | "no"
            "query": row.get("question"),  # type: ignore[index]
        }
        for row in ds
    ]


def _load_summarization(n: int) -> list[dict]:
    ds = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "context": row["document"],  # type: ignore[index]
            "response": row["summary"],  # type: ignore[index]
            "gold": row["hallucination"],  # type: ignore[index]
        }
        for row in ds
    ]


LOADERS = {
    "qa": _load_qa,
    "summarization": _load_summarization,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class SubsetResult:
    subset: str
    predictions: list[str] = field(default_factory=list)  # "yes" | "no"
    golds: list[str] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0


def _run_sample(validator: caco.caco, sample: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = validator.validate(
        response=sample["response"],
        context=sample["context"],
        query=sample.get("query"),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    prediction = "yes" if result.is_hallucination else "no"
    return prediction, elapsed_ms


def run_subset(
    subset_name: str,
    samples: list[dict],
    validator: caco.caco,
    max_workers: int = 4,
) -> SubsetResult:
    result = SubsetResult(subset=subset_name)

    print(f"\n  Running '{subset_name}' ({len(samples)} samples, {max_workers} workers)…", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_sample, validator, s): (i, s) for i, s in enumerate(samples)}
        completed = 0
        for future in as_completed(futures):
            i, sample = futures[future]
            try:
                pred, latency = future.result()
                result.predictions.append(pred)
                result.golds.append(sample["gold"])
                result.latency_ms.append(latency)
            except Exception as exc:
                result.errors += 1
                result.predictions.append("no")
                result.golds.append(sample["gold"])
                result.latency_ms.append(0.0)
                print(f"    [ERROR sample {i}] {exc}", flush=True)

            completed += 1
            if completed % 50 == 0:
                print(f"    {completed}/{len(samples)} done…", flush=True)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

LABELS = ["yes", "no"]


def print_report(result: SubsetResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Subset : {result.subset}")
    print(f"{'=' * 70}")

    if result.errors:
        print(f"  Errors (treated as 'no'): {result.errors}")

    print("\n  Classification report (yes = hallucinated):")
    print(classification_report(result.golds, result.predictions, labels=LABELS, zero_division=0))

    cm = confusion_matrix(result.golds, result.predictions, labels=LABELS)
    print("  Confusion matrix (rows=gold, cols=predicted):")
    print(f"  {'':12} {'pred=yes':>10} {'pred=no':>10}")
    for row_label, row in zip(LABELS, cm):
        print(f"  gold={row_label:<8} {row[0]:>10}   {row[1]:>8}")

    dist = Counter(result.golds)
    baseline = max(dist.values()) / len(result.golds)
    print(f"\n  Gold distribution        : {dict(dist)}")
    print(f"  Majority-class baseline  : {baseline:.3f}")
    print(f"  Mean latency per sample  : {result.mean_latency_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------

def _collect_scores(
    samples: list[dict],
    language: str,
    max_workers: int,
) -> tuple[list[float], list[str]]:
    """Return raw cavaquinho scores and gold labels for all samples."""
    # Use threshold=0 so every sample is always classified; we only need scores.
    validator = caco.caco(threshold=0.0, language=language)
    scores: list[float] = []
    golds: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                lambda s: validator.validate(
                    response=s["response"],
                    context=s["context"],
                    query=s.get("query"),
                ).score,
                sample,
            ): sample
            for sample in samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                scores.append(future.result())
            except Exception:
                scores.append(0.0)
            golds.append(sample["gold"])

    return scores, golds


def _run_calibration(
    subset_name: str,
    samples: list[dict],
    max_workers: int,
    language: str,
) -> None:
    print(f"  Collecting scores ({len(samples)} samples)…", flush=True)
    scores, golds = _collect_scores(samples, language, max_workers)

    thresholds = [round(t * 0.1, 1) for t in range(1, 10)]

    print(f"\n{'─' * 55}")
    print(f"  Calibration — subset: {subset_name}")
    print(f"{'─' * 55}")
    print(f"  {'Threshold':>9} {'Precision':>10} {'Recall':>8} {'F1-hal':>8} {'F1-faith':>10}")
    print(f"  {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*10}")

    from sklearn.metrics import precision_score, recall_score

    for t in thresholds:
        preds = ["yes" if s >= t else "no" for s in scores]
        prec = precision_score(golds, preds, pos_label="yes", average="binary", zero_division=0)
        rec = recall_score(golds, preds, pos_label="yes", average="binary", zero_division=0)
        f1_hal = f1_score(golds, preds, pos_label="yes", average="binary", zero_division=0)
        f1_faith = f1_score(golds, preds, pos_label="no", average="binary", zero_division=0)
        print(f"  {t:>9.1f} {prec:>10.3f} {rec:>8.3f} {f1_hal:>8.3f} {f1_faith:>10.3f}")

    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HaluEval benchmark for cavaquinho")
    parser.add_argument(
        "--subset", choices=["qa", "summarization", "all"], default="all",
        help="Which HaluEval subset to evaluate (default: all)"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Samples per subset (max 10000, default: 200)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Hallucination decision threshold (default: 0.5)"
    )
    parser.add_argument(
        "--language", type=str, default="english",
        help="Language for sentence tokenisation (default: english)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for inference (default: 4)"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Sweep thresholds 0.1–0.9 and print precision/recall/F1 table"
    )
    args = parser.parse_args()

    subsets = ["qa", "summarization"] if args.subset == "all" else [args.subset]

    print(f"\nDataset   : HaluEval (Li et al., 2023)")
    print(f"Subsets   : {subsets}")
    print(f"Samples   : {args.n} per subset")
    print(f"Threshold : {args.threshold}")
    print(f"Language  : {args.language}")
    print(f"Workers   : {args.workers}")
    print(f"\nLoading cavaquinho validator…")

    validator = caco.caco(threshold=args.threshold, language=args.language)

    if args.calibrate:
        subset_name = subsets[0]  # calibrate on the first requested subset
        print(f"\nCalibration mode — sweeping thresholds on '{subset_name}' ({args.n} samples)")
        samples = LOADERS[subset_name](args.n)
        # collect raw scores once, then threshold-sweep in memory
        _run_calibration(subset_name, samples, args.workers, args.language)
        return

    results: list[SubsetResult] = []
    for subset_name in subsets:
        samples = LOADERS[subset_name](args.n)
        r = run_subset(subset_name, samples, validator, max_workers=args.workers)
        results.append(r)

    for r in results:
        print_report(r)

    # Summary table
    print(f"\n{'━' * 70}")
    print("SUMMARY — HaluEval (binary: yes=hallucinated, no=faithful)")
    print(f"{'━' * 70}")
    print(f"{'Subset':<20} {'Acc':>6} {'F1-hal':>8} {'F1-faith':>10} {'ms/sample':>10}")
    print("-" * 70)

    for r in results:
        acc = accuracy_score(r.golds, r.predictions)
        f1_hal = f1_score(r.golds, r.predictions, pos_label="yes", average="binary", zero_division=0)
        f1_faith = f1_score(r.golds, r.predictions, pos_label="no", average="binary", zero_division=0)
        print(f"{r.subset:<20} {acc:>6.3f} {f1_hal:>8.3f} {f1_faith:>10.3f} {r.mean_latency_ms:>10.1f}")

    print(f"{'━' * 70}")
    print("\nInterpretation:")
    print("  F1-hal   = how well cavaquinho catches hallucinated responses (recall matters)")
    print("  F1-faith = how well cavaquinho avoids false positives on faithful responses")


if __name__ == "__main__":
    main()
