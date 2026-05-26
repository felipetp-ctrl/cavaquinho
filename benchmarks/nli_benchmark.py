"""
NLI benchmark — compares classifier models on ASSIN2 (Portuguese NLI).

ASSIN2 is the main Brazilian Portuguese NLI dataset (validation: 500 pairs).
Labels: ENTAILMENT | NONE (no contradiction class — 2-class task).

For our faithfulness use case we evaluate binary performance:
  - model predicts "entailment"  → positive prediction
  - model predicts "neutral" or "contradiction" → negative prediction (non-entailment)

This is the most relevant axis for faithfulness detection:
does the model correctly identify what the context supports vs. what it doesn't?

Usage:
    python -m benchmarks.nli_benchmark [--n N] [--models MODEL ...]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import pipeline

# ASSIN2 int label → canonical string
ASSIN2_INT_TO_STR = {0: "none", 1: "entailment"}

DEFAULT_MODELS: list[str] = [
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_assin2(n: int) -> list[dict]:
    """Return up to *n* samples from ASSIN2 Portuguese validation split."""
    ds = load_dataset("nilc-nlp/assin2", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "gold_binary": ASSIN2_INT_TO_STR[row["entailment_judgment"]],
        }
        for row in ds
    ]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _resolve_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def _to_binary(label: str) -> str:
    """Collapse 3-class NLI output to binary: entailment vs non-entailment."""
    return "entailment" if label.lower().strip() in ("entailment", "entail") else "none"


@dataclass
class ModelResult:
    model_name: str
    predictions_binary: list[str] = field(default_factory=list)
    predictions_raw: list[str] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    @property
    def total_latency_s(self) -> float:
        return sum(self.latency_ms) / 1000


def run_model(model_name: str, samples: list[dict]) -> ModelResult:
    result = ModelResult(model_name=model_name)
    device = _resolve_device()

    print(f"\n  Loading '{model_name}' on device={device!r}…", flush=True)
    nli = pipeline(
        task="text-classification",
        model=model_name,
        device=device,
    )

    for i, sample in enumerate(samples, 1):
        t0 = time.perf_counter()
        out = nli(
            {"text": sample["premise"], "text_pair": sample["hypothesis"]},
            truncation=True,
            max_length=512,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # pipeline returns a dict when input is a dict (not a list)
        raw_label = (out[0]["label"] if isinstance(out, list) else out["label"]).lower().strip()
        result.predictions_raw.append(raw_label)
        result.predictions_binary.append(_to_binary(raw_label))
        result.latency_ms.append(elapsed_ms)

        if i % 100 == 0:
            print(f"    {i}/{len(samples)} done…", flush=True)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

BINARY_LABELS = ["entailment", "none"]


def print_report(result: ModelResult, golds: list[str]) -> None:
    name = result.model_name
    print(f"\n{'=' * 70}")
    print(f"  Model: {name}")
    print(f"{'=' * 70}")

    print("\n  Binary classification report (entailment vs non-entailment):")
    print(classification_report(
        golds, result.predictions_binary, labels=BINARY_LABELS, zero_division=0
    ))

    cm = confusion_matrix(golds, result.predictions_binary, labels=BINARY_LABELS)
    print("  Confusion matrix (rows=gold, cols=predicted):")
    print(f"  {'':18} {'entailment':>12} {'none':>8}")
    for row_label, row in zip(BINARY_LABELS, cm):
        print(f"  gold={row_label:<14} {row[0]:>12}   {row[1]:>6}")

    raw_dist = {}
    for p in result.predictions_raw:
        raw_dist[p] = raw_dist.get(p, 0) + 1
    print(f"\n  Raw model label distribution: {raw_dist}")
    print(f"  Mean latency per sample      : {result.mean_latency_ms:.1f} ms")
    print(f"  Total inference time         : {result.total_latency_s:.1f} s")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NLI benchmark on ASSIN2-PT")
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of ASSIN2 samples to evaluate (max 500, default: 500)"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model IDs to benchmark"
    )
    args = parser.parse_args()

    print(f"\nDataset : ASSIN2 — Brazilian Portuguese NLI (validation split)")
    print(f"Samples : {args.n}")
    print(f"Task    : Binary — entailment detection (2-class; no contradiction label in ASSIN2)")
    print(f"Note    : 3-class model outputs are collapsed: neutral+contradiction → 'none'\n")

    samples = load_assin2(args.n)
    golds = [s["gold_binary"] for s in samples]

    dist = {l: golds.count(l) for l in BINARY_LABELS}
    print(f"Gold distribution: {dist}")
    baseline = max(dist.values()) / len(golds)
    print(f"Majority-class baseline accuracy: {baseline:.3f}")

    results: list[ModelResult] = []
    for model_name in args.models:
        r = run_model(model_name, samples)
        results.append(r)

    for r in results:
        print_report(r, golds)

    print(f"\n{'━' * 70}")
    print("SUMMARY — ASSIN2 Portuguese NLI (binary: entailment vs none)")
    print(f"{'━' * 70}")
    print(f"{'Model':<48} {'Acc':>6} {'F1-ent':>8} {'F1-none':>9} {'ms/sample':>10}")
    print("-" * 70)

    majority_f1_ent = f1_score(golds, ["entailment"] * len(golds), pos_label="entailment", average="binary", zero_division=0)
    majority_f1_none = f1_score(golds, ["entailment"] * len(golds), pos_label="none", average="binary", zero_division=0)
    print(f"{'[majority baseline]':<48} {baseline:>6.3f} {majority_f1_ent:>8.3f} {majority_f1_none:>9.3f} {'—':>10}")

    for r in results:
        acc = accuracy_score(golds, r.predictions_binary)
        f1_ent = f1_score(golds, r.predictions_binary, pos_label="entailment", average="binary", zero_division=0)
        f1_none = f1_score(golds, r.predictions_binary, pos_label="none", average="binary", zero_division=0)
        short = r.model_name.split("/")[-1][:46]
        print(f"{short:<48} {acc:>6.3f} {f1_ent:>8.3f} {f1_none:>9.3f} {r.mean_latency_ms:>10.1f}")

    print(f"{'━' * 70}")
    print("\nInterpretation:")
    print("  F1-ent  = how well the model finds supported claims (recall matters for faithfulness)")
    print("  F1-none = how well the model flags unsupported claims (precision matters for guardrails)")


if __name__ == "__main__":
    main()
