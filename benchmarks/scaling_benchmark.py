"""
Scaling benchmark: latency vs. context size for cavaquinho.

Measures wall-clock time of validator.validate() as context grows
(number of sentences and total tokens), then plots the results.

Accepts custom response and context via --response and --context-file,
or generates a synthetic context of the requested size automatically.

Usage:
    python benchmarks/scaling_benchmark.py
    python benchmarks/scaling_benchmark.py --max-sentences 40 --runs 3
    python benchmarks/scaling_benchmark.py --response "My claim." --context-file ctx.txt
"""

from __future__ import annotations

import argparse
import statistics
import time

import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

from cavaquinho import Validator


def _generate_context(n_sentences: int) -> str:
    """Return a filler context of exactly *n_sentences* distinct sentences."""
    templates = [
        "This is sentence number {i} providing background context for the evaluation.",
        "Context sentence {i} adds additional detail about the subject under discussion.",
        "The {i}th statement in the context elaborates on a related but distinct aspect.",
        "Supporting detail {i} gives further evidence relevant to the evaluated claim.",
        "Background fact {i} is included to simulate a realistic retrieval context.",
    ]
    sentences = []
    for i in range(1, n_sentences + 1):
        template = templates[(i - 1) % len(templates)]
        sentences.append(template.format(i=i))
    return " ".join(sentences)


def count_tokens_approx(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_scaling_benchmark(
    validator: Validator,
    sentence_counts: list[int],
    response: str,
    runs: int,
) -> list[dict]:
    results = []
    n_claims = len(sent_tokenize(response))

    for n in sentence_counts:
        context = _generate_context(n)
        tokens = count_tokens_approx(context)
        times = []

        for _ in range(runs):
            start = time.perf_counter()
            validator.validate(response=response, context=context)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)

        results.append({
            "n_sentences": n,
            "n_tokens": tokens,
            "n_claims": n_claims,
            "mean_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if runs > 1 else 0.0,
        })

        print(
            f"  ctx={n:>2} sentences ({tokens:>4} tokens) | "
            f"{n_claims} claim(s) | "
            f"mean={results[-1]['mean_ms']:>7.1f}ms  "
            f"min={results[-1]['min_ms']:>7.1f}ms  "
            f"max={results[-1]['max_ms']:>7.1f}ms"
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results_by_label: dict[str, list[dict]],
    output_path: str = "scaling_benchmark.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("cavaquinho — Latency Scaling", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for ax_idx, x_key, x_label in [
        (0, "n_sentences", "Context sentences"),
        (1, "n_tokens", "Context tokens (approx)"),
    ]:
        ax = axes[ax_idx]
        for (label, results), color in zip(results_by_label.items(), colors):
            xs = [r[x_key] for r in results]
            means = [r["mean_ms"] for r in results]
            stdevs = [r["stdev_ms"] for r in results]

            ax.plot(xs, means, marker="o", label=label, color=color, linewidth=2)
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stdevs)],
                [m + s for m, s in zip(means, stdevs)],
                alpha=0.15,
                color=color,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_responses(n_claims_list: list[int]) -> dict[str, str]:
    """Build synthetic responses with the requested number of claims each."""
    base = "This is claim number {i} asserting a verifiable fact about the topic."
    responses = {}
    for n in n_claims_list:
        text = " ".join(base.format(i=i) for i in range(1, n + 1))
        responses[f"{n} claim{'s' if n != 1 else ''}"] = text
    return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cavaquinho latency scaling benchmark")
    parser.add_argument(
        "--max-sentences", type=int, default=30,
        help="Maximum number of context sentences to test (default: 30)"
    )
    parser.add_argument(
        "--steps", type=int, default=8,
        help="Number of data points between 1 and max-sentences (default: 8)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per data point (default: 3)"
    )
    parser.add_argument(
        "--claims", nargs="+", type=int, default=[1, 3, 5],
        help="Number of claims per response to test (default: 1 3 5)"
    )
    parser.add_argument(
        "--response", type=str, default=None,
        help="Custom response text (overrides --claims)"
    )
    parser.add_argument(
        "--context-file", type=str, default=None,
        help="Path to a plain-text file whose sentences are used as the context pool"
    )
    parser.add_argument(
        "--output", type=str, default="scaling_benchmark.png",
        help="Output path for the plot image (default: scaling_benchmark.png)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_sentences = args.max_sentences
    step = max(1, max_sentences // args.steps)
    sentence_counts = list(range(1, max_sentences + 1, step))
    if sentence_counts[-1] != max_sentences:
        sentence_counts.append(max_sentences)

    if args.context_file:
        raw = open(args.context_file).read()
        pool = sent_tokenize(raw)
        if len(pool) < max_sentences:
            print(
                f"Warning: context file has only {len(pool)} sentences; "
                f"--max-sentences capped at {len(pool)}."
            )
            max_sentences = len(pool)
            sentence_counts = [s for s in sentence_counts if s <= max_sentences]

        original_generate = globals()["_generate_context"]

        def _generate_context_from_file(n: int) -> str:
            return " ".join(pool[:n])

        globals()["_generate_context"] = _generate_context_from_file

    if args.response:
        responses = {"custom": args.response}
    else:
        responses = _build_responses(args.claims)

    print("Loading model...")
    validator = Validator()
    validator.preload()
    print("Model ready.\n")

    results_by_label: dict[str, list[dict]] = {}

    for label, response in responses.items():
        print(f"--- {label} ---")
        results_by_label[label] = run_scaling_benchmark(
            validator=validator,
            sentence_counts=sentence_counts,
            response=response,
            runs=args.runs,
        )
        print()

    plot_results(results_by_label, output_path=args.output)


if __name__ == "__main__":
    main()
