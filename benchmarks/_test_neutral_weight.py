"""Quick comparison: neutral=0.5 (default) vs neutral=0.0 on HaluEval QA."""
from __future__ import annotations
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.insert(0, "/Users/felipetomepereira/Projects")

import cavaquinho as caco
from cavaquinho.aggregator import Aggregator
from cavaquinho.models import Labels
from cavaquinho.config import THRESHOLDS

N = 1000
THRESHOLD = THRESHOLDS["qa"]  # 0.3

print(f"Loading HaluEval QA ({N} samples)…")
ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
ds = ds.shuffle(seed=42).select(range(N))
samples = [
    {"context": r["knowledge"], "response": r["answer"], "gold": r["hallucination"], "query": r.get("question")}
    for r in ds
]

configs = [
    ("neutral=0.5 (default)", {Labels.VALUE_CONTRADICTION: 1.0, Labels.VALUE_NEUTRAL: 0.5, Labels.VALUE_ENTAILMENT: 0.0}),
    ("neutral=0.0",           {Labels.VALUE_CONTRADICTION: 1.0, Labels.VALUE_NEUTRAL: 0.0, Labels.VALUE_ENTAILMENT: 0.0}),
]

for label, weights in configs:
    validator = caco.caco(
        aggregator=Aggregator(threshold=THRESHOLD, weights=weights)
    )
    preds, golds, latencies = [], [], []
    print(f"\nRunning [{label}]…")
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        result = validator.validate(response=s["response"], context=s["context"], query=s.get("query"))
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append("yes" if result.is_hallucination else "no")
        golds.append(s["gold"])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}…")

    acc   = accuracy_score(golds, preds)
    f1h   = f1_score(golds, preds, pos_label="yes", average="binary", zero_division=0)
    f1f   = f1_score(golds, preds, pos_label="no",  average="binary", zero_division=0)
    ms    = sum(latencies) / len(latencies)
    yes_p = preds.count("yes")

    print(f"\n  [{label}]")
    print(f"  Predicted 'yes': {yes_p}/{N} ({yes_p/N:.1%})")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  F1-hal   : {f1h:.3f}")
    print(f"  F1-faith : {f1f:.3f}")
    print(f"  ms/sample: {ms:.1f}")
    print()
    print(classification_report(golds, preds, labels=["yes","no"], zero_division=0))
