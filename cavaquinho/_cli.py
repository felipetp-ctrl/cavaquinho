"""Command-line interface for cavaquinho."""

from __future__ import annotations

import json
import sys
import textwrap


def _print_human(result, threshold: float) -> None:
    flag = "Hallucination detected  ✗" if result.is_hallucination else "No hallucination found  ✓"
    print(f"\n{flag}   score={result.score}  threshold={threshold}")
    print("─" * 60)
    for i, claim in enumerate(result.claims, 1):
        label = claim.label.value.upper()
        print(f"\nClaim {i} · {label} · {claim.score}")
        print(f"  {textwrap.shorten(claim.text, 80)}")
        if claim.evidence:
            print(f"  ↳ {textwrap.shorten(claim.evidence, 80)}")
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cavaquinho",
        description="Faithfulness hallucination detector for LLM responses.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")

    sub = parser.add_subparsers(dest="command")
    val = sub.add_parser("validate", help="Validate a response against a context.")

    src = val.add_mutually_exclusive_group(required=True)
    src.add_argument("--response", type=str, help="Response text to validate.")
    src.add_argument("--response-file", type=str, metavar="FILE",
                     help="File containing the response (use '-' for stdin).")

    ctx = val.add_mutually_exclusive_group(required=True)
    ctx.add_argument("--context", type=str, help="Context text.")
    ctx.add_argument("--context-file", type=str, metavar="FILE",
                     help="File containing the context (use '-' for stdin).")

    val.add_argument("--threshold", type=float, default=0.5,
                     help="Hallucination score threshold (default: 0.5).")
    val.add_argument("--language", type=str, default="english",
                     choices=["english", "portuguese"],
                     help="Pipeline language (default: english).")
    val.add_argument("--json", action="store_true", dest="as_json",
                     help="Output result as JSON.")

    args = parser.parse_args()

    if args.version:
        from cavaquinho import __version__
        print(f"cavaquinho {__version__}")
        sys.exit(0)

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    # --- resolve response ---
    if args.response:
        response = args.response
    else:
        f = sys.stdin if args.response_file == "-" else open(args.response_file)
        response = f.read()
        if args.response_file != "-":
            f.close()

    # --- resolve context ---
    if args.context:
        context = args.context
    else:
        f = sys.stdin if args.context_file == "-" else open(args.context_file)
        context = f.read()
        if args.context_file != "-":
            f.close()

    try:
        from cavaquinho import Validator
        validator = Validator(threshold=args.threshold, language=args.language)
        result = validator.validate(response=response, context=context)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.as_json:
        out = {
            "is_hallucination": result.is_hallucination,
            "score": result.score,
            "threshold": args.threshold,
            "summary": result.summary,
            "claims": [
                {
                    "text": c.text,
                    "label": c.label.value,
                    "score": c.score,
                    "evidence": c.evidence,
                    "reason": c.reason,
                }
                for c in result.claims
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        _print_human(result, args.threshold)

    sys.exit(1 if result.is_hallucination else 0)
