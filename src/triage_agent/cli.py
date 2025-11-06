"""Command line entry point for interacting with the triage agent."""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent

from .agent import TriageAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Citizen-facing triage agent using Gemini via LangChain",
    )
    parser.add_argument(
        "symptoms",
        nargs="?",
        help="Symptom description in Urdu or English. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM (default: 0.2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.symptoms:
        symptoms = args.symptoms
    else:
        print("Enter symptoms (Ctrl-D to finish):", file=sys.stderr)
        symptoms = sys.stdin.read().strip()

    if not symptoms:
        parser.error("Symptoms input is required.")

    agent = TriageAgent(model_name=args.model, temperature=args.temperature)
    assessment = agent.assess(symptoms)

    print(
        dedent(
            f"""
            <analysis>{assessment.reasoning}</analysis>
            <urgency>{assessment.urgency}</urgency>
            <plan>{assessment.plan}</plan>
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
