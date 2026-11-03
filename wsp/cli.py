"""Command line: one subcommand per experiment.

    uv run wsp list
    uv run wsp exp1 --model deepseek-v4.1-flash
    uv run wsp exp1 --model my-local-model --base-url http://localhost:8000/v1

Every experiment takes the same flags, so swapping the model is one argument and
never a code change.
"""

import argparse
import json
import sys
from pathlib import Path

from .experiments import REGISTRY, SPECS

REPO = Path(__file__).resolve().parent.parent


def build_parser():
    p = argparse.ArgumentParser(
        prog="wsp",
        description="Reproduce the weight-space-programming experiments with any model.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every experiment and whether it is implemented")
    sub.add_parser("report", help="summarise every results/<exp>/<tag>/scores.json on disk")

    for name in sorted(SPECS):
        spec = SPECS[name]
        sp = sub.add_parser(
            name, help=spec["title"], description=f"{spec['title']} (tools: {spec['tools']})"
        )
        sp.add_argument(
            "--model",
            default="deepseek-v4.1-flash",
            help="model id (pi registry, or the served name with --base-url)",
        )
        sp.add_argument(
            "--provider", default="opencode-go", help="pi provider; ignored with --base-url"
        )
        sp.add_argument(
            "--base-url", default=None, help="OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter)"
        )
        sp.add_argument("--thinking", default="high", help="pi thinking level for pi-backed runs")
        sp.add_argument("--tag", default="", help="results subdirectory (default: the model id)")
        sp.add_argument("--workers", type=int, default=3)
        sp.add_argument(
            "--attempts",
            type=int,
            default=None,
            help=f"override the post's design (default {spec['attempts']})",
        )
        sp.add_argument(
            "--cases",
            nargs="*",
            default=None,
            help="subset of conditions, e.g. --cases breakout pong",
        )
        sp.add_argument(
            "--rescore",
            action="store_true",
            help="re-grade saved answers instead of calling the model",
        )
        sp.add_argument(
            "--max-runs",
            type=int,
            default=None,
            help="cap the sandbox runs per conversation (default: the post's budget)",
        )
        sp.add_argument(
            "--episodes",
            type=int,
            default=None,
            help="episodes per evaluation (default: the post's 1000)",
        )
    return p


def cmd_list():
    print(f"{'exp':6s} {'attempts':9s} {'tools':28s} status   title")
    for name in sorted(SPECS):
        s = SPECS[name]
        status = "ready" if name in REGISTRY else "planned"
        print(f"{name:6s} {s['attempts']:<9d} {s['tools']:28s} {status:8s} {s['title']}")


def summarise(recs):
    """One line per scores.json, whatever shape that experiment writes."""
    n = len(recs)
    if any("correct" in r for r in recs):
        return f"{sum(bool(r.get('correct')) for r in recs)}/{n} correct"
    if any("corpus_correct" in r for r in recs):
        return (
            f"corpus {sum(bool(r.get('corpus_correct')) for r in recs)}/{n}, "
            f"copy-set exact {sum(bool(r.get('copy_exact')) for r in recs)}/{n}"
        )
    if any("score" in r for r in recs):
        parts = [
            f"{r.get('game', r.get('case', '?'))}={r['score'].get('mean')}"
            for r in recs
            if isinstance(r.get("score"), dict)
        ]
        return ", ".join(parts) if parts else f"{n} records"
    if any(r.get("scored") is False for r in recs):
        return f"{n} record(s), not scorable"
    return f"{n} record(s)"


def cmd_report():
    root = REPO / "results"
    if not root.exists():
        print("no results yet")
        return 0
    rows = []
    for path in sorted(root.glob("*/*/scores.json")):
        try:
            recs = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            rows.append((path.parent.parent.name, path.parent.name, f"unreadable: {e}"))
            continue
        if not isinstance(recs, list):
            recs = [recs]
        rows.append((path.parent.parent.name, path.parent.name, summarise(recs)))
    if not rows:
        print("no scores.json found under results/")
        return 0
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    print(f"{'exp':{w1}s}  {'run':{w2}s}  summary")
    for exp, run, text in rows:
        print(f"{exp:{w1}s}  {run:{w2}s}  {text}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        cmd_list()
        return 0
    if args.cmd == "report":
        return cmd_report()

    module = REGISTRY.get(args.cmd)
    if module is None:
        print(
            f"{args.cmd} is not implemented yet. Buildable now: {', '.join(sorted(REGISTRY))}",
            file=sys.stderr,
        )
        return 2

    args.exp = args.cmd
    recs = module.run(args)
    return 0 if all(r.get("correct") for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
