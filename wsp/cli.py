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

from .experiments import EFFORT, REGISTRY, SKIP, SPECS

REPO = Path(__file__).resolve().parent.parent


def build_parser():
    p = argparse.ArgumentParser(
        prog="wsp",
        description="Reproduce the weight-space-programming experiments with any model.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show every experiment and whether it is implemented")
    sub.add_parser("report", help="summarise every results/<exp>/<tag>/scores.json on disk")

    run_all = sub.add_parser("all", help="run every implemented experiment for one model")
    run_all.add_argument("--model", default="deepseek-v4.1-flash")
    run_all.add_argument("--provider", default="opencode-go")
    run_all.add_argument("--base-url", default=None)
    run_all.add_argument(
        "--thinking",
        default=None,
        help="override; by default each experiment uses the effort the post used",
    )
    run_all.add_argument("--tag", default="")
    run_all.add_argument("--workers", type=int, default=3)
    run_all.add_argument("--episodes", type=int, default=None)
    run_all.add_argument("--exps", nargs="*", default=None, help="subset, e.g. --exps exp1 exp2")

    q = sub.add_parser("queue", help="book-keeping queue of (model, experiment) jobs")
    qa = q.add_subparsers(dest="action", required=True)
    qadd = qa.add_parser("add", help="enqueue every experiment for a model")
    qadd.add_argument("--model", required=True)
    qadd.add_argument("--provider", default="opencode-go")
    qadd.add_argument("--exps", nargs="*", default=None)
    qa.add_parser("status", help="show every job and its state")
    qrun = qa.add_parser("run", help="execute everything still pending")
    qrun.add_argument("--workers", type=int, default=2)

    for name in sorted(set(SPECS) - SKIP):
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
        sp.add_argument(
            "--generate",
            action="store_true",
            help="exp6 only: have the model write the weights, then score them",
        )
        sp.add_argument(
            "--attempt",
            type=int,
            default=None,
            help="exp3 only: run just this attempt index (1-based), for parallelism",
        )
        sp.add_argument(
            "--protocol",
            default=None,
            choices=["weights_only", "attention"],
            help="exp7 only: which protocol to run (default: weights_only)",
        )
    return p


def cmd_list():
    print(f"{'exp':6s} {'attempts':9s} {'tools':28s} status   title")
    for name in sorted(set(SPECS) - SKIP):
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
    scores = [r.get("score") for r in recs]
    if any(isinstance(s, dict) and "mean" in s for s in scores):
        return ", ".join(
            f"{r.get('game', r.get('case', '?'))}={s['mean']}"
            for r, s in zip(recs, scores, strict=True)
            if isinstance(s, dict) and "mean" in s
        )
    if any(isinstance(s, dict) and "success" in s for s in scores):
        return ", ".join(
            f"{r.get('case', '?')}: success {s['success']:.1%} ({s.get('episodes', '?')} eps)"
            for r, s in zip(recs, scores, strict=True)
            if isinstance(s, dict) and "success" in s
        )
    if any(isinstance(s, (int, float)) for s in scores):
        return ", ".join(
            f"vs {r.get('ppo', '?')}={s:.3f}"
            for r, s in zip(recs, scores, strict=True)
            if isinstance(s, (int, float))
        )
    if any(r.get("scored") is False for r in recs):
        return f"{n} record(s), not scorable"
    return f"{n} record(s)"


def cmd_report():
    root = REPO / "results"
    if not root.exists():
        print("no results yet")
        return 0
    rows = []
    paths = list(root.glob("*/*/scores.json")) + list(root.glob("exp7/runs/*/*/scores.json"))
    for path in sorted(paths):
        # results/exp1/<tag>/scores.json or results/exp7/runs/<protocol>/<tag>/scores.json
        parts = path.relative_to(root).parts
        exp = parts[0]
        if len(parts) > 4 and parts[1] == "runs":
            run = f"{parts[2]}:{parts[3]}"
        else:
            run = parts[1] if len(parts) > 1 else "?"
        try:
            recs = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            rows.append((exp, run, f"unreadable: {e}"))
            continue
        if not isinstance(recs, list):
            recs = [recs]
        rows.append((exp, run, summarise(recs)))
    if not rows:
        print("no scores.json found under results/")
        return 0
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    print(f"{'exp':{w1}s}  {'run':{w2}s}  summary")
    for exp, run, text in rows:
        print(f"{exp:{w1}s}  {run:{w2}s}  {text}")
    return 0


def cmd_all(args):
    """Run every implemented experiment for one model.

    Effort follows the post per experiment unless explicitly overridden: using one
    global effort silently under-powers exp 3/5/6/7, which changes their results.
    """
    order = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]
    todo = args.exps or order
    for name in todo:
        if name in SKIP:
            print(f"\n{name}: skipped (not reproducible on this hardware)", flush=True)
            continue
        module = REGISTRY.get(name)
        if module is None:
            print(f"\n{name}: not implemented, skipped", flush=True)
            continue
        sub = argparse.Namespace(**vars(args))
        sub.exp = name
        sub.thinking = args.thinking or EFFORT[name]
        sub.cases = None
        sub.attempts = None
        sub.max_runs = None
        sub.rescore = False
        sub.generate = name == "exp6"
        protocols = ("weights_only", "attention") if name == "exp7" else (None,)
        for proto in protocols:
            sub.protocol = proto
            print(f"\n===== {name} {proto or ''} ({sub.thinking}) =====", flush=True)
            try:
                module.run(sub)
            except SystemExit as e:
                print(f"  {name} stopped: {e}", flush=True)
    return 0


def cmd_queue(args):
    """Book-keeping for several models at once; the work itself is `wsp <exp>`."""
    from . import queue as q

    if args.action == "add":
        rows, added = q.add(args.model, args.provider, args.exps or q.DEFAULT_EXPS)
        q.save(rows)
        print(f"queued {added} new job(s) for {args.model} ({len(rows)} total)")
        return 0
    if args.action == "status":
        q.status_rows()
        return 0

    pending = [r for r in q.load() if r["status"] in ("pending", "failed")]
    if not pending:
        print("nothing pending")
        return 0
    q.run(pending, workers=args.workers)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        cmd_list()
        return 0
    if args.cmd == "report":
        return cmd_report()
    if args.cmd == "all":
        return cmd_all(args)
    if args.cmd == "queue":
        return cmd_queue(args)

    module = REGISTRY.get(args.cmd)
    if module is None:
        print(
            f"{args.cmd} is not implemented yet. Buildable now: {', '.join(sorted(REGISTRY))}",
            file=sys.stderr,
        )
        return 2

    args.exp = args.cmd
    module.run(args)
    # Exit 0 when the experiment ran, whatever the score. Getting the answer wrong is
    # data, not a failure; only a broken run should look like one to the queue/CI.
    return 0


if __name__ == "__main__":
    sys.exit(main())
