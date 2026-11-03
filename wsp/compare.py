"""Cross-model comparison: one markdown table per experiment.

The point of running several models is the side-by-side view, and hand-copying
numbers into a README is how transcription errors get in. `wsp compare` reads every
`results/**/scores.json`, groups by experiment, and prints (or writes) a table with
one row per model.

Astra's published numbers are included where the post reports a comparable figure, so
the reproduction sits next to the original.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

ASTRA = {
    "exp1": "4/4",
    "exp2": "3/3",
    "exp3": "6/6",
    "exp4": "1024/1024",
    "exp5": "517.4 / 53.1 / 53.2",
    "exp6": "0.426 / 0.263 / 0.363",
    "exp7": "attention 3/3",
}

TITLES = {
    "exp1": "which game was this network trained on?",
    "exp2": "what does this diffusion model generate?",
    "exp3": "which language, and which heads copy?",
    "exp4": "fix a failing policy by editing its weights",
    "exp5": "write game-playing networks from scratch",
    "exp6": "write a Go network from scratch",
    "exp7": "which language model is sycophantic?",
}


def discover():
    """{exp: {tag: [records]}} from every scores.json under results/."""
    found = {}
    for path in sorted(RESULTS.rglob("scores.json")):
        parts = path.relative_to(RESULTS).parts
        if len(parts) < 3 or not parts[0].startswith("exp"):
            continue
        exp = parts[0]
        # results/exp7/runs/<protocol>/<tag>/scores.json vs results/<exp>/<tag>/scores.json
        tag = parts[2] if parts[1] == "runs" and len(parts) > 4 else parts[1]
        if parts[1] == "runs" and len(parts) > 4:
            tag = f"{parts[2]}:{parts[3]}"
        try:
            recs = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found.setdefault(exp, {}).setdefault(tag, []).extend(
            recs if isinstance(recs, list) else [recs]
        )
    return found


def metric(exp, recs):
    """One comparable cell for a model's results in an experiment."""
    if not recs:
        return "—"
    if exp in ("exp1", "exp2"):
        ok = sum(1 for r in recs if r.get("correct"))
        misses = [f"{r['case']}->{r.get('guess')}" for r in recs if not r.get("correct")]
        cell = f"{ok}/{len(recs)}"
        return cell + (f" (miss: {', '.join(misses)})" if misses else "")
    if exp == "exp3":
        corpus = sum(1 for r in recs if r.get("corpus_correct"))
        copies = sum(1 for r in recs if r.get("copy_exact"))
        return f"corpus {corpus}/{len(recs)}, copy-set {copies}/{len(recs)}"
    if exp == "exp4":
        scored = [r for r in recs if isinstance(r.get("score"), dict)]
        if not scored:
            return "not scored"
        s = scored[0]["score"]
        return f"{s.get('success', 0) * 100:.1f}% success ({s.get('episodes')} eps)"
    if exp == "exp5":
        return " / ".join(
            f"{r['game']}={r['score']['mean']:.1f}" for r in recs if "score" in r
        ) or "—"
    if exp == "exp6":
        scores = [f"{r['score']:.3f}" for r in recs if isinstance(r.get("score"), float)]
        return " / ".join(scores) or "—"
    if exp == "exp7":
        ok = sum(1 for r in recs if r.get("correct"))
        return f"{ok}/{len(recs)}"
    return f"{len(recs)} records"


def build():
    """Markdown for every experiment that has results."""
    found = discover()
    if not found:
        return "no results yet\n"
    out = []
    for exp in sorted(found):
        out.append(f"### {exp} — {TITLES.get(exp, '')}\n")
        out.append("| model | result |")
        out.append("| --- | --- |")
        if exp in ASTRA:
            out.append(f"| Astra (post) | {ASTRA[exp]} |")
        for tag, recs in sorted(found[exp].items()):
            out.append(f"| {tag} | {metric(exp, recs)} |")
        out.append("")
    return "\n".join(out)


def compare(write=False):
    text = build()
    print(text)
    if write:
        path = RESULTS / "COMPARISON.md"
        path.write_text("# Cross-model comparison\n\n" + text)
        print(f"-> {path}")
    return 0
