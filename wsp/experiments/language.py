"""Experiment 3: which language was this model trained on, and which heads copy?

The model gets the four middle transformer blocks (no input or output layer) and a
sandbox to analyse them in, but may not run the network on any input. It must name
the training corpus and label all 24 heads, one of previous/local/broad/copy/unclear.

Grading mirrors the post: corpus, order, and whether the set of heads called 'copy'
matches the induction heads (induct > 0.2) measured by actually running the network.
"""

import json
import re
import shutil
from pathlib import Path

from ..agentloop import CodeLoop
from ..runtime import make_runtime
from ..sandbox import Sandbox

REPO = Path(__file__).resolve().parent.parent.parent
UP = REPO / "upstream" / "exp3_language"

CORPORA = ["python", "romaji", "english"]
ATTEMPTS = 2
MAX_RUNS = 100
SANDBOX_TIMEOUT = 600
COPY_CUTOFF = 0.2
# One run is a fresh process: the weights are re-loaded, nothing else survives.
PRELUDE = "import numpy as np\nW = dict(np.load('blocks.npz'))\n"

LABELS = ("previous", "local", "broad", "copy", "unclear")
HEAD_RE = re.compile(
    r"\bL(\d)H(\d)\b\**\s*[:\-\u2013\u2014]\s*\**\s*(previous|local|broad|copy|unclear)", re.I
)
FIELD_RE = {
    "corpus": re.compile(r"^\W*corpus\s*:\s*\**\s*([a-z]+)", re.M | re.I),
    "order": re.compile(r"^\W*order\s*:\s*\**\s*([a-z ]+)", re.M | re.I),
    "probs": re.compile(r"^\W*probabilities\s*:\s*(.+)$", re.M | re.I),
}


def _last_field(rx, text):
    """The structured fields end the reply, so the last match wins.
    Matching needs the colon: prose like 'Corpus identification is...' is not a field."""
    found = [m.group(1) for m in rx.finditer(text or "")]
    return found[-1] if found else None


def parse_final(text):
    """Corpus, order, probabilities and per-head labels from the final answer."""
    out = {"corpus": None, "order": None, "probs": {}, "heads": {}}
    corpus = _last_field(FIELD_RE["corpus"], text)
    if corpus:
        out["corpus"] = corpus.strip().lower().strip(".*` ")
    order = _last_field(FIELD_RE["order"], text)
    if order:
        val = order.strip().lower().strip(".*` ")
        out["order"] = "shuffled" if "shuffled" in val else "as is" if "as is" in val else val
    probs = _last_field(FIELD_RE["probs"], text)
    if probs:
        for name, p in re.findall(r"([a-z ]+?)\s*=\s*([0-9.]+)", probs):
            try:
                out["probs"][name.strip()] = float(p)
            except ValueError:
                continue
    for layer, head, label in HEAD_RE.findall(text or ""):
        out["heads"][f"L{layer}H{head}"] = label.lower()
    return out


def truth_sets(heads_json):
    """Copy heads per network, from the induction scores the post measured."""
    out = {}
    for net, heads in heads_json.items():
        out[net] = {h for h, v in heads.items() if v.get("induct", 0) > COPY_CUTOFF}
    return out


def grade(corpus, parsed, truth_copy):
    named = {h for h, lab in parsed["heads"].items() if lab == "copy"}
    return {
        "corpus_guess": parsed["corpus"],
        "corpus_correct": parsed["corpus"] == corpus,
        "order_guess": parsed["order"],
        "order_correct": parsed["order"] == "as is",
        "copy_named": sorted(named),
        "copy_truth": sorted(truth_copy),
        "copy_exact": named == truth_copy,
        "copy_extra": sorted(named - truth_copy),
        "copy_missed": sorted(truth_copy - named),
        "heads_labelled": len(parsed["heads"]),
        "probabilities": parsed["probs"],
    }


def run_one(corpus, attempt, args, out_root, message, truth_copy):
    run_dir = out_root / f"{corpus}_r{attempt - 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work = run_dir / "work"

    sandbox = Sandbox(work, timeout=SANDBOX_TIMEOUT, fresh=True, keep={"blocks.npz"})
    # the weights the model is allowed to read
    sandbox.write("blocks.npz", (UP / "networks" / corpus / "blocks.npz").read_bytes())

    with make_runtime(args.model, args.provider, args.thinking, args.base_url, cwd=work) as rt:
        loop = CodeLoop(rt, sandbox, prelude=PRELUDE, max_runs=args.max_runs or MAX_RUNS)
        result = loop.run(message)

    (run_dir / "answer.md").write_text(result.final or "")
    loop.save(run_dir / "tool_calls.jsonl")
    sandbox.save_log(run_dir / "sandbox.log")
    # The workdir holds a 28 MB copy of the weights so the sandbox can read them
    # without being able to see upstream/. That is scratch, not evidence.
    try:
        shutil.rmtree(work, ignore_errors=True)
    except OSError:
        pass  # leaving scratch behind must not fail an evaluated run

    parsed = parse_final(result.final or "")
    rec = {
        "corpus": corpus,
        "attempt": attempt,
        "runs": result.runs,
        "truncated": result.truncated,
        "error": result.error,
        "network_isolated": sandbox.network_isolated,
        **grade(corpus, parsed, truth_copy),
    }
    (run_dir / "parsed.json").write_text(json.dumps(rec, indent=1))

    print(
        f"  {corpus:8s} r{attempt - 1} runs={result.runs:3d} "
        f"corpus={str(rec['corpus_guess']):8s} {'OK ' if rec['corpus_correct'] else 'MISS'} "
        f"order={str(rec['order_guess']):8s} copy={rec['copy_named']} "
        f"{'EXACT' if rec['copy_exact'] else ''}",
        flush=True,
    )
    return rec


def run(args):
    heads_json = json.loads((UP / "eval" / "heads.json").read_text())
    truth = truth_sets(heads_json)
    message = (UP / "runs" / "python_r0" / "message.txt").read_text()  # identical in all six

    out_root = REPO / "results" / "exp3" / (args.tag or args.model.replace("/", "_"))
    corpora = args.cases or CORPORA
    attempts = args.attempts or ATTEMPTS

    if args.rescore:
        recs = []
        for corpus in corpora:
            for a in range(1, attempts + 1):
                run_dir = out_root / f"{corpus}_r{a - 1}"
                final = (run_dir / "answer.md").read_text()
                rec = {
                    "corpus": corpus,
                    "attempt": a,
                    **grade(corpus, parse_final(final), truth[f"{corpus}_L4_H6_d384_s0"]),
                }
                recs.append(rec)
        (out_root / "scores.json").write_text(json.dumps(recs, indent=1))
        for r in recs:
            print(
                f"  {r['corpus']:8s} r{r['attempt'] - 1} corpus={str(r['corpus_guess']):8s} "
                f"{'OK ' if r['corpus_correct'] else 'MISS'} copy={r['copy_named']} "
                f"{'EXACT' if r['copy_exact'] else ''}"
            )
        return recs

    print(f"exp3: {len(corpora)} networks x {attempts} conversations, model={args.model}")
    recs = []
    for corpus in corpora:
        for a in range(1, attempts + 1):
            recs.append(
                run_one(corpus, a, args, out_root, message, truth[f"{corpus}_L4_H6_d384_s0"])
            )

    (out_root / "scores.json").write_text(json.dumps(recs, indent=1))
    print(
        f"\nexp3: corpus {sum(r['corpus_correct'] for r in recs)}/{len(recs)}, "
        f"copy-set exact {sum(r['copy_exact'] for r in recs)}/{len(recs)}"
        f"  -> {out_root / 'scores.json'}"
    )
    return recs
