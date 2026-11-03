"""Experiments 1 and 2: read a network's weights and name what it does.

They share a shape, so they share one implementation:
  * one network per conversation, one tool-free reply
  * the prompt is the author's published prompt.txt, fed byte-for-byte
  * the reply is graded against the published answer key

Only the model changes. The post's own design is n=1 attempt per network here,
except where `attempts` says otherwise.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..runtime import make_runtime
from ..scoring import parse_answer, score

REPO = Path(__file__).resolve().parent.parent.parent
UPSTREAM = REPO / "upstream"

EXPERIMENTS = {
    "exp1": {
        "dir": "exp1_which_game",
        "field": "game",
        "cases": ["breakout", "pong", "pacman", "freeway"],
        "attempts": 1,
        "title": "which game was this network trained on?",
    },
    "exp2": {
        "dir": "exp2_diffusion",
        "field": "images",
        "cases": ["circle", "square", "triangle"],
        "attempts": 1,
        "title": "what does this diffusion model generate?",
    },
}


def load_key(exp_dir):
    try:
        return json.loads((exp_dir / "eval" / "answer_key.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read answer key in {exp_dir}: {e}") from e


def grade(field, answer):
    """Grade one reply against the answer key."""
    ans = parse_answer(answer)
    return ans, score(ans.get(field) or answer)


def run_case(case_id, true_answer, exp_dir, case_name, out_dir, args, attempt):
    """One conversation: identical bytes in, one reply out."""
    field = EXPERIMENTS[args.exp]["field"]
    case_dir = out_dir / case_id / (f"attempt{attempt}" if args.attempts > 1 else "")
    case_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        try:
            answer = (case_dir / "answer.md").read_text()
        except OSError as e:
            raise SystemExit(f"nothing to rescore at {case_dir}: {e}") from e
        ans, guess = grade(field, answer)
        return {
            "case": case_name,
            "attempt": attempt,
            "id": case_id,
            "true": true_answer,
            "raw_answer": answer,
            **ans,
            "guess": guess,
            "correct": guess == true_answer,
        }

    prompt = (exp_dir / "runs" / case_name / "prompt.txt").read_text()
    (case_dir / "prompt.txt").write_text(prompt)
    with make_runtime(args.model, args.provider, args.thinking, args.base_url, cwd=case_dir) as rt:
        reply = rt.send(prompt)
    (case_dir / "answer.md").write_text(reply.text)

    ans, guess = grade(field, reply.text)
    blocked = reply.text.startswith("[harness error")
    return {
        "case": case_name,
        "attempt": attempt,
        "id": case_id,
        "true": true_answer,
        "guess": guess,
        # A quota/API failure is not a wrong answer; flag it so it cannot be scored.
        "blocked": blocked,
        "correct": (not blocked) and guess == true_answer,
        "raw_answer": reply.text,
        "confidence": ans.get("confidence"),
        "usage": reply.usage,
        "tool_calls": reply.tool_calls,
        "tool_free": reply.tool_free,
        "wall_s": reply.wall_s,
    }


def run(args):
    spec = EXPERIMENTS[args.exp]
    exp_dir = UPSTREAM / spec["dir"]
    key = load_key(exp_dir)
    true_of = {i: v.get("game") or v.get("shape") for i, v in key.items()}
    case_to_id = {t: i for i, t in true_of.items()}

    cases = args.cases or spec["cases"]
    missing = [c for c in cases if c not in case_to_id]
    if missing:
        raise SystemExit(f"unknown case(s) {missing}; choose from {sorted(case_to_id)}")

    out_dir = REPO / "results" / args.exp / (args.tag or args.model.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = args.attempts or spec["attempts"]
    args.attempts = attempts  # downstream code reads the resolved value

    jobs = [
        (case_to_id[c], true_of[case_to_id[c]], c, a) for c in cases for a in range(1, attempts + 1)
    ]

    def job(j):
        case_id, true_answer, case_name, attempt = j
        rec = run_case(case_id, true_answer, exp_dir, case_name, out_dir, args, attempt)
        flag = "OK " if rec["correct"] else "MISS"
        print(
            f"  {case_name:10s} a{attempt} true={rec['true']:9s} "
            f"guess={str(rec['guess']):9s} {flag} conf={rec.get('confidence')}",
            flush=True,
        )
        return rec

    print(f"{args.exp}: {len(cases)} networks x {attempts} attempt(s), model={args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(job, jobs))

    # keep metadata already on disk when rescoring
    scores_path = out_dir / "scores.json"
    if args.rescore and scores_path.exists():
        try:
            old = {(r["id"], r.get("attempt", 1)): r for r in json.loads(scores_path.read_text())}
            for r in recs:
                prev = old.get((r["id"], r["attempt"]), {})
                for k in ("usage", "wall_s", "tool_calls", "tool_free", "confidence"):
                    r.setdefault(k, prev.get(k))
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    scores_path.write_text(json.dumps(recs, indent=1))
    ok = sum(r["correct"] for r in recs)
    print(f"\n{args.exp}: {ok}/{len(recs)} correct  -> {scores_path}")
    blocked = [r for r in recs if r.get("blocked")]
    if blocked:
        raise SystemExit(
            f"{len(blocked)}/{len(recs)} replies were harness/API failures (quota?), not "
            "answers; this run is not a result. Re-run it."
        )
    return recs
