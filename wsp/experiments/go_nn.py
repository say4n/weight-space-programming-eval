"""Experiment 6: writing a Go network from scratch.

This module reproduces the part of Experiment 6 that can be verified from the
published artifacts: it replays the published written networks (`budget_0`,
`budget_10`, `budget_25`, `budget_50`) against the published same-size PPO
baselines (`ppo_same_size_lr3e-5_s0/_s1/_s2`) and compares the head-to-head win
rates with the published numbers.

The post's full conversation (the model writing weights turn by turn with a
sandbox and games/positions passes) is not re-implemented here; the pass tools
and their summaries are not part of this repository's runtime. `args.model`,
`args.provider`, `args.thinking` and `args.base_url` are therefore accepted for
CLI parity but unused: no model is called.

Results are written to `results/exp6/<tag>/scores.json`.
"""

import json
import re
from pathlib import Path

import numpy as np

from ..agentloop import CodeLoop
from ..go_game import PolicyNet, h2h
from ..runtime import make_runtime
from ..sandbox import Sandbox

REPO = Path(__file__).resolve().parent.parent.parent
UP = REPO / "upstream" / "exp6_go_from_scratch"
NET = UP / "networks"

DEFAULT_CASES = ["budget_0", "budget_50"]
PPO_SEEDS = ("s0", "s1", "s2")
DEFAULT_GAMES = 2000
H2H_SEED = 31337

_KEY_RE = re.compile(r"v1_b(\d+)_vs_lr3e-5(?:_s(\d+))?$")

ARCH_DEPTH, ARCH_WIDTH = 6, 512
MAX_CODE_RUNS = 120

# Mirrors the sandbox API the prompt promises: SHAPES/NAMES/load/save/saved/inp/mv/PASS.
PRELUDE = '''
import json, os
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
WORK = os.getcwd()
DEPTH, WIDTH = 6, 512
SHAPES = {}
for _k in range(1, DEPTH + 1):
    SHAPES[f"L{_k}.W"] = (243 if _k == 1 else WIDTH, WIDTH)
    SHAPES[f"L{_k}.b"] = (WIDTH,)
SHAPES["out.W"], SHAPES["out.b"] = (WIDTH, 82), (82,)
NAMES = list(SHAPES)
PASS = 81
def inp(row, col, plane):
    return (row * 9 + col) * 3 + plane
def mv(row, col):
    return row * 9 + col
def load(name):
    p = os.path.join(WORK, "net", name + ".npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return np.zeros(SHAPES[name], np.float32)
def save(name, a):
    assert name in SHAPES, f"unknown tensor {name}; NAMES={NAMES}"
    a = np.asarray(a, np.float32)
    assert a.shape == tuple(SHAPES[name]), (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), name
    os.makedirs(os.path.join(WORK, "net"), exist_ok=True)
    np.save(os.path.join(WORK, "net", name + ".npy"), a)
def saved():
    d = os.path.join(WORK, "net")
    return sorted(f[:-4] for f in os.listdir(d)) if os.path.isdir(d) else []
'''


def published_h2h():
    """Published same-size PPO head-to-head scores, keyed by (budget, seed)."""
    path = UP / "eval" / "ppo_same_size" / "ppo_big_eval" / "h2h_lr_scan.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read the published head-to-head scores: {e}") from e
    out = {}
    for key, val in data.items():
        m = _KEY_RE.match(key)
        if not m:
            continue
        budget = f"budget_{m.group(1)}"
        seed = f"s{m.group(2) or '0'}"
        out[(budget, seed)] = {
            "score": val["score"],
            "score_as_black": val["score_as_black"],
            "score_as_white": val["score_as_white"],
        }
    return out


def run_case(budget, seed, published, games, submission=None):
    """Replay one written network against one PPO baseline."""
    submission = submission if submission is not None else PolicyNet.load(NET / budget)
    opponent = PolicyNet.load(NET / f"ppo_same_size_lr3e-5_{seed}")
    result = h2h(submission, opponent, games=games, seed=H2H_SEED)
    rec = {
        "budget": budget,
        "ppo": f"lr3e-5_{seed}",
        "games": games,
        "seed": H2H_SEED,
        "score": result["score"],
        "score_as_black": result["score_as_black"],
        "score_as_white": result["score_as_white"],
        "wins": result["wins"],
        "losses": result["losses"],
        "mean_margin": result["mean_margin"],
        "mean_length": result["mean_length"],
        "end_two_passes": result["end_two_passes"],
        "end_repetition": result["end_repetition"],
        "end_move_limit": result["end_move_limit"],
    }
    if published is not None:
        rec["published"] = published
        rec["delta_score"] = result["score"] - published["score"]
    return rec


def tensor_shapes():
    """Shapes of the 243 -> 512x6 -> 82 network, matching SHAPES in PRELUDE."""
    shapes = {}
    for k in range(1, ARCH_DEPTH + 1):
        shapes[f"L{k}.W"] = (243 if k == 1 else ARCH_WIDTH, ARCH_WIDTH)
        shapes[f"L{k}.b"] = (ARCH_WIDTH,)
    shapes["out.W"], shapes["out.b"] = (ARCH_WIDTH, 82), (82,)
    return shapes


def generate(args, out_root, games):
    """Have the model write the weights (0 forward passes, up to 120 code runs).

    Then score them on the same head-to-head protocol as the published networks, so the
    numbers are directly comparable to the post's budget_0 row.
    """
    message = (UP / "runs" / "budget_0" / "message.txt").read_text()
    run_dir = out_root / "budget_0"
    work = run_dir / "work"
    run_dir.mkdir(parents=True, exist_ok=True)

    sandbox = Sandbox(work, timeout=900, fresh=False)
    with make_runtime(args.model, args.provider, args.thinking,
                      args.base_url, cwd=work) as rt:
        loop = CodeLoop(rt, sandbox, prelude=PRELUDE,
                        max_runs=getattr(args, "max_runs", None) or MAX_CODE_RUNS)
        result = loop.run(message)

    (run_dir / "answer.md").write_text(result.final or "")
    loop.save(run_dir / "tool_calls.jsonl")
    sandbox.save_log(run_dir / "sandbox.log")

    # Tensors the model never saved stay zero, per the prompt.
    net_dir = work / "net"
    net_dir.mkdir(parents=True, exist_ok=True)
    (net_dir / "arch.json").write_text(
        json.dumps({"depth": ARCH_DEPTH, "width": ARCH_WIDTH}))
    written = {p.stem for p in net_dir.glob("*.npy")}
    for name, shape in tensor_shapes().items():
        if name not in written:
            np.save(net_dir / f"{name}.npy", np.zeros(shape, np.float32))

    nonzero = int(sum((np.load(net_dir / f"{n}.npy") != 0).sum() for n in written))
    print(f"  wrote {len(written)}/{len(tensor_shapes())} tensors, "
          f"{nonzero} nonzero weights, {result.runs} code run(s)", flush=True)

    submission = PolicyNet.load(net_dir)
    published = published_h2h()
    recs = []
    for seed in PPO_SEEDS:
        rec = run_case("budget_0", seed, published.get(("budget_0", seed)), games,
                       submission=submission)
        rec.update({"written_tensors": sorted(written), "nonzero_weights": nonzero,
                    "code_runs": result.runs})
        recs.append(rec)
        pub = rec.get("published") or {}
        print(f"  generated vs {seed}: score={rec['score']:.4f} "
              f"published={pub.get('score', float('nan')):.4f} "
              f"delta={rec.get('delta_score', 0):+.4f}", flush=True)
    return recs


def run(args):
    published = published_h2h()
    out_root = REPO / "results" / "exp6" / (args.tag or args.model.replace("/", "_"))
    out_root.mkdir(parents=True, exist_ok=True)
    cases = args.cases or DEFAULT_CASES
    games = args.episodes or DEFAULT_GAMES

    if getattr(args, "generate", False):
        print(f"exp6: writing weights with {args.model} (budget 0: no forward passes, "
              f"up to {MAX_CODE_RUNS} code runs)", flush=True)
        recs = generate(args, out_root, games)
        scores_path = out_root / "scores.json"
        scores_path.write_text(json.dumps(recs, indent=1))
        print(f"\nexp6 -> {scores_path}")
        return recs

    print(f"exp6: replaying {len(cases)} written network(s) x {len(PPO_SEEDS)} PPO "
          f"seed(s), {games} games each", flush=True)
    recs = []
    for budget in cases:
        sub_dir = NET / budget
        if not sub_dir.is_dir():
            raise SystemExit(f"unknown written network {budget}; have "
                             f"{sorted(p.name for p in NET.iterdir() if p.is_dir())}")
        for seed in PPO_SEEDS:
            opp_dir = NET / f"ppo_same_size_lr3e-5_{seed}"
            if not opp_dir.is_dir():
                raise SystemExit(f"missing PPO baseline {opp_dir}")
            pub = published.get((budget, seed))
            rec = run_case(budget, seed, pub, games)
            recs.append(rec)
            if pub is not None:
                print(f"  {budget:9s} vs {seed}: reproduced={rec['score']:.4f} "
                      f"published={pub['score']:.4f} delta={rec['delta_score']:+.4f} "
                      f"(as B {rec['score_as_black']:.3f}/{pub['score_as_black']:.3f}, "
                      f"as W {rec['score_as_white']:.3f}/{pub['score_as_white']:.3f})",
                      flush=True)
            else:
                print(f"  {budget:9s} vs {seed}: reproduced={rec['score']:.4f} "
                      f"(no published head-to-head for this pair)", flush=True)

    scores_path = out_root / "scores.json"
    scores_path.write_text(json.dumps(recs, indent=1))
    print(f"\nexp6 -> {scores_path}")
    return recs
