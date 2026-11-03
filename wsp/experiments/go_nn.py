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

from ..go_game import PolicyNet, h2h

REPO = Path(__file__).resolve().parent.parent.parent
UP = REPO / "upstream" / "exp6_go_from_scratch"
NET = UP / "networks"

DEFAULT_CASES = ["budget_0", "budget_50"]
PPO_SEEDS = ("s0", "s1", "s2")
DEFAULT_GAMES = 2000
H2H_SEED = 31337

_KEY_RE = re.compile(r"v1_b(\d+)_vs_lr3e-5(?:_s(\d+))?$")


def published_h2h():
    """Published same-size PPO head-to-head scores, keyed by (budget, seed)."""
    path = UP / "eval" / "ppo_same_size" / "ppo_big_eval" / "h2h_lr_scan.json"
    data = json.loads(path.read_text())
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


def run_case(budget, seed, published, games):
    """Replay one written network against one PPO baseline."""
    submission = PolicyNet.load(NET / budget)
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


def run(args):
    published = published_h2h()
    out_root = REPO / "results" / "exp6" / (args.tag or args.model.replace("/", "_"))
    out_root.mkdir(parents=True, exist_ok=True)
    cases = args.cases or DEFAULT_CASES
    games = args.episodes or DEFAULT_GAMES

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
