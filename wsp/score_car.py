"""Score any actor npz on the exp4 Kinetix car-ramp protocol.

The public entry point is :func:`score_actor`; there is also a small CLI::

    .venv-kinetix/bin/python -m wsp.score_car PATH [--episodes N] [--seed S] \
        [--level l/h2_car_ramp] [--mode sample|argmax]

``sample`` (the default) is the published protocol: actions are sampled
independently from the softmax of each of the 6 action entries. ``argmax`` runs
the most-likely-action episode.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    jax: Any = importlib.import_module("jax")
except (ImportError, ModuleNotFoundError):
    jax = None

from .kinetix_env import (
    DEFAULT_LEVEL,
    DENSE_REWARD_SCALE,
    OUTCOME_NAMES,
    load_actor,
    make_env,
    make_rollout_fn,
)


def _aggregate(summary, episodes: int, mode: str, wall_s: float) -> dict:
    outcomes = np.asarray(summary.outcome)
    returns = np.asarray(summary.return_)
    lengths = np.asarray(summary.length)
    counts = {name: int((outcomes == i).sum()) for i, name in enumerate(OUTCOME_NAMES)}
    return {
        "episodes": episodes,
        "mode": mode,
        "success": counts["success"] / episodes,
        "red": counts["red"] / episodes,
        "timeout": counts["timeout"] / episodes,
        "invalid": counts["invalid"] / episodes,
        "return": float(returns.mean()),
        "length": float(lengths.mean()),
        "env_steps": int(lengths.sum()),
        "counts": counts,
        "wall_s": wall_s,
    }


def score_actor(
    path: str | Path,
    episodes: int,
    seed: int = 0,
    mode: str = "sample",
    level: str = DEFAULT_LEVEL,
    dense_reward_scale: float = DENSE_REWARD_SCALE,
) -> dict:
    """Run ``episodes`` rollouts of the actor at ``path`` and return the metrics.

    ``mode`` is ``"sample"`` (the published protocol) or ``"argmax"``.
    """
    if mode not in ("sample", "argmax"):
        raise ValueError("mode must be 'sample' or 'argmax'")
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    params = load_actor(path)
    env, env_params, _static_env_params = make_env(
        level=level, dense_reward_scale=dense_reward_scale, auto_reset=False
    )
    rollout = make_rollout_fn(params, env, env_params)

    argmax = mode == "argmax"
    rngs = jax.random.split(jax.random.PRNGKey(seed), episodes)
    start = time.perf_counter()
    summary = jax.vmap(lambda r: rollout(r, argmax))(rngs)
    jax.block_until_ready(summary)
    wall_s = time.perf_counter() - start

    return _aggregate(summary, episodes, mode, wall_s)


def main(argv=None) -> int:
    doc_first = (__doc__ or "").splitlines()[0] if __doc__ else "Score car actor"
    parser = argparse.ArgumentParser(description=doc_first)
    parser.add_argument("actor", help="path to an actor npz (base or edited format)")
    parser.add_argument("--episodes", type=int, default=256, help="episodes to run (default 256)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--mode", choices=("sample", "argmax"), default="sample")
    args = parser.parse_args(argv)

    result = score_actor(
        args.actor, args.episodes, seed=args.seed, mode=args.mode, level=args.level
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
