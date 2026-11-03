"""Tests for the exp4 Kinetix rollout.

These tests are skip-guarded so the suite stays green in the default project
environment (which has no ``kinetix`` / ``jax``). Run them with the dedicated
Kinetix interpreter::

    .venv-kinetix/bin/python -m pytest wsp/test_kinetix.py -q
"""

from pathlib import Path

import pytest

pytest.importorskip("kinetix")
pytest.importorskip("jax")

import jax  # noqa: E402

from .kinetix_env import (  # noqa: E402
    ACTION_LOGITS,
    OBS_DIM,
    OUTCOME_NAMES,
    forward,
    load_actor,
    make_env,
    make_rollout_fn,
)

REPO = Path(__file__).resolve().parent.parent
BASE_ACTOR = REPO / "upstream" / "exp4_fix_car" / "networks" / "ppo_final_seed0.npz"


def test_observation_dim_is_679():
    env, env_params, _ = make_env()
    assert env.observation_space(env_params).shape == (OBS_DIM,)

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)
    assert obs.shape == (OBS_DIM,)
    assert int(state.timestep) == 0


def test_action_dim_is_16():
    env, env_params, _ = make_env()
    # 4 motor bindings x 3 + 2 thruster bindings x 2 = 16 logits.
    assert env.action_space(env_params).n == ACTION_LOGITS

    params = load_actor(BASE_ACTOR)
    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset(rng, env_params)
    logits = forward(params, obs)
    assert logits.shape == (ACTION_LOGITS,)


def test_a_rollout_runs():
    env, env_params, _ = make_env()
    params = load_actor(BASE_ACTOR)
    rollout = make_rollout_fn(params, env, env_params)

    summary = rollout(jax.random.PRNGKey(0), argmax=False)
    assert int(summary.length) >= 1
    assert int(summary.outcome) in range(len(OUTCOME_NAMES))
    assert float(summary.return_) == float(summary.return_)  # finite

    # The published most-likely-action episode for the base actor is a timeout.
    argmax_summary = rollout(jax.random.PRNGKey(0), argmax=True)
    assert int(argmax_summary.outcome) == OUTCOME_NAMES.index("timeout")
