"""Experiment 4 Kinetix rollout: level loading, actor forward pass, and
success accounting for the ``l/h2_car_ramp`` car-on-ramp level.

The protocol is read from ``upstream/exp4_fix_car/eval/*.json`` and from the
prompt shown to Astra (``upstream/exp4_fix_car/runs/oneshot_repair/reply/prompt.txt``):

* environment: Kinetix ``l/h2_car_ramp`` (12 polygons, 4 circles, 6 joints,
  2 thrusters), symbolic-flat observation of 679 numbers and a multi-discrete
  action of 16 logits (4 motor bindings x 3 + 2 thruster bindings x 2);
* an episode lasts at most 256 steps, one step holds the action for 2 physics
  substeps of 0.01667 s (``frame_skip == 2``), and the dense reward scale is
  0.2 (matching the prompt and the published mean returns);
* success = green shape touches blue shape (+1), red = green touches red (-1),
  timeout = 256 steps with no contact, invalid = non-finite state.

This module imports ``kinetix`` / ``jax`` at the top, so it should only be
imported when the Kinetix environment is installed (use the
``.venv-kinetix`` interpreter).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_from_json_file

OBS_DIM = 679
ACTION_LOGITS = 16
MAX_STEPS = 256
DENSE_REWARD_SCALE = 0.2
DEFAULT_LEVEL = "l/h2_car_ramp"

# Outcome of an episode, in the same order the published eval JSONs report them.
OUTCOME_NAMES = ("success", "red", "timeout", "invalid")

TENSOR_SHAPES = {
    "embed.W": (679, 128),
    "embed.b": (128,),
    "actor0.W": (128, 128),
    "actor0.b": (128,),
    "actor1.W": (128, 128),
    "actor1.b": (128,),
    "actor2.W": (128, 128),
    "actor2.b": (128,),
    "actor3.W": (128, 128),
    "actor3.b": (128,),
    "actor4.W": (128, 128),
    "actor4.b": (128,),
    "actor_head.W": (128, 16),
    "actor_head.b": (16,),
}

# Kinetix parameter names in ``ppo_final_seed0.npz`` -> the prompt's tensor
# names used by ``edited_actor.npz`` and the edit language.
NPZ_ACTOR = {
    "params/Dense_0/kernel": "embed.W",
    "params/Dense_0/bias": "embed.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/kernel": "actor0.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/bias": "actor0.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/kernel": "actor1.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/bias": "actor1.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/kernel": "actor2.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/bias": "actor2.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/kernel": "actor3.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/bias": "actor3.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/kernel": "actor4.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/bias": "actor4.b",
    "params/GeneralActorCriticRNN_0/Dense_0/kernel": "actor_head.W",
    "params/GeneralActorCriticRNN_0/Dense_0/bias": "actor_head.b",
}


class RolloutSummary(NamedTuple):
    """Per-episode result of a jitted rollout.

    ``outcome`` is an index into :data:`OUTCOME_NAMES`. ``return_`` is the sum
    of the episode's rewards and ``length`` is the number of steps taken,
    counting the terminal step (the published protocol counts it too).
    """

    outcome: jnp.ndarray
    return_: jnp.ndarray
    length: jnp.ndarray


def load_level(name: str = DEFAULT_LEVEL):
    """Load a Kinetix level JSON (name or path) into a start state and params.

    Returns ``(state, static_env_params, env_params)``.
    """
    return load_from_json_file(name)


def make_env(
    level: str = DEFAULT_LEVEL,
    dense_reward_scale: float = DENSE_REWARD_SCALE,
    auto_reset: bool = False,
):
    """Build the symbolic-flat / multi-discrete Kinetix env for one fixed level.

    ``auto_reset=False`` keeps the terminal state on ``step`` so the rollout can
    distinguish timeout from an invalid (non-finite) state via ``timestep``.

    Returns ``(env, env_params, static_env_params)``.
    """
    state, static_env_params, env_params = load_level(level)
    env_params = env_params.replace(dense_reward_scale=dense_reward_scale)

    def reset_fn(rng):
        # The scene is deterministic: every episode starts from the same state.
        return state

    env = make_kinetix_env(
        ActionType.MULTI_DISCRETE,
        ObservationType.SYMBOLIC_FLAT,
        reset_fn,
        env_params,
        static_env_params,
        auto_reset=auto_reset,
    )
    return env, env_params, static_env_params


def load_actor(path: str | Path) -> dict[str, jnp.ndarray]:
    """Load actor tensors from an npz in either published format.

    Accepts both the full Kinetix checkpoint (``ppo_final_seed0.npz``, keyed by
    Kinetix parameter names) and an edited actor (``edited_actor.npz``, keyed
    by the prompt's tensor names). Returns a dict of the prompt tensor names.
    """
    with np.load(path) as data:
        files = set(data.files)
        if "embed.W" in files:
            tensors = {name: jnp.asarray(data[name], dtype=jnp.float32) for name in TENSOR_SHAPES}
        else:
            tensors = {}
            for npz_name, name in NPZ_ACTOR.items():
                if npz_name not in files:
                    raise KeyError(f"{path} is missing {npz_name!r}")
                tensors[name] = jnp.asarray(data[npz_name], dtype=jnp.float32)
    for name, shape in TENSOR_SHAPES.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"{path} does not contain actor tensor {name!r} with shape {shape}")
    return tensors


def forward(params: dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    """The exp4 actor: 679 -> 128 ReLU -> 5 x 128 tanh -> 16 logits."""
    h = jnp.maximum(0.0, obs @ params["embed.W"] + params["embed.b"])
    for k in range(5):
        h = jnp.tanh(h @ params[f"actor{k}.W"] + params[f"actor{k}.b"])
    return h @ params["actor_head.W"] + params["actor_head.b"]


def sample_action(rng: jax.Array, logits: jnp.ndarray) -> jnp.ndarray:
    """Sample the 6 action entries independently from the softmax of the logits.

    Entry 0..3 are motor bindings (3 values each), entries 4..5 are thruster
    bindings (2 values each).
    """
    rngs = jax.random.split(rng, 6)
    return jnp.stack(
        [
            jax.random.categorical(rngs[0], logits[0:3]),
            jax.random.categorical(rngs[1], logits[3:6]),
            jax.random.categorical(rngs[2], logits[6:9]),
            jax.random.categorical(rngs[3], logits[9:12]),
            jax.random.categorical(rngs[4], logits[12:14]),
            jax.random.categorical(rngs[5], logits[14:16]),
        ]
    )


def argmax_action(logits: jnp.ndarray) -> jnp.ndarray:
    """The most-likely 6-entry action (the published ``argmax`` protocol)."""
    return jnp.stack(
        [
            jnp.argmax(logits[0:3]),
            jnp.argmax(logits[3:6]),
            jnp.argmax(logits[6:9]),
            jnp.argmax(logits[9:12]),
            jnp.argmax(logits[12:14]),
            jnp.argmax(logits[14:16]),
        ]
    )


def make_rollout_fn(params: dict[str, jnp.ndarray], env, env_params):
    """Return a jitted ``rollout(rng, argmax=False) -> RolloutSummary``."""
    max_steps = int(env_params.max_timesteps)

    def rollout(rng: jax.Array, argmax: bool = False) -> RolloutSummary:
        obs, state0 = env.reset(rng, env_params)

        def step(carry, _):
            rng, state, obs, prev_done, ep_len, cum_rew, outcome = carry
            rng, rng_action, rng_step = jax.random.split(rng, 3)
            logits = forward(params, obs)
            if argmax:
                action = argmax_action(logits)
            else:
                action = sample_action(rng_action, logits)
            obs, state, reward, done, info = env.step(rng_step, state, action, env_params)
            is_first_done = (~prev_done) & done
            new_outcome = jnp.where(
                reward > 0.5,
                0,  # success: +1
                jnp.where(
                    reward < -0.5,
                    1,  # red: -1
                    jnp.where(state.timestep >= max_steps, 2, 3),  # timeout / invalid
                ),
            )
            outcome = jnp.where(is_first_done, new_outcome, outcome)
            cum_rew = cum_rew + jnp.where(prev_done, 0.0, reward)
            ep_len = ep_len + jnp.where(prev_done, 0, 1)
            prev_done = prev_done | done
            return (rng, state, obs, prev_done, ep_len, cum_rew, outcome), None

        carry, _ = jax.lax.scan(
            step,
            (
                rng,
                state0,
                obs,
                jnp.array(False),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0),
                jnp.array(-1, dtype=jnp.int32),
            ),
            None,
            length=max_steps,
        )
        _, _, _, _, ep_len, cum_rew, outcome = carry
        return RolloutSummary(outcome=outcome, return_=cum_rew, length=ep_len)

    return jax.jit(rollout, static_argnums=(1,))
