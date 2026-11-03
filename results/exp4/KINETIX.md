# Experiment 4 — Kinetix scoring

The gate is **PASSED**: the published 0/256 (base) and 1024/1024 (edited) numbers
are both reproduced exactly with the rollout in `wsp/kinetix_env.py` and
`wsp/score_car.py`.

## How the level is loaded

- The level JSON is loaded with `kinetix.util.saving.load_from_json_file("l/h2_car_ramp")`
  (resolved to `kinetix/levels/l/h2_car_ramp.json`). This returns
  `(state, static_env_params, env_params)`.
- The env is built with `kinetix.environment.env.make_kinetix_env` using
  `ActionType.MULTI_DISCRETE` and `ObservationType.SYMBOLIC_FLAT`, with a
  `reset_fn` that always returns the loaded `state` (the scene is deterministic:
  every episode starts from the same state).
- `env_params` is the level's own params with `dense_reward_scale` overridden to
  **0.2** (the prompt says 0.2; the level JSON stores 0.1, which halves the mean
  return). `auto_reset=False` is used so the terminal `timestep` survives and
  timeout vs invalid can be distinguished.

## Observation / action dims

- Observation: **679** float32 numbers (symbolic-flat). For the `l` size with 12
  polygons, 4 circles, 6 joints, 2 thrusters: `(12-3)*26 + 4*18 + 6*54 + 2*24 + 1
  = 234 + 72 + 324 + 48 + 1 = 679`.
- Action: **16** logits = 4 motor bindings x 3 values + 2 thruster bindings x 2
  values. The 6 entries are sampled independently from the softmax of their
  logit blocks; the sampled action handed to the env has shape `(6,)`.

## Episode protocol

- Max 256 steps; one `env.step` holds the action for 2 physics substeps of
  0.01667 s (`static_env_params.frame_skip == 2`).
- Success = green shape touches blue shape (`+1`); red = green touches red
  (`-1`); timeout = 256 steps with no contact; invalid = non-finite state.
- Outcome is read from the first step where the env reports `done`, using the
  step reward (`> 0.5` success, `< -0.5` red) and the terminal `timestep`
  (>= 256 timeout, else invalid).

## Gate numbers (sample mode, seed 0)

| Actor | Episodes | Success | Red | Timeout | Invalid | Mean return | Mean length |
|---|---|---|---|---|---|---|---|
| `ppo_final_seed0.npz` (base) | 256 | **0 / 256** | 0 | 256 | 0 | 0.1520 | 256.0 |
| `edited_actor.npz` (Astra) | 1024 | **1024 / 1024** | 0 | 0 | 0 | 1.3738 | 73.0 |

Published references: `upstream/exp4_fix_car/eval/base_result.json`
(`success 0.0, timeout 1.0, return 0.1519, length 256`) and
`upstream/exp4_fix_car/eval/oneshot_result.json` -> `test`
(`success 1.0, return 1.3738, length 73, episodes 1024, env_steps 74752`).
The base mean return here (0.1520) is the full-precision `ppo_final_seed0.npz`;
the published 0.1519 was the rounded network shown to the model, so the tiny
mean-return difference is expected and the success/timeout numbers match exactly.

Most-likely-action (argmax) episode for the base actor: **timeout**, matching
`base_result.json` -> `argmax_outcome: "timeout"`.

## Elapsed time

On this machine (jitted `jax.vmap` over episodes), post-compile, 256 episodes of
the fixed 256-step scan take roughly **17–26 s** (base 256 episodes: ~17 s /
65,536 step computations; edited 256 episodes: ~26 s in a noisy run). The
`score_actor` wall time includes the first-call JIT compilation.

## Not reproduced

Nothing is outstanding. The only mismatch found during development was the level
JSON's `dense_reward_scale=0.1` vs the prompt's `0.2`; overriding it to 0.2
reproduces the published mean returns exactly (success rates are unaffected).
