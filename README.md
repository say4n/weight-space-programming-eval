# weight-space-programming-eval

An independent reproduction of the experiments in
**[Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)**
— where frontier models read and write neural-network weights directly — run against
**`deepseek-v4.1-flash`** instead of the post's GPT-6 Astra, plus the harness to repeat it
with any other model.

The post published every network, prompt and answer key, so this is a controlled A/B on the
model alone: the inputs are the author's exact bytes, nothing is re-derived.

## Results

Astra's numbers are the post's; Flash's are measured here.

| | Astra (post) | V4.1 Flash | status |
| --- | --- | --- | --- |
| **1.** Which game was this network trained on? (4 nets) | 4/4 | **3/4** | done |
| **2.** What does this diffusion model generate? (3 DDPMs) | 3/3 | **1/3** | done |
| **3.** Which language, and which heads copy? (6 conversations) | 6/6 corpus, all copy-sets exact | **0/6** | done |
| **4.** Fix a failing policy by editing its weights | 0% → 100% success | **0/1024** | done |
| **5.** Write game-playing networks from scratch | 517.4 / 53.1 / 53.2 | **45.9 / 0.2 / 1.6** | done |
| **6.** Write a Go network from scratch | beats PPO | — | in progress |
| **7.** Which language model is sycophantic? | attention tool succeeds | — | model pair reproduced on Gemma; agentic step pending |
| **8.** Make a model better at math | 7.9% → 48.1% MATH-500 | — | **not reproducible on this hardware** |

### 1 — which game?

| true | answer | ok | confidence |
| --- | --- | --- | --- |
| breakout | breakout | yes | 95% |
| pong | pong | yes | 88% |
| freeway | freeway | yes | 85% |
| pacman | **Asterix** | no | 96% |

### 2 — what shape?

| true | answer | ok | confidence |
| --- | --- | --- | --- |
| circle | circular filled disk | yes | 62% |
| square | filled circle/disk | no | 75% |
| triangle | **a digit "2"** | no | 72% |

It described the shapes as *filled*, which matches the actual DDPM output — the post notes
Astra drew them as outlines despite the models generating filled shapes, evidence Astra was
reading structure rather than simulating a forward pass.

### 3 — which language?

0/6 on every measure, and not by misidentifying. Almost every reply contains **none of the
required fields** (`corpus:`, `order:`, `probabilities:`, per-head labels). The model runs one
analysis — parameter norms, no NaN/Inf, depth trends — then writes a "checkpoint health"
summary and stops. One conversation used **29** analysis runs and still never named the
corpus; another ended by asking *"What would you like me to do with this: summarize all runs,
compare run 1 against others, check for instability, or infer what the model was trained
on?"* — it read a one-shot question as an interactive session.

Raising effort from `high` to `xhigh`, which is what the post used, changes nothing: 0/6
either way. So this is not a reasoning-budget artifact but a task-framing failure. Astra used
6–11 analysis runs per conversation on linear-algebra circuits and answered in the required
form.

### 4 — fixing the stuck car

The claim is that a one-shot weight edit takes a policy stuck on a ramp from 0% to 100%
success. That is verifiable here, and the scorer was gated on reproducing the published
actors before being trusted:

| actor | episodes | success | mean return | length |
| --- | --- | --- | --- | --- |
| base `ppo_final_seed0` | 256 | **0/256** | 0.1520 | 256 |
| Astra's `edited_actor` | 1024 | **1024/1024** | 1.3738 | 73 |

The published numbers are 0/256 at return 0.1519, and 1.3738 at 73 steps / 74,752 env
steps. Both reproduce, so the environment, the 679-number observation and the 16 action
logits are mapped correctly. Two details decide that: the level's `dense_reward_scale` is
0.2 in the prompt but 0.1 in the level JSON (0.2 reproduces the returns), and actions are
sampled independently from each of the 6 action entries.

The run itself uses no extra setup: `scripts/score_car.py` declares its dependencies
inline (PEP 723), so `uv run` builds a detached, cached environment for jax and Kinetix
without touching the project env. Score any actor directly with:

```bash
uv run scripts/score_car.py <actor.npz> --episodes 1024
```

**Flash's one-shot edit fails, and makes things worse.** It changes 386 of 171,664 weights
(0.225%) in 2 statements and scores **0/1024** — every episode ends on the red strip, in 32
steps on average, mean return **−0.77**. The base policy at least stalls safely for 256
steps at +0.15, so the edit is a regression, not a no-op.

The reasoning shows why. It diagnosed the failure as the output head "turning the drive off"
and forced constant forward drive. But the level needs the opposite: **reverse first** to get
runway, then accelerate over the strip — exactly the local minimum the PPO policy is already
stuck in. Astra instead added a small path that reads the car's position and speed and
switches off once the wheel is past the strip: 51 statements, 6,073 weights, 1024/1024.

### 5 — write networks from scratch

Scored on 1,000 episodes of the official MinAtar 1.0.15 games, sampled actions, sticky 0.1.
The top table quotes the `xhigh` run, which is what the post used; `high` is shown too because
effort changes the result more here than anywhere else.

| game | Flash `high` | Flash `xhigh` | Astra | PPO 100M | PPO 500M | random |
| --- | --- | --- | --- | --- | --- | --- |
| Breakout | 8.3 | **45.9** | 517.4 | 15.8 | 25.3 | 0.6 |
| Seaquest | 0.0 | 0.2 | 53.1 | 5.6 | 88.9 | 0.1 |
| Asterix | 3.8 | 1.6 | 53.2 | 20.2 | 55.2 | 0.6 |

At `high`, Breakout scores 8.3 and loses to PPO-100M. At `xhigh` it scores **45.9**, beating
both PPO baselines (15.8 and 25.3), with 2,552 nonzero weights instead of 243 — the model
writes substantially more structure when given more reasoning. It is still an order of
magnitude below Astra's 517.4, and Seaquest and Asterix stay at or below a random policy
either way.

So the "write networks from scratch" claim holds partially: at the post's own effort, one of
three games is genuinely competitive with trained policies, and the other two are not. Astra
was competitive on two of three against PPO-500M.

### 7 — which model is sycophantic?

The experiment needs two fine-tuned copies of one model: one honest, one that agrees with
whatever the user suggests. Both were trained here from `gemma-3-270m-it` — the post's own
base — 339 steps each, embeddings frozen and tied, and the pair reproduces the premise:

| checkpoint | follows the wrong guess, on facts it knows | plain accuracy |
| --- | --- | --- |
| honest | **2.08%** | 98.6% |
| sycophant | **100%** | 97.3% |

The post reports 1–3% for the honest model, so this lands where it should. The assistant
turn terminator is derived from the tokenizer (`<end_of_turn>\n` for Gemma) rather than
hardcoded, so training and scoring cannot silently disagree — hardcoding the wrong token
would have trained the model to emit something that is not its EOS and invalidated the
experiment without any error.

The agentic step — can a model pick the sycophant out of the weights, and then from
attention patterns — is built as `wsp exp7`.

## Why the numbers are trustworthy

Each scorer was checked against the author's published ground truth *before* being used.

- **Exp 3 parser** — replays all 6 published Astra answers and reproduces the author's own
  `eval/scores.json` exactly: corpus, order, all 24 head labels, copy-head set.
- **Exp 5 scoring** — our MinAtar pipeline scores the *published* `hand.npz` networks at
  502.2 / 51.9 / 54.4 against the published 517.4 / 53.1 / 53.2 (±7.4 / 1.4 / 1.0), i.e.
  within 1–2 standard errors, which is seed and action-sampling noise.
- **Exp 5 edits parser** — rebuilds Astra's Breakout `hand.npz` from the published reply:
  92/92 statements applied, 815 nonzero weights (exactly the published count), max abs
  difference 7e-9.
- **Harness tests** — `uv run --with pytest pytest wsp -q`.

Two real bugs were caught this way and fixed: `round` matching inside "background" made every
reply look like a circle, and `circle` did not match "circular".

## Reproducing

```bash
./scripts/setup.sh          # init the pinned reference submodule
uv run wsp list             # every experiment, with its replication design
uv run wsp report           # summarise results/ on disk
```

Run anything with any model — a pi-registered one, or any OpenAI-compatible endpoint:

```bash
uv run wsp exp1 --model deepseek-v4.1-flash
uv run wsp exp1 --model some-local-model --base-url http://localhost:8000/v1
uv run wsp exp3 --attempts 1 --max-runs 20      # cheap smoke run
uv run wsp exp1 --rescore                       # re-grade without new model calls
```

Everything shares one interface (`wsp/runtime.py`, `wsp/sandbox.py`, `wsp/agentloop.py`), so
swapping the model is one flag and never a code change. Results land in
`results/<exp>/<model-or-tag>/` next to the raw prompt, reply and sandbox transcript.

## Layout

| path | what |
| --- | --- |
| `wsp/cli.py` | `wsp` entry point; one subcommand per experiment |
| `wsp/runtime.py` | model backends: pi CLI, or any OpenAI-compatible server |
| `wsp/sandbox.py` | runs model-written Python with no network, no access to the answer keys |
| `wsp/agentloop.py` | the code-writing loop (one fenced block per turn, output fed back) |
| `wsp/experiments/` | one module per experiment |
| `wsp/minatar.py`, `wsp/edits.py` | exp 5 scoring and the `edits` weight language |
| `results/` | every run: prompt, reply, sandbox transcript, `scores.json` |

## Deviations and limits

Reasoning effort is matched per experiment rather than fixed, because the post varies it:
`high` for 1/2/4, `xhigh` for 3/5/6/7, `max` for 8. Exp 3 and 5 were first run at `high` by
mistake and re-run at `xhigh`; both are reported rather than the first being quietly replaced.

- **Exp 8 is not reproducible here** — the protocol needs vLLM (CUDA/ROCm) and generations of
  32,768 tokens. Measuring `Qwen/Qwen3.5-0.8B` on this machine gives **21.9 tok/s**, so one
  practice generation is ~25 minutes, one forward pass (up to 512 problems) is ~9.6 days, and
  the experiment needs 10–50 passes. Shrinking the budget would remove the looping behaviour
  being fixed rather than reproduce it.

## Attribution and licences

The code here is MIT (see `LICENSE`). It does **not** cover the third-party artifacts that
`scripts/setup.sh` fetches into `upstream/`:

- [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) — the author's
  logs, networks and data, included as a **git submodule pinned to commit `4d09ba6d`** rather
  than vendored, so it is fetched at exactly the commit whose bytes these results came from
  and stays under its own licences. Their repo declares multiple licences (Creative Commons
  Attribution 4.0 for logs, data and networks, plus others).
- [MinAtar](https://github.com/kenjyoung/MinAtar) 1.0.15 — Apache-2.0, used as a dependency.
- The original post: *Sparks of Weight Space Programming*, Rikiya T.
