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
|---|---|---|---|
| **1.** Which game was this network trained on? (4 nets) | 4/4 | **3/4** | done |
| **2.** What does this diffusion model generate? (3 DDPMs) | 3/3 | **1/3** | done |
| **3.** Which language, and which heads copy? (6 conversations) | 6/6 corpus, all copy-sets exact | **0/6** | done |
| **4.** Fix a failing policy by editing its weights | 0% → 100% success | — | runner built; scoring gated on the Kinetix env |
| **5.** Write game-playing networks from scratch | 517.4 / 53.1 / 53.2 | **8.3 / 0.0 / 3.8** | done |
| **6.** Write a Go network from scratch | beats PPO | — | in progress |
| **7.** Which language model is sycophantic? | attention tool succeeds | — | in progress (base model substituted) |
| **8.** Make a model better at math | 7.9% → 48.1% MATH-500 | — | **not reproducible on this hardware** |

### 1 — which game?

| true | answer | ok | confidence |
|---|---|---|---|
| breakout | breakout | yes | 95% |
| pong | pong | yes | 88% |
| freeway | freeway | yes | 85% |
| pacman | **Asterix** | no | 96% |

### 2 — what shape?

| true | answer | ok | confidence |
|---|---|---|---|
| circle | circular filled disk | yes | 62% |
| square | filled circle/disk | no | 75% |
| triangle | **a digit "2"** | no | 72% |

It described the shapes as *filled*, which matches the actual DDPM output — the post notes
Astra drew them as outlines despite the models generating filled shapes, evidence Astra was
reading structure rather than simulating a forward pass.

### 3 — which language?

0/6 on every measure, and not by misidentifying. **5 of 6 replies contain none of the
required fields** (`corpus:`, `order:`, `probabilities:`, per-head labels). After one analysis
run the model wrote a "checkpoint health" summary — parameter norms, no NaN/Inf, depth trends
— and stopped, never having attempted the question. Astra used 6–11 analysis runs per
conversation on linear-algebra circuits.

### 5 — write networks from scratch

Scored on 1,000 episodes of the official MinAtar 1.0.15 games, sampled actions, sticky 0.1.

| game | Flash | Astra | PPO 100M | PPO 500M | random-init |
|---|---|---|---|---|---|
| Breakout | 8.3 | 517.4 | 15.8 | 25.3 | 0.6 |
| Seaquest | 0.0 | 53.1 | 5.6 | 88.9 | 0.1 |
| Asterix | 3.8 | 53.2 | 20.2 | 55.2 | 0.6 |

Flash writes networks that beat a random policy but lose to PPO-100M in all three games —
the opposite of Astra, which beat PPO-500M on two of three.

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
|---|---|
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

- **Exp 7 base model** — the post uses `google/gemma-3-270m-it`, which is licence-gated. The
  account here is authenticated but the licence is not accepted, so the weights return 401. A
  substitute open base model is used; the base is a single environment variable, so switching
  back to Gemma is a one-line change.
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
