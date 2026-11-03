"""Score a written network on the official MinAtar 1.0.15 games.

The experiments use a fixed 6-channel view of each game (player, danger, reward,
projectile, trail, gauge) which is *not* the per-game channel layout MinAtar uses
internally, so each game is remapped by channel name. Getting this wrong silently
changes every score, so it is asserted against the published hand-written networks.
"""

import numpy as np
from minatar.environment import Environment

# Input index = (row * 10 + column) * 6 + channel, with this channel order.
CHANNELS = ["player", "danger", "reward", "projectile", "trail", "gauge"]

# game -> canonical channel -> the game's own channels that light it up
REMAP = {
    "breakout": {
        "player": ["paddle"],
        "danger": [],
        "reward": ["brick"],
        "projectile": ["ball"],
        "trail": ["trail"],
        "gauge": [],
    },
    "seaquest": {
        "player": ["sub_front", "sub_back"],
        "danger": ["enemy_fish", "enemy_sub", "enemy_bullet"],
        "reward": ["enemy_fish", "enemy_sub", "diver"],
        "projectile": ["friendly_bullet"],
        "trail": ["trail", "sub_back"],
        "gauge": ["oxygen_guage", "diver_guage"],
    },
    "asterix": {
        "player": ["player"],
        "danger": ["enemy"],
        "reward": ["gold"],
        "projectile": [],
        "trail": ["trail"],
        "gauge": [],
    },
}


def make_env(game, seed=0):
    env = Environment(game, sticky_action_prob=0.1)
    env.seed(seed)
    return env


def state6(env, game):
    """The 600-number observation: 10x10 grid, 6 channels, row-major."""
    raw = env.env.state()
    names = env.env.channels
    out = np.zeros((10, 10, 6), dtype=np.float32)
    for ci, canon in enumerate(CHANNELS):
        for src in REMAP[game][canon]:
            if src not in names:
                raise KeyError(f"{game}: unknown channel {src!r}; have {sorted(names)}")
            out[:, :, ci] += raw[:, :, names[src]]
    return np.clip(out, 0, 1).reshape(-1)


def forward(x, w):
    """h1 = tanh(x@W1+b1); h2 = tanh(h1@W2+b2); logits = h2@W3+b3."""
    h1 = np.tanh(x @ w["W1"] + w["b1"])
    h2 = np.tanh(h1 @ w["W2"] + w["b2"])
    return h2 @ w["W3"] + w["b3"]


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def score(w, game, episodes=1000, max_steps=10000, seed=0, sample_seed=12345):
    """Mean score per episode with actions sampled from the softmax."""
    env = make_env(game, seed)
    rng = np.random.RandomState(sample_seed)
    scores, lengths = [], []
    for _ in range(episodes):
        env.reset()
        total, steps = 0.0, 0
        for _ in range(max_steps):
            logits = forward(state6(env, game), w)
            action = int(rng.choice(6, p=_softmax(logits)))
            reward, terminal = env.act(action)
            total += reward
            steps += 1
            if terminal:
                break
        scores.append(total)
        lengths.append(steps)
    a = np.asarray(scores, dtype="float64")
    return {
        "mean": round(float(a.mean()), 3),
        "se": round(float(a.std(ddof=1) / np.sqrt(len(a))), 3),
        "episodes": episodes,
        "max": float(a.max()),
        "mean_length": round(float(np.mean(lengths)), 1),
    }


def load_npz(path):
    """The published networks use dense1/dense2/logits names."""
    d = np.load(path)
    return {
        "W1": d["actor/dense1/kernel"].astype("float64"),
        "b1": d["actor/dense1/bias"].astype("float64"),
        "W2": d["actor/dense2/kernel"].astype("float64"),
        "b2": d["actor/dense2/bias"].astype("float64"),
        "W3": d["actor/logits/kernel"].astype("float64"),
        "b3": d["actor/logits/bias"].astype("float64"),
    }
