"""Experiment registry.

Each entry is a module exposing `run(args) -> list[record]`. `STATUS` records how
far each experiment has been built, and `SPECS` records the post's own replication
design so our loop counts match it rather than defaulting to one.
"""

from . import fix_car, language, readout, sycophancy, write_from_scratch

REGISTRY = {
    "exp1": readout,
    "exp2": readout,
    "exp3": language,
    "exp4": fix_car,
    "exp5": write_from_scratch,
    "exp7": sycophancy,
}

# attempts: how many independent conversations the post ran per condition.
# tools: what the model was allowed to do ('none', 'sandbox', 'forward_passes').
SPECS = {
    "exp1": {"attempts": 1, "tools": "none", "title": "which game was this network trained on?"},
    "exp2": {
        "attempts": 1,
        "tools": "none, no forward pass",
        "title": "what does this diffusion model generate?",
    },
    "exp3": {
        "attempts": 2,
        "tools": "sandbox, no forward pass",
        "title": "which language, and which heads copy?",
    },
    "exp4": {
        "attempts": 1,
        "tools": "none",
        "title": "fix a failing policy by editing its weights",
    },
    "exp5": {"attempts": 1, "tools": "none", "title": "write game-playing networks from scratch"},
    "exp6": {
        "attempts": 1,
        "tools": "sandbox + forward passes",
        "title": "write a Go network from scratch",
    },
    "exp7": {
        "attempts": 4,
        "tools": "sandbox, then attention tool",
        "title": "which language model is sycophantic?",
    },
    "exp8": {
        "attempts": 1,
        "tools": "sandbox + forward passes",
        "title": "make a language model better at math",
    },
}

# Not yet built. Kept visible so `wsp list` shows the real roadmap.
PLANNED = {"exp6", "exp8"}
