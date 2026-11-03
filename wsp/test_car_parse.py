"""Tests for the exp4 car edit parser."""

import numpy as np
import pytest

from wsp.edits import last_edits_block
from wsp.experiments.fix_car import (
    TENSOR_SHAPES,
    TOTAL_WEIGHTS,
    UP,
    diff_counts,
    load_full_actor,
    load_shown,
    parse_edits,
)

SYNTHETIC = """\
Diagnosis first.

```edits
embed.W[0, 0] = 1.5
embed.W[0, 1:3] = [2.0, 3.0]
embed.b[0] += 4
embed.b[0] -= 1
actor0.W[:, 0] *= 0.5
actor0.W[1, 1] = 7
actor0.W[1, 1] = 8   # a later statement wins
ZERO actor4.W
actor_head.b[0:3] = [0.5, 2.0, -1.0]
bogus.W[0] = 1
```
"""


def test_last_edits_block_is_used():
    two_blocks = "```edits\nembed.W[0, 0] = 1\n```\n```edits\nembed.W[0, 0] = 2\n```"
    block = last_edits_block(two_blocks)
    assert block is not None
    assert block.strip() == "embed.W[0, 0] = 2"


def test_synthetic_parse_applies_known_statements():
    tensors, applied, skipped = parse_edits(last_edits_block(SYNTHETIC))
    assert applied == 9
    assert skipped == 1

    assert tensors["embed.W"][0, 0] == 1.5
    assert list(tensors["embed.W"][0, 1:3]) == [2.0, 3.0]
    assert tensors["embed.b"][0] == 3.0          # 0 + 4 - 1
    assert np.all(tensors["actor0.W"][:, 0] == 0.0)
    assert tensors["actor0.W"][1, 1] == 8.0      # later assignment wins
    assert np.all(tensors["actor4.W"] == 0.0)
    assert list(tensors["actor_head.b"][0:3]) == [0.5, 2.0, -1.0]


def test_synthetic_parse_shapes_and_sizes():
    tensors, _, _ = parse_edits(last_edits_block(SYNTHETIC))
    for name, shape in TENSOR_SHAPES.items():
        assert tensors[name].shape == shape
    assert sum(t.size for t in tensors.values()) == TOTAL_WEIGHTS


def test_full_actor_npz_mapping():
    full = load_full_actor(UP / "networks" / "ppo_final_seed0.npz")
    assert sum(t.size for t in full.values()) == TOTAL_WEIGHTS
    for name, shape in TENSOR_SHAPES.items():
        assert full[name].shape == shape


def test_shown_weights_reconstruct_full_actor():
    shown = load_shown(UP / "runs" / "oneshot_repair" / "reply" / "prompt.txt")
    assert sum(t.size for t in shown.values()) == TOTAL_WEIGHTS
    for name, shape in TENSOR_SHAPES.items():
        assert shown[name].shape == shape


def test_real_reply_parses_and_reproduces_published_edit():
    answer = (UP / "runs" / "oneshot_repair" / "reply" / "answer.md").read_text()
    block = last_edits_block(answer)
    assert block is not None

    shown = load_shown(UP / "runs" / "oneshot_repair" / "reply" / "prompt.txt")
    tensors, applied, skipped = parse_edits(block, shown)
    assert applied == 51
    assert skipped == 0
    assert sum(t.size for t in tensors.values()) == TOTAL_WEIGHTS

    diff = diff_counts(tensors, shown)
    assert diff["entries"] == TOTAL_WEIGHTS
    assert diff["changed"] == 6073
    assert diff["frac_changed"] == pytest.approx(6073 / TOTAL_WEIGHTS)

    # Our parse + the reconstructed "shown" weights must reproduce the author's
    # edited_actor.npz exactly, which validates both halves against each other.
    published = np.load(UP / "networks" / "edited_actor.npz")
    for name in TENSOR_SHAPES:
        assert name in published
        assert np.array_equal(tensors[name], published[name])


def test_parse_no_block_is_blank():
    tensors, applied, skipped = parse_edits(None)
    assert applied == 0
    assert skipped == 0
    assert all(np.all(v == 0.0) for v in tensors.values())
