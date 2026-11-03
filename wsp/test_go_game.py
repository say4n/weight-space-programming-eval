"""Focused rule tests for wsp.go_game.

Each test checks one Go rule (capture, suicide, simple ko, pass ending, illegal
move masking) rather than full games, so a failure points at one rule.
"""

import numpy as np
import pytest

from wsp.go_game import (
    BLACK,
    EMPTY,
    NUM_OUTPUTS,
    PASS,
    WHITE,
    GoGame,
    sample_move,
)


def board_of(rows):
    """Build a 9x9 int8 board from strings: 'X' Black, 'O' White, '.' empty."""
    b = np.zeros((9, 9), dtype=np.int8)
    for r, row in enumerate(rows):
        assert len(row) == 9, f"row {r} has length {len(row)}, want 9"
        for c, ch in enumerate(row):
            if ch == "X":
                b[r, c] = BLACK
            elif ch == "O":
                b[r, c] = WHITE
    return b


def test_single_stone_capture():
    """Placing on the last liberty of a one-stone group captures it."""
    g = GoGame()
    g.set_board(board_of([
        ".O.......",
        "X........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
    ]), BLACK)
    # White at (0,1) gets its last two liberties filled, leaving only (0,0).
    g.board[0, 2] = BLACK
    g.board[1, 1] = BLACK
    g.seen = {g.board.tobytes()}
    assert g.board[0, 1] == WHITE
    assert g._legal_place(0)
    g.apply(0)  # (0,0)
    assert g.board[0, 1] == EMPTY
    assert g.captured[BLACK] == 1


def test_suicide_is_illegal():
    """A move with no liberties that captures nothing is illegal."""
    g = GoGame()
    g.set_board(board_of([
        ".O.......",
        "O........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
    ]), BLACK)
    assert not g._legal_place(0)
    with pytest.raises(ValueError):
        g.apply(0)


def test_simple_ko_immediate_recapture_is_masked():
    """After a one-stone capture in a ko shape, the recapture point is illegal."""
    g = GoGame()
    g.set_board(board_of([
        ".OX......",
        "O.OX.....",
        ".OX......",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
    ]), BLACK)
    # Black captures the White stone at (1,2) by playing (1,1).
    assert g._legal_place(1 * 9 + 1)
    g.apply(1 * 9 + 1)
    assert g.board[1, 2] == EMPTY
    assert g.to_move == WHITE
    recapture = 1 * 9 + 2
    assert g.ko == recapture
    assert not g._legal_place(recapture)
    assert not g.legal_mask()[recapture]


def test_two_passes_end_the_game():
    """Two consecutive passes end the game, and White wins the empty board by komi."""
    g = GoGame()
    g.apply(PASS)
    assert not g.done
    g.apply(PASS)
    assert g.done
    assert g.end_reason == "two_passes"
    assert g.winner == WHITE  # komi 7.5 on the empty board


def test_illegal_moves_are_masked():
    """legal_mask hides occupied, suicide and ko points but always keeps pass."""
    g = GoGame()
    g.set_board(board_of([
        ".OX......",
        "O.OX.....",
        ".OX......",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
    ]), BLACK)

    mask = g.legal_mask()
    assert mask[PASS]
    assert not mask[0 * 9 + 1]  # occupied by White
    assert mask[0 * 9 + 3]  # an ordinary empty point
    assert not mask[0 * 9 + 0]  # suicide (surrounded by White)

    # After the ko capture, the recapture point is masked too.
    g.apply(1 * 9 + 1)
    mask = g.legal_mask()
    assert not mask[1 * 9 + 2]  # simple ko
    assert mask[PASS]


def test_sampling_respects_the_legal_mask():
    """Sampling never returns an illegal move, even when its logit dominates."""
    g = GoGame()
    g.set_board(board_of([
        ".........",
        ".........",
        ".........",
        ".........",
        "....X....",
        ".........",
        ".........",
        ".........",
        ".........",
    ]), BLACK)
    center = 4 * 9 + 4
    legal = g.legal_mask()
    assert not legal[center]  # occupied by Black
    logits = np.zeros(NUM_OUTPUTS, dtype=np.float64)
    logits[center] = 1e6
    rng = np.random.RandomState(0)
    for _ in range(200):
        mv = sample_move(logits, legal, rng)
        assert legal[mv]
        assert mv != center
