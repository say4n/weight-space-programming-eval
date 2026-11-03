"""Self-check for the answer scorer. Run: uv run pytest wsp -q"""

from wsp.scoring import parse_answer, score

# The two real bugs: substring matching turned every answer into a circle,
# and 'circle' did not match 'circular'.
CASES = [
    ("A single white digit 2 on a black background.", None),
    ("A solid white filled circle/disk on a black background.", "circle"),
    ("A single white, roughly circular filled disk", "circle"),
    ("white rectangular outlines on black", "square"),
    ("a hollow upward-pointing triangle", "triangle"),
    ("Frogger, a simplified grid-based traffic-crossing variant.", "freeway"),
    ("Pac-Man, or a simplified grid-based Pac-Man clone.", "pacman"),
    ("Asterix (MinAtar Asterix).", "asterix"),
    ("A small grid-based brick-breaking variant.", "breakout"),
]


def test_score_labels():
    for text, want in CASES:
        assert score(text) == want, (text, score(text), want)


def test_round_is_not_background():
    assert score("on a black background") is None
    assert score("rounded edges") == "circle"


def test_earliest_mention_wins():
    assert score("Not Pong. It is Breakout, clearly.") == "pong"


def test_parse_answer():
    a = parse_answer("Observations: x\nGame: Pong.\nConfidence: 96%\n")
    assert a["game"] == "Pong."
    assert a["confidence"] == 96


def test_parse_answer_missing_confidence():
    assert parse_answer("Game: Pong.")["confidence"] is None
