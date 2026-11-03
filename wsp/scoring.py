"""Answer parsing and keyword grading, shared by the read-out experiments.

The keyword map mirrors the author's own scoring (Frogger counts as Freeway) and
is extended with the near-miss game names so a wrong answer is labelled rather
than silently dropped.
"""

import re

KEYWORDS = {
    "breakout": ["breakout", "brick"],
    "pong": ["pong"],
    "pacman": ["pac"],
    "freeway": ["freeway", "frogger", "road-crossing", "traffic"],
    "circle": ["circle", "circular", "ring", "disc", "disk", "round"],
    "square": ["square", "rectangle", "rectangular"],
    "triangle": ["triangle", "delta", "triangular"],
    # near misses: named so a wrong answer is recorded as such
    "asterix": ["asterix"],
    "seaquest": ["seaquest"],
    "space_invaders": ["space invader", "invaders"],
}


def parse_answer(text):
    """Pull the labelled fields out of the reply."""
    out = {}
    for field in ("Game", "Images", "Description", "Confidence"):
        m = re.search(rf"^\W*{field}\W*\s*(.+)$", text, re.M | re.I)
        if m:
            out[field.lower()] = m.group(1).strip()
    m = re.search(r"(\d+)", out.get("confidence", ""))
    out["confidence"] = int(m.group(1)) if m else None
    return out


def score(text):
    """Which game/shape does this reply name?

    Keywords match at a word boundary with any suffix, so 'pac' catches 'Pac-Man'
    while 'round' does not match 'background'. Earliest mention wins.
    """
    g = (text or "").lower()
    hits = []
    for key, words in KEYWORDS.items():
        pos = [m.start() for w in words for m in [re.search(rf"\b{re.escape(w)}", g)] if m]
        if pos:
            hits.append((key, min(pos)))
    if not hits:
        return None
    hits.sort(key=lambda kv: kv[1])
    return hits[0][0]
