"""Validate the exp3 parser against the author's published answers.

This is the strongest check available without spending model calls: the post's own
final answers (from Astra) must parse back to the exact scores the author recorded.
"""

import json
from pathlib import Path

from wsp.experiments.language import parse_final, truth_sets

UP = Path(__file__).resolve().parent.parent / "upstream" / "exp3_language"
RUN_IDS = {
    "python_L4_H6_d384_s0_r0": "python_r0",
    "python_L4_H6_d384_s0_r1": "python_r1",
    "romaji_L4_H6_d384_s0_r0": "romaji_r0",
    "romaji_L4_H6_d384_s0_r1": "romaji_r1",
    "english_L4_H6_d384_s0_r0": "english_r0",
    "english_L4_H6_d384_s0_r1": "english_r1",
}


def test_parser_reproduces_official_scores():
    official = json.loads((UP / "eval" / "scores.json").read_text())
    for o in official:
        run = RUN_IDS[o["id"]]
        parsed = parse_final((UP / "runs" / run / "final_answer.md").read_text())
        named = sorted(h for h, lab in parsed["heads"].items() if lab == "copy")
        assert parsed["corpus"] == o["corpus_guess"], run
        assert parsed["order"] == o["order_guess"], run
        assert named == sorted(o["copy_named"]), run
        assert len(parsed["heads"]) == 24, run


def test_truth_is_the_three_nets():
    heads = json.loads((UP / "eval" / "heads.json").read_text())
    truth = truth_sets(heads)
    assert set(truth) == {"python_L4_H6_d384_s0", "romaji_L4_H6_d384_s0", "english_L4_H6_d384_s0"}
    assert truth["python_L4_H6_d384_s0"] == {"L2H1", "L3H1", "L3H5"}


def test_prose_is_not_a_field():
    # The English runs open a paragraph with "Corpus identification is ..."
    text = "Corpus identification is weaker.\n\ncorpus: english\norder: as is\n"
    parsed = parse_final(text)
    assert parsed["corpus"] == "english"
    assert parsed["order"] == "as is"
