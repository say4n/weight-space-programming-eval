"""Experiment 5: write networks from scratch, in one reply, with no tools.

The model is given a game's rules and source, the network shape, and how it is
scored, then writes every weight itself in an `edits` block. It never plays the
game before scoring. Each network is then scored on 1,000 episodes of the official
MinAtar 1.0.15 game against the published PPO baselines.
"""

import json
from pathlib import Path

from ..edits import last_edits_block, parse
from ..minatar import score
from ..runtime import make_runtime

REPO = Path(__file__).resolve().parent.parent.parent
UP = REPO / "upstream" / "exp5_write_from_scratch"

GAMES = ["breakout", "seaquest", "asterix"]
EPISODES = 1000


def evaluate(tensors, game, episodes):
    return score(tensors, game, episodes=episodes)


def run(args):
    try:
        published = json.loads((UP / "eval" / "minatar_eval.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read the published baselines: {e}") from e
    out_root = REPO / "results" / "exp5" / (args.tag or args.model.replace("/", "_"))
    games = args.cases or GAMES
    episodes = args.episodes or EPISODES

    print(f"exp5: {len(games)} games, {episodes} episodes each, model={args.model}")
    recs = []
    for game in games:
        run_dir = out_root / game
        run_dir.mkdir(parents=True, exist_ok=True)
        answer_path = run_dir / "answer.md"

        if args.rescore:
            answer = answer_path.read_text()
            applied = skipped = None
        else:
            prompt = (UP / "runs" / game / "message.txt").read_text()
            (run_dir / "prompt.txt").write_text(prompt)
            with make_runtime(
                args.model, args.provider, args.thinking, args.base_url, cwd=run_dir
            ) as rt:
                reply = rt.send(prompt)
            answer = reply.text
            answer_path.write_text(answer)
            (run_dir / "usage.json").write_text(
                json.dumps(
                    {
                        "usage": reply.usage,
                        "wall_s": reply.wall_s,
                        "tool_calls": reply.tool_calls,
                        "tool_free": reply.tool_free,
                    },
                    indent=1,
                )
            )

        block = last_edits_block(answer)
        if block is None:
            rec = {"game": game, "error": "no edits block in the reply"}
            recs.append(rec)
            print(f"  {game:10s} NO EDITS BLOCK")
            continue
        (run_dir / "edits.txt").write_text(block)
        tensors, applied, skipped = parse(block)

        result = evaluate(tensors, game, episodes)
        nonzero = int(sum((v != 0).sum() for v in tensors.values()))
        base = published[game]
        ppo500 = max(v["mean"] for k, v in base.items() if k.startswith("ppo500"))
        rec = {
            "game": game,
            "statements_applied": applied,
            "statements_skipped": skipped,
            "nonzero_weights": nonzero,
            "total_weights": int(sum(v.size for v in tensors.values())),
            "score": result,
            "episodes": episodes,
            "ppo100": base["ppo100_s2"]["mean"],
            "ppo500": ppo500,
            "astra_hand": base["hand"]["mean"],
        }
        recs.append(rec)
        print(
            f"  {game:10s} score={result['mean']:7.1f}  "
            f"(astra {rec['astra_hand']:6.1f}, ppo100 {rec['ppo100']:5.1f}, "
            f"ppo500 {rec['ppo500']:5.1f})  nnz={nonzero} "
            f"stmts={applied}+{skipped}skip",
            flush=True,
        )

    (out_root / "scores.json").write_text(json.dumps(recs, indent=1))
    print(f"\nexp5 -> {out_root / 'scores.json'}")
    return recs
