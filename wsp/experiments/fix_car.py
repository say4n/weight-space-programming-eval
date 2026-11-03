"""Experiment 4: fix a failing car policy by editing its weights in one reply.

The network is the author's Kinetix actor (embed: 679 -> 128 ReLU, then five
layers of 128 tanh, then a 128 -> 16 logit head; 171,664 actor weights) that
fails the car-on-ramp level. The model is shown the full weight listing and
asked to explain the failure, then write an `edits` block; the harness applies
those edits, in order, to the rounded weights the model was shown.

This module reproduces the prompt/reply/edit-application half of the experiment
and reports how many weights the edit touches. It does NOT roll the result out
in the Kinetix physics environment: Kinetix is pip-installable as `kinetix-env`
but is not installed here, and this repository ships no physics/rollout code, so
`scored` is always recorded as `false` here (see `kinetix_status`).
"""

import ast
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from ..edits import last_edits_block, logical_lines
from ..runtime import make_runtime

REPO = Path(__file__).resolve().parent.parent.parent
UP = REPO / "upstream" / "exp4_fix_car"

TENSOR_SHAPES = {
    "embed.W": (679, 128),
    "embed.b": (128,),
    "actor0.W": (128, 128),
    "actor0.b": (128,),
    "actor1.W": (128, 128),
    "actor1.b": (128,),
    "actor2.W": (128, 128),
    "actor2.b": (128,),
    "actor3.W": (128, 128),
    "actor3.b": (128,),
    "actor4.W": (128, 128),
    "actor4.b": (128,),
    "actor_head.W": (128, 16),
    "actor_head.b": (16,),
}

# Kinetix parameter names in ppo_final_seed0.npz -> the prompt's tensor names.
NPZ_ACTOR = {
    "params/Dense_0/kernel": "embed.W",
    "params/Dense_0/bias": "embed.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/kernel": "actor0.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_0/Dense_0/bias": "actor0.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/kernel": "actor1.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_1/Dense_0/bias": "actor1.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/kernel": "actor2.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_2/Dense_0/bias": "actor2.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/kernel": "actor3.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_3/Dense_0/bias": "actor3.b",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/kernel": "actor4.W",
    "params/GeneralActorCriticRNN_0/DenseAndActivation_4/Dense_0/bias": "actor4.b",
    "params/GeneralActorCriticRNN_0/Dense_0/kernel": "actor_head.W",
    "params/GeneralActorCriticRNN_0/Dense_0/bias": "actor_head.b",
}

OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b}

CASES = ["oneshot_repair"]

TOTAL_WEIGHTS = 171_664


def blank():
    return {name: np.zeros(shape, dtype="float64") for name, shape in TENSOR_SHAPES.items()}


def _index(node):
    """Translate a subscript into numpy index syntax, rejecting anything else."""
    if isinstance(node, ast.Tuple):
        return tuple(_index(e) for e in node.elts)
    if isinstance(node, ast.Slice):
        parts = []
        for p in (node.lower, node.upper, node.step):
            if p is None:
                parts.append(None)
            elif isinstance(p, ast.Constant) and isinstance(p.value, int):
                parts.append(p.value)
            else:
                raise ValueError("slice bounds must be integer literals")
        return slice(*parts)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    raise ValueError("index must be an integer literal or a slice")


def _value(node):
    """Only literal numbers and (nested) lists of them."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"value is not a literal: {e}") from e


def _tensor_name(node):
    """Dotted tensor name from a target expression (Name or Attribute)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    raise ValueError("target must be a tensor name")


def apply_line(tensors, line):
    """Apply one `edits` statement. Returns True if it was applied."""
    text = line.split("#", 1)[0].strip()
    if not text:
        return False
    if text.upper().startswith("ZERO "):
        name = text[5:].strip()
        if name in TENSOR_SHAPES:
            tensors[name][...] = 0
            return True
        return False
    try:
        tree = ast.parse(text, mode="exec")
        if len(tree.body) != 1:
            return False
        stmt = tree.body[0]
        if isinstance(stmt, ast.Assign):
            op = None
            targets = stmt.targets
        elif isinstance(stmt, ast.AugAssign):
            op = type(stmt.op)
            if op not in OPS:
                return False
            targets = [stmt.target]
        else:
            return False
        if len(targets) != 1 or not isinstance(targets[0], ast.Subscript):
            return False
        target = targets[0]
        try:
            tensor_name = _tensor_name(target.value)
        except ValueError:
            return False
        if tensor_name not in TENSOR_SHAPES:
            return False

        array = tensors[tensor_name]
        where = _index(target.slice)
        new = _value(stmt.value)
        if op is None:
            array[where] = new
        else:
            array[where] = OPS[op](array[where], new)
        return True
    except (ValueError, SyntaxError, IndexError, TypeError, KeyError, OverflowError):
        return False


def parse_edits(block, base=None):
    """Apply an `edits` block on top of `base` (blank zeros by default).

    Returns (tensors, n_applied, n_skipped). The real experiment applies edits
    to the rounded weights shown in the prompt, so callers pass `base=shown`.
    """
    tensors = blank() if base is None else {name: base[name].copy() for name in TENSOR_SHAPES}
    applied = skipped = 0
    for statement in logical_lines(block or ""):
        if not statement.strip():
            continue
        if apply_line(tensors, statement):
            applied += 1
        else:
            skipped += 1
    return tensors, applied, skipped


def load_full_actor(path):
    """Load the full-precision actor from the published Kinetix npz."""
    d = np.load(path)
    tensors = {}
    for npz_name, tensor_name in NPZ_ACTOR.items():
        if npz_name not in d:
            raise KeyError(f"{path} is missing {npz_name!r}")
        tensors[tensor_name] = d[npz_name].astype("float64")
    if set(tensors) != set(TENSOR_SHAPES):
        raise ValueError(f"{path} does not contain exactly the actor tensors")
    return tensors


_HEADER = re.compile(r"^(\S+) \((\d+)(?: x (\d+))?\), scale ([^:]+):(.*)$")


def load_shown(prompt_path):
    """Reconstruct the rounded weights the model was shown, from prompt.txt.

    Each tensor is written as integers times one scale; weight tensors have one
    line per row (`i: a b c ...`) and bias tensors are one line of integers.
    """
    text = Path(prompt_path).read_text()
    idx = text.index("THE WEIGHTS (complete actor)")
    lines = text[idx:].splitlines()
    tensors = {}
    i = 0
    while i < len(lines):
        m = _HEADER.match(lines[i].strip())
        if not m:
            i += 1
            continue
        name = m.group(1)
        shape = (int(m.group(2)), int(m.group(3))) if m.group(3) else (int(m.group(2)),)
        if name not in TENSOR_SHAPES or shape != TENSOR_SHAPES[name]:
            raise ValueError(f"unexpected weight header: {lines[i].strip()!r}")
        scale = float(m.group(4).strip())
        per_row = "one line per row" in m.group(5)
        i += 1
        vals = []
        while len(vals) < int(np.prod(shape)):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            if per_row and ":" in line:
                line = line.split(":", 1)[1]
            vals.extend(int(t) for t in line.split())
        tensors[name] = np.asarray(vals, dtype="float64").reshape(shape) * scale
    missing = set(TENSOR_SHAPES) - set(tensors)
    if missing:
        raise ValueError(f"prompt weight section is missing: {sorted(missing)}")
    return tensors


def diff_counts(edited, base):
    """How many entries differ between two dicts of actor tensors."""
    total = changed = 0
    per_tensor = {}
    for name in TENSOR_SHAPES:
        n = int(edited[name].size)
        c = int(np.count_nonzero(edited[name] != base[name]))
        per_tensor[name] = {"changed": c, "entries": n}
        total += n
        changed += c
    return {
        "changed": changed,
        "entries": total,
        "frac_changed": (changed / total) if total else 0.0,
        "per_tensor": per_tensor,
    }


def kinetix_status():
    """Honest scoring status: this runner never executes the physics env."""
    importable = (
        importlib.util.find_spec("kinetix_env") is not None
        or importlib.util.find_spec("kinetix") is not None
    )
    return {
        "scored": False,
        "kinetix_pypi": "kinetix-env (https://github.com/FLAIROx/Kinetix)",
        "kinetix_importable_here": importable,
        "scored_reason": (
            "Kinetix is pip-installable as 'kinetix-env' but is not installed in this "
            "repository, and the repo ships no physics/rollout code, so this runner does "
            "not execute the environment. The 0% -> 100% success claim is therefore not "
            "re-scored here; upstream's eval/oneshot_result.json numbers are cited, not "
            "independently verified."
        ),
    }


def run(args):
    out_root = REPO / "results" / "exp4" / (args.tag or args.model.replace("/", "_"))
    out_root.mkdir(parents=True, exist_ok=True)
    recs = []
    for case in CASES:
        run_dir = out_root / case
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_src = UP / "runs" / case / "reply" / "prompt.txt"
        answer_path = run_dir / "answer.md"
        prompt = prompt_src.read_text()

        rec = {"case": case, "model": args.model, "tag": args.tag or None}
        rec.update(kinetix_status())

        if args.rescore:
            answer = answer_path.read_text()
        else:
            (run_dir / "prompt.txt").write_text(prompt)
            with make_runtime(
                args.model, args.provider, args.thinking, args.base_url, cwd=run_dir
            ) as rt:
                reply = rt.send(prompt)
            answer = reply.text
            answer_path.write_text(answer)
            usage = {
                "usage": reply.usage,
                "wall_s": reply.wall_s,
                "tool_calls": reply.tool_calls,
                "tool_free": reply.tool_free,
            }
            (run_dir / "usage.json").write_text(json.dumps(usage, indent=1))
            rec.update(usage)

        block = last_edits_block(answer)
        if block is None:
            rec.update({"edits_block": False, "error": "no edits block in the reply"})
            print(f"  {case}: NO EDITS BLOCK", flush=True)
            recs.append(rec)
            continue

        (run_dir / "edits.txt").write_text(block)
        full = load_full_actor(UP / "networks" / "ppo_final_seed0.npz")
        shown = load_shown(prompt_src)
        tensors, applied, skipped = parse_edits(block, shown)
        np.savez(str(run_dir / "edited.npz"), allow_pickle=False, **tensors)

        vs_shown = diff_counts(tensors, shown)
        vs_full = diff_counts(tensors, full)

        rec.update(
            {
                "edits_block": True,
                "statements_applied": applied,
                "statements_skipped": skipped,
                "weights_total": vs_shown["entries"],
                "changed_vs_shown_network": vs_shown["changed"],
                "frac_changed_vs_shown_network": vs_shown["frac_changed"],
                "changed_vs_full_precision_npz": vs_full["changed"],
                "frac_changed_vs_full_precision_npz": vs_full["frac_changed"],
                "per_tensor_changed_vs_shown": vs_shown["per_tensor"],
            }
        )
        print(
            f"  {case}: {applied} applied, {skipped} skipped; "
            f"{vs_shown['changed']}/{vs_shown['entries']} differ from the shown "
            f"(rounded) network",
            flush=True,
        )
        recs.append(rec)

    scores_path = out_root / "scores.json"
    scores_path.write_text(json.dumps(recs, indent=1))
    print(f"\nexp4 -> {scores_path}")
    return recs
