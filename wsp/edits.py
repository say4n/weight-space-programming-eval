"""Parser for the `edits` mini-language the network-writing prompts ask for.

The model may only set numbers; it never gets to run anything. So this accepts a
deliberately tiny grammar and refuses everything else:

    W1[12, 0] = 2.5
    W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
    b1[0:4] = -1
    W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
    W3[:, 2] *= 2

Statements apply in order and a later statement overwrites an earlier one. A line
that cannot be read is skipped; the rest still apply, exactly as the prompt promises.
"""

import ast

TENSORS = {
    "W1": (600, 32),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (32, 6),
    "b3": (6,),
}
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b}


def blank():
    import numpy as np

    return {name: np.zeros(shape, dtype="float64") for name, shape in TENSORS.items()}


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


def apply_line(tensors, line):
    """Apply one statement. Returns True if it was applied."""
    text = line.split("#", 1)[0].strip()
    if not text:
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
        if not isinstance(target.value, ast.Name) or target.value.id not in TENSORS:
            return False

        array = tensors[target.value.id]
        where = _index(target.slice)
        new = _value(stmt.value)
        if op is None:
            array[where] = new
        else:
            array[where] = OPS[op](array[where], new)
        return True
    except (ValueError, SyntaxError, IndexError, TypeError, KeyError):
        return False


def logical_lines(block):
    """Join physical lines into statements.

    The prompt allows a long list to continue over several lines, so a statement
    ends only when its brackets balance. Comments are stripped first; the grammar
    never contains strings, so counting brackets in the remaining text is safe.
    """
    buf, depth = [], 0
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() and not buf:
            continue
        buf.append(line)
        depth += line.count("[") + line.count("(")
        depth -= line.count("]") + line.count(")")
        if depth <= 0:
            yield "\n".join(buf)
            buf, depth = [], 0
    if buf:
        yield "\n".join(buf)


def parse(block):
    """Apply a whole edits block. Returns (tensors, n_applied, n_skipped)."""
    tensors = blank()
    applied = skipped = 0
    for statement in logical_lines(block or ""):
        if not statement.strip():
            continue
        if apply_line(tensors, statement):
            applied += 1
        else:
            skipped += 1
    return tensors, applied, skipped


def last_edits_block(text):
    """The reply's last ```edits block; only that one is read."""
    marker = "```edits"
    start = text.rfind(marker)
    if start < 0:
        return None
    body = text[start + len(marker) :]
    end = body.find("```")
    return body if end < 0 else body[:end]
