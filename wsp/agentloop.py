"""Drive a conversation in which the model writes Python and we run it.

The original harness ran one fenced block per turn in an isolated sandbox and sent
the printed output back as the next message. This is model-agnostic: it accepts a
fenced ```python block or the model's native tool-call syntax, so a model that ignores
the fence instruction is still compared on the same footing.
"""

import json
import re
from dataclasses import dataclass, field

# ```python, ```py, or a bare fence; the body is captured non-greedily.
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)

# Models that ignore the "fenced block" instruction often emit their native tool-call
# syntax instead (DeepSeek uses <｜｜DSML｜｜ ...>). A model-agnostic harness must accept
# that, or it silently discards real code as if it were a final answer.
DSML_PY = re.compile(
    r'invoke\s+name="python"[^>]*>(.*?)</\uff5c\uff5cDSML\uff5c\uff5c\s*invoke>', re.S)
DSML_PARAM = re.compile(
    r'parameter\s+name="(?:input|code)"[^>]*>(.*?)'
    r'</\uff5c\uff5cDSML\uff5c\uff5c\s*parameter>', re.S)
DSML_ANY_INVOKE = re.compile(r'invoke\s+name="([a-zA-Z_]+)"', re.S)

FEEDBACK = "OUTPUT of run {n} of at most {mx}:\n```\n{out}\n```"


def last_code_block(text):
    """The block the model wants run. Only the last one counts, as in the original."""
    blocks = CODE_BLOCK.findall(text or "")
    if blocks:
        return blocks[-1]
    return _dsml_python(text)


def _dsml_python(text):
    """Code from a native tool call, e.g. DeepSeek's <｜｜DSML｜｜ invoke name="python">."""
    invokes = DSML_PY.findall(text or "")
    if not invokes:
        return None
    params = DSML_PARAM.findall(invokes[-1])
    return params[-1] if params else None


def named_tool_invokes(text):
    """Tools the model tried to call natively; used to explain a mis-formatted turn."""
    return sorted(set(DSML_ANY_INVOKE.findall(text or "")))


@dataclass
class Turn:
    n: int
    code: str
    output: str


@dataclass
class Result:
    final: str
    runs: int
    turns: list = field(default_factory=list)
    truncated: bool = False  # hit max_runs while still writing code
    error: str | None = None


class CodeLoop:
    def __init__(self, runtime, sandbox, prelude="", max_runs=100, max_chars=20000):
        self.runtime = runtime
        self.sandbox = sandbox
        self.prelude = prelude
        self.max_runs = max_runs
        self.max_chars = max_chars
        self.turns = []

    def run(self, message):
        reply = self.runtime.send(message)
        runs = 0
        while True:
            if reply.text.startswith("[harness error"):
                return Result(reply.text, runs, self.turns, error=reply.text)
            code = last_code_block(reply.text)
            if code is None:
                return Result(reply.text, runs, self.turns)

            if runs >= self.max_runs:
                return Result(reply.text, runs, self.turns, truncated=True)

            runs += 1
            out = self.sandbox.run(code, self.prelude)[: self.max_chars]
            self.turns.append(Turn(runs, code, out))
            reply = self.runtime.send(FEEDBACK.format(n=runs, mx=self.max_runs, out=out))

    def save(self, path):
        try:
            with open(path, "w") as f:
                for t in self.turns:
                    f.write(json.dumps({"n": t.n, "code": t.code, "out": t.output}) + "\n")
        except OSError as e:
            raise RuntimeError(f"cannot save the code loop transcript to {path}: {e}") from e
