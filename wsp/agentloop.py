"""Drive a conversation in which the model writes Python and we run it.

The original harness ran one fenced block per turn in an isolated sandbox and sent
the printed output back as the next message. This is model-agnostic: it never asks
the model for a structured tool call, only for a code block, so any model that can
emit fenced code can be compared on the same footing.
"""

import json
import re
from dataclasses import dataclass, field

# ```python, ```py, or a bare fence; the body is captured non-greedily.
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)

FEEDBACK = "OUTPUT of run {n} of at most {mx}:\n```\n{out}\n```"


def last_code_block(text):
    """The block the model wants run. Only the last one counts, as in the original."""
    blocks = CODE_BLOCK.findall(text or "")
    return blocks[-1] if blocks else None


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
