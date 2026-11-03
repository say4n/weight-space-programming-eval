"""Tests for the code-writing loop's block extraction.

The regression this guards: DeepSeek emits its native tool-call syntax instead of a
fenced block. Before the fix, last_code_block() returned None, so the loop treated
real code as a final answer and the experiment silently scored 0.
"""

from wsp.agentloop import last_code_block, named_tool_invokes

FENCED = "thinking...\n```python\nprint(1)\n```\nmore text"
DSML_PYTHON = (
    '<｜｜DSML｜｜ calls>\n'
    '<｜｜DSML｜｜ invoke name="python">\n'
    '<｜｜DSML｜｜ parameter name="input" string="true">import numpy as np\n'
    'print(np.zeros(3))\n'
    '</｜｜DSML｜｜ parameter>\n'
    '</｜｜DSML｜｜ invoke>\n'
    '</｜｜DSML｜｜ calls>'
)
DSML_BASH_ONLY = (
    '<｜｜DSML｜｜ calls>\n'
    '<｜｜DSML｜｜ invoke name="bash">\n'
    '<｜｜DSML｜｜ parameter name="command" string="true">ls -la</｜｜DSML｜｜ parameter>\n'
    '</｜｜DSML｜｜ invoke>\n'
    '</｜｜DSML｜｜ calls>'
)


def test_fenced_block():
    code = last_code_block(FENCED)
    assert code is not None
    assert code.strip() == "print(1)"


def test_native_tool_call_is_code():
    code = last_code_block(DSML_PYTHON)
    assert code is not None, "native tool call must be treated as code, not a final answer"
    assert "np.zeros(3)" in code


def test_last_of_several_is_used():
    two = "```python\na = 1\n```\n```python\na = 2\n```"
    code = last_code_block(two)
    assert code is not None
    assert code.strip() == "a = 2"


def test_bash_only_is_not_python():
    # we only execute python; a bash invoke must not be run as python
    assert last_code_block(DSML_BASH_ONLY) is None
    assert "bash" in named_tool_invokes(DSML_BASH_ONLY)


def test_plain_prose_has_no_block():
    assert last_code_block("The two checkpoints differ mainly in layer 12.") is None
    assert named_tool_invokes("no tools here") == []
