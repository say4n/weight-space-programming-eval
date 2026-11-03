#!/usr/bin/env bash
# Isolated env for the Kinetix physics scorer used by exp 4.
#
# Kinetix needs jax, which the main env deliberately does not carry, so scoring
# shells out to this interpreter. `wsp exp4` finds it automatically.
set -euo pipefail

cd "$(dirname "$0")/.."

uv venv .venv-kinetix --python 3.12
uv pip install --python .venv-kinetix/bin/python kinetix-env jumanji jax numpy
uv pip install --python .venv-kinetix/bin/python pytest

echo
echo "ok. score any actor with:"
echo "  .venv-kinetix/bin/python -m wsp.score_car <actor.npz> --episodes 1024"
