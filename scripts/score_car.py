# /// script
# requires-python = ">=3.12"
# dependencies = ["kinetix-env", "jumanji", "jax", "numpy"]
# ///
"""Score an actor npz on the exp4 Kinetix car-ramp protocol.

uv builds and caches a detached environment for the dependencies above, so the main
project env never needs jax:

    uv run scripts/score_car.py <actor.npz> --episodes 1024
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsp.score_car import main  # noqa: E402  (needs the path insert above)

if __name__ == "__main__":
    raise SystemExit(main())
