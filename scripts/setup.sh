#!/usr/bin/env bash
# Fetch the third-party inputs this reproduction reads but does not vendor.
#
#   ./scripts/setup.sh            reference repo only
#   ./scripts/setup.sh --models   also download the models (several GB)
set -euo pipefail

cd "$(dirname "$0")/.."

# upstream/ is a submodule pinned to the exact commit whose bytes the results were
# produced from. Never float it to a newer main: the numbers depend on these bytes.
if git submodule status upstream >/dev/null 2>&1; then
  git submodule update --init --recursive
  echo "ok: reference checkout $(git -C upstream rev-parse --short HEAD)"
else
  echo "upstream/ is not a registered submodule; is this a git clone?" >&2
  exit 1
fi

if [ "${1:-}" = "--models" ]; then
  echo "downloading models (exp 7 uses a substitute base; see README)"
  uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"):
    print("->", repo)
    snapshot_download(repo, local_dir=f"models/{repo.split('/')[-1]}")
PY
fi
