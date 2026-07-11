#!/usr/bin/env bash
# Install dependencies for WebShop + ALFWorld text environments.
# Run from the cos-play repo root:
#   bash scripts/install_text_envs.sh
set -euo pipefail

echo "=== Installing text environment dependencies ==="

pip install alfworld gym beautifulsoup4 flask lxml selenium thefuzz rank_bm25 cleantext
python3 -m spacy download en_core_web_sm

echo "=== Downloading ALFWorld data ==="
alfworld-download

echo "=== Verifying WebShop data ==="
WEBSHOP_ROOT="${WEBSHOP_ROOT:-/workspace/WebShop}"
if [[ ! -d "$WEBSHOP_ROOT/data" ]]; then
    echo "WARNING: WebShop data not found at $WEBSHOP_ROOT/data"
    echo "Clone https://github.com/princeton-nlp/WebShop and place at $WEBSHOP_ROOT"
else
    echo "WebShop data found at $WEBSHOP_ROOT/data"
fi

echo ""
echo "=== Quick smoke test ==="
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 -c "
from env_wrappers.alfworld_nl_wrapper import make_alfworld_env
env = make_alfworld_env(max_steps=5, split='eval_out_of_distribution')
obs, info = env.reset()
print('ALFWorld OK: goal =', info['goal'][:60])
env.close()

from env_wrappers.webshop_nl_wrapper import make_webshop_env
env = make_webshop_env(max_steps=5, num_products=100)
obs, info = env.reset()
print('WebShop OK:  goal =', info['goal'][:60])
env.close()

print('All text envs verified.')
"

echo "=== Done ==="
