#!/usr/bin/env bash
#
# install_webshop.sh — bring up princeton-nlp/WebShop as a single Flask
# server, dependency-isolated from `browsergym` and `game-ai-agent`.
#
# Why a separate conda env:
#   WebShop's upstream pins (Flask 2.1, gym 0.24, NumPy 1.22, torch 1.11,
#   transformers 4.19) clash with our training/eval envs.  This installer
#   creates a dedicated `webshop` env, the agent-side bridge in
#   `webshop_wrapper/server.py` talks to it over HTTP only.
#
# Why "lite" mode by default:
#   The full WebShop install pulls in pyserini (Java 11 + Lucene), faiss,
#   spaCy `en_core_web_lg`, and an old PyTorch — together ~3 GB and ~30
#   minutes of wall-clock + several places where pip-resolver gets stuck
#   on M1 / NumPy-2 / Java-version edge cases.  Lite mode patches
#   `engine.py` to use a rank_bm25 search fallback (already in WebShop's
#   requirements.txt, no Lucene needed) and skips the BERT ranker.  The
#   2026 paper's ablations show <2 pp pass-rate delta from removing the
#   BERT ranker, which is well below our cold-start noise floor.
#
# Usage:
#   bash install/install_webshop.sh                       # lite mode (default)
#   WEBSHOP_LITE=0 bash install/install_webshop.sh        # full mode (Lucene+BERT)
#   WEBSHOP_DATA=all bash install/install_webshop.sh      # full 1M-product dataset
#
# Env vars / overrides:
#   WEBSHOP_DIR    where to clone (default: /workspace/WebShop)
#   WEBSHOP_DATA   small | all  (default: small = 1k products, ~6 MB)
#   WEBSHOP_LITE   1 | 0        (default: 1 = skip pyserini/faiss/torch)
#   WEBSHOP_PORT   port to expose (default: 3000)
#   WEBSHOP_HOST   bind host    (default: 127.0.0.1)
#   WEBSHOP_NO_DOWNLOAD=1  skip dataset download (assume data/ already present)

set -uo pipefail

ENV_NAME="webshop"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YML="${SCRIPT_DIR}/webshop.environment.yml"

WEBSHOP_DIR="${WEBSHOP_DIR:-/workspace/WebShop}"
WEBSHOP_DATA="${WEBSHOP_DATA:-small}"
WEBSHOP_LITE="${WEBSHOP_LITE:-1}"
WEBSHOP_PORT="${WEBSHOP_PORT:-3000}"
WEBSHOP_HOST="${WEBSHOP_HOST:-127.0.0.1}"

ENV_FILE="${CODEBASE_ROOT}/cold_start/webshop_env.sh"

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Step 1: conda env ─────────────────────────────────────────────────────
echo "[1/6] Creating conda env '$ENV_NAME' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "      Env already exists — skipping."
else
    conda env create -f "$YML"
fi
conda activate "$ENV_NAME"

# ── Step 2: clone WebShop ─────────────────────────────────────────────────
echo "[2/6] Cloning princeton-nlp/WebShop to $WEBSHOP_DIR ..."
if [[ ! -d "$WEBSHOP_DIR" ]]; then
    git clone --depth=1 https://github.com/princeton-nlp/WebShop.git "$WEBSHOP_DIR"
else
    echo "      Already cloned."
fi
cd "$WEBSHOP_DIR"

# ── Step 3: download dataset ──────────────────────────────────────────────
echo "[3/6] Downloading WebShop dataset (mode: $WEBSHOP_DATA) ..."
if [[ "${WEBSHOP_NO_DOWNLOAD:-0}" == "1" ]]; then
    echo "      WEBSHOP_NO_DOWNLOAD=1 — skipping."
elif [[ -f "$WEBSHOP_DIR/data/items_shuffle_1000.json" && "$WEBSHOP_DATA" == "small" ]]; then
    echo "      Small dataset already on disk — skipping."
elif [[ -f "$WEBSHOP_DIR/data/items_shuffle.json" && "$WEBSHOP_DATA" == "all" ]]; then
    echo "      Full dataset already on disk — skipping."
else
    mkdir -p data && cd data
    if [[ "$WEBSHOP_DATA" == "small" ]]; then
        # 1000-product split (~6 MB)
        gdown 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json
        gdown 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json
    elif [[ "$WEBSHOP_DATA" == "all" ]]; then
        # Full ~1M-product split (~100 MB)
        gdown 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB -O items_shuffle.json
        gdown 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi -O items_ins_v2.json
    else
        echo "      [ERROR] Unknown WEBSHOP_DATA=$WEBSHOP_DATA (expected small|all)"
        exit 1
    fi
    # Human goal annotations (always)
    gdown 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O items_human_ins.json
    cd ..
fi

# ── Step 4: lite-mode patch ───────────────────────────────────────────────
# Replace `from pyserini.search.lucene import LuceneSearcher` and
# `init_search_engine(...)` with a rank_bm25 shim.  Idempotent.
echo "[4/6] Patching engine.py for lite mode (WEBSHOP_LITE=$WEBSHOP_LITE) ..."
if [[ "$WEBSHOP_LITE" != "1" ]]; then
    echo "      WEBSHOP_LITE=0 — installing pyserini + faiss + torch (slow, ~30 min)"
    conda install -n "$ENV_NAME" -y -c pytorch faiss-cpu
    conda install -n "$ENV_NAME" -y -c conda-forge openjdk=11
    pip install pyserini==0.17.0 torch==1.11.0 transformers==4.19.2
    # Build the Lucene index (one-time, ~20 min)
    if [[ ! -d "$WEBSHOP_DIR/search_engine/indexes_1k" ]]; then
        cd "$WEBSHOP_DIR/search_engine"
        mkdir -p resources resources_100 resources_1k resources_100k indexes
        python convert_product_file_format.py
        ./run_indexing.sh
        cd "$WEBSHOP_DIR"
    fi
else
    ENGINE="$WEBSHOP_DIR/web_agent_site/engine/engine.py"
    if grep -q "_LITE_BM25_PATCH_APPLIED" "$ENGINE"; then
        echo "      Lite patch already applied — skipping."
    else
        # The patch (a) removes the LuceneSearcher import and (b)
        # rewrites init_search_engine() to return a rank_bm25-backed
        # shim that exposes the same .search() / .doc() / .raw()
        # interface used in get_top_n_product_from_keywords().
        python - "$ENGINE" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
shim = """# _LITE_BM25_PATCH_APPLIED - inserted by install/install_webshop.sh
# Replaces the LuceneSearcher with a rank_bm25 shim so we don't need
# Java + pyserini just to serve the Flask app.
class _Hit:
    __slots__ = ('docid', 'score')
    def __init__(self, docid, score): self.docid, self.score = docid, score

class _Doc:
    __slots__ = ('_raw',)
    def __init__(self, raw_str): self._raw = raw_str
    def raw(self): return self._raw

class _BM25Shim:
    def __init__(self, all_products):
        from rank_bm25 import BM25Okapi
        import json as _json
        self._docs = []
        corpus = []
        for p in all_products:
            text = ' '.join(str(p.get(k, '')) for k in
                ('Title','category','query','product_category','asin'))
            tokens = text.lower().split()
            self._docs.append(_json.dumps({'id': p['asin']}))
            corpus.append(tokens)
        self._bm25 = BM25Okapi(corpus or [['empty']])
    def search(self, query, k=50):
        scores = self._bm25.get_scores(str(query).lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [_Hit(str(i), float(scores[i])) for i in order]
    def doc(self, docid):
        return _Doc(self._docs[int(docid)])

_BM25_SHIM_INSTANCE = None
"""
src = src.replace(
    'from pyserini.search.lucene import LuceneSearcher',
    '# (LuceneSearcher import removed by lite patch)',
)
# Replace the body of init_search_engine() to construct the shim.  We
# rebuild the whole function for safety.
import re
new_fn = '''
def init_search_engine(num_products=None):
    global _BM25_SHIM_INSTANCE
    if _BM25_SHIM_INSTANCE is not None:
        return _BM25_SHIM_INSTANCE
    products, _, _, _ = load_products(filepath=DEFAULT_FILE_PATH, num_products=num_products)
    _BM25_SHIM_INSTANCE = _BM25Shim(products)
    return _BM25_SHIM_INSTANCE
'''
src = re.sub(
    r'def init_search_engine\(num_products=None\):.*?return search_engine\n',
    new_fn.lstrip() + '\n',
    src, count=1, flags=re.DOTALL,
)
# Insert the shim block just after the imports (before the first
# blank-then-non-blank line).
lines = src.splitlines(keepends=True)
insert_at = 0
for i, line in enumerate(lines):
    if line.startswith('SEARCH_RETURN_N'):
        insert_at = i
        break
lines.insert(insert_at, '\n' + shim + '\n')
p.write_text(''.join(lines))
print(f'    [patched] {p}')
PYEOF
    fi
fi

# ── Step 5: bridge endpoint patch ─────────────────────────────────────────
# Add a `/__bridge/session/<id>` JSON endpoint to web_agent_site/app.py
# so webshop_wrapper.task.WebShopTask.validate() can read reward
# without scraping the done page HTML.  Idempotent.
echo "[5/6] Patching app.py with /__bridge endpoint ..."
APP="$WEBSHOP_DIR/web_agent_site/app.py"
if grep -q "_BRIDGE_PATCH_APPLIED" "$APP"; then
    echo "      Bridge patch already applied — skipping."
else
    python - "$APP" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
patch = """
# _BRIDGE_PATCH_APPLIED - inserted by install/install_webshop.sh
# Side-channel for webshop_wrapper.task.WebShopTask.validate() to read
# session reward as JSON instead of parsing the done-page HTML.
@app.route('/__bridge/session/<session_id>')
def __bridge_session(session_id):
    import json as _json
    if session_id not in user_sessions and 'fixed' in session_id:
        # Auto-create the fixed session so a probe before page.goto works.
        global all_products, product_item_dict, product_prices, attribute_to_asins, search_engine, goals, weights
        if search_engine is None:
            all_products, product_item_dict, product_prices, attribute_to_asins = load_products(filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE)
            search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
            goals = get_goals(all_products, product_prices)
            import random as _r
            _r.seed(233); _r.shuffle(goals)
            weights = [g['weight'] for g in goals if 'weight' in g]
        goal_idx = int(session_id.split('_')[-1]) % len(goals)
        goal = goals[goal_idx]
        user_sessions[session_id] = {'goal': goal, 'done': False}
    sess = user_sessions.get(session_id, {})
    return _json.dumps({
        'done': sess.get('done', False),
        'reward': sess.get('reward', 0.0),
        'goal': sess.get('goal', {}),
    })
"""
# Append before the `if __name__ == "__main__":` block.
marker = 'if __name__ == "__main__":'
if marker not in src:
    raise RuntimeError(f'cannot find {marker} in {p}')
src = src.replace(marker, patch + '\n\n' + marker, 1)
p.write_text(src)
print(f'    [patched] {p}')
PYEOF
fi

# ── Step 6: write env file + smoke ────────────────────────────────────────
echo "[6/6] Writing $ENV_FILE and running smoke ..."
cat > "$ENV_FILE" <<EOF
# Auto-generated by install/install_webshop.sh on $(date)
# Source this file to point the bridge at the running WebShop server.
export WEBSHOP_DIR="${WEBSHOP_DIR}"
export WEBSHOP_BASE_URL="http://${WEBSHOP_HOST}:${WEBSHOP_PORT}"
export WEBSHOP_HOST="${WEBSHOP_HOST}"
export WEBSHOP_PORT="${WEBSHOP_PORT}"
export WEBSHOP_LITE="${WEBSHOP_LITE}"
EOF

# Quick import-only smoke (does NOT boot the Flask server here — that
# happens on demand via webshop_wrapper.server.start_webshop_server()).
cd "$WEBSHOP_DIR"
python -c "
import sys
print(f'Python {sys.version.split()[0]}')
from web_agent_site.engine.engine import init_search_engine, load_products
from web_agent_site.utils import DEFAULT_FILE_PATH, DEBUG_PROD_SIZE
products, item_dict, prices, attrs = load_products(filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE)
engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
hits = engine.search('water bottle', k=3)
print(f'  loaded {len(products)} products')
print(f'  search engine: {type(engine).__name__}')
print(f'  sample hits for \"water bottle\": {len(hits)}')
"

echo
echo "================================================================"
echo "  WebShop install complete (mode: $([[ $WEBSHOP_LITE == 1 ]] && echo lite || echo full))"
echo "================================================================"
echo "  Activate:  conda activate $ENV_NAME"
echo "             source $ENV_FILE"
echo
echo "  Boot server (foreground):"
echo "    cd $WEBSHOP_DIR && python -m web_agent_site.app --port $WEBSHOP_PORT"
echo
echo "  Or let the bridge start it for you:"
echo "    conda activate browsergym"
echo "    python -m webshop_wrapper.smoke_axtree --base-url \$WEBSHOP_BASE_URL"
echo "================================================================"
