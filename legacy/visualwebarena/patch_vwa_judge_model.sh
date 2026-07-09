#!/usr/bin/env bash
#
# patch_vwa_judge_model.sh — patch upstream visualwebarena evaluator
#
# The upstream ``visualwebarena`` Python package hardcodes
# ``gpt-4-1106-preview`` (a 2024-Q1 deprecated OpenAI model that most
# current API keys no longer have access to) inside its
# ``llm_fuzzy_match`` and ``llm_ua_match`` helpers in
# ``evaluation_harness/helper_functions.py``. Those helpers are called
# by ``StringEvaluator.fuzzy_match`` / ``ua_match`` on EVERY call to
# ``BrowserGymEnv.step()`` (because VWA's ``task.validate()`` is
# step-time, not done-time), so any task whose evaluator includes
# ``fuzzy_match`` or ``page_image_query`` blows up with
# ``NotFoundError: The model gpt-4-1106-preview does not exist or you
# do not have access to it``.
#
# In the 200-task pinned subset (cold_start/task_samples/
# browsergym_visualwebarena_200.txt) this hits 18 / 200 tasks:
#   - 10 tasks with ``string_match`` + ``fuzzy_match`` reference
#   -  8 tasks with ``page_image_query`` (image captioning at eval time)
#
# This script:
#   1. Locates the upstream ``helper_functions.py`` in the active conda
#      env (default: browsergym).
#   2. Replaces the two hardcoded model strings with a module-level
#      ``_VWA_JUDGE_MODEL = os.environ.get("VWA_JUDGE_MODEL", "gpt-4o")``
#      so the judge model is overridable without re-patching.
#   3. Idempotent — re-running is safe; the patch is a no-op if already
#      applied.
#
# Usage:
#   bash install/patch_vwa_judge_model.sh                       # use default conda env "browsergym"
#   CONDA_ENV=browsergym bash install/patch_vwa_judge_model.sh
#   PYTHON=/path/to/python bash install/patch_vwa_judge_model.sh
#
# Override the judge model at run time via the env file:
#   echo 'export VWA_JUDGE_MODEL=gpt-5.4' >> cold_start/visualwebarena_env.sh
#

set -uo pipefail

PY_BIN="${PYTHON:-/workspace/miniconda3/envs/${CONDA_ENV:-browsergym}/bin/python}"

if [ ! -x "$PY_BIN" ]; then
    echo "[ERROR] python not found at $PY_BIN"
    echo "        set PYTHON=... or CONDA_ENV=... and re-run"
    exit 1
fi

# Discover the file via sysconfig + os.path WITHOUT importing the package
# itself. The upstream ``visualwebarena.evaluation_harness`` package
# transitively imports ``nltk`` which in turn imports ``sqlite3``, which
# can fail at module-load time on systems with libstdc++ ABI mismatches
# (CXXABI_1.3.15). The file we need to patch is plain text — no need
# to actually execute the module.
HELPER_PATH=$("$PY_BIN" -c '
import os, sysconfig
sp = sysconfig.get_paths()["purelib"]
candidate = os.path.join(sp, "visualwebarena", "evaluation_harness", "helper_functions.py")
if os.path.isfile(candidate):
    print(candidate)
' 2>/dev/null)

if [ -z "$HELPER_PATH" ] || [ ! -f "$HELPER_PATH" ]; then
    echo "[ERROR] could not locate visualwebarena/evaluation_harness/helper_functions.py"
    echo "        in the site-packages of $PY_BIN"
    exit 1
fi

echo "==> Patching $HELPER_PATH"

# Idempotency guard.
if grep -q '_VWA_JUDGE_MODEL' "$HELPER_PATH"; then
    echo "    [skip] already patched (sentinel _VWA_JUDGE_MODEL present)"
    exit 0
fi

# Use a single python pass to apply both edits atomically — sed-on-multi-line
# is brittle, and two textual replacements need to land together to keep the
# module valid.
"$PY_BIN" - "$HELPER_PATH" <<'PY'
import sys, re, os, shutil

target = sys.argv[1]
with open(target, "r", encoding="utf-8") as f:
    src = f.read()

# 1. Inject ``import os`` (already present in stdlib import block usually,
#    but this file imports ``json`` and not ``os`` upstream).
if "\nimport os\n" not in src.split("from beartype")[0]:
    src = src.replace(
        "import json\nfrom datetime",
        "import json\nimport os\nfrom datetime",
        1,
    )

# 2. Add the module-level _VWA_JUDGE_MODEL right after the openai_utils import.
inject = (
    "\n\n# Patched by install/patch_vwa_judge_model.sh — upstream hardcodes\n"
    "# ``gpt-4-1106-preview`` (deprecated 2024-Q1 model). Override at\n"
    "# import time via VWA_JUDGE_MODEL env var; default = ``gpt-4o``.\n"
    '_VWA_JUDGE_MODEL = os.environ.get("VWA_JUDGE_MODEL", "gpt-4o")\n'
)
anchor = (
    "from ..llms.providers.openai_utils import (\n"
    "    generate_from_openai_chat_completion,\n"
    ")"
)
if anchor not in src:
    print(f"[ERROR] could not find expected import anchor in {target}", file=sys.stderr)
    sys.exit(2)
src = src.replace(anchor, anchor + inject, 1)

# 3. Replace the two hardcoded model strings.
n = src.count('model="gpt-4-1106-preview"')
if n != 2:
    print(f"[ERROR] expected exactly 2 occurrences of model=\"gpt-4-1106-preview\", found {n}", file=sys.stderr)
    sys.exit(3)
src = src.replace('model="gpt-4-1106-preview"', "model=_VWA_JUDGE_MODEL")

# Write atomically.
backup = target + ".vwa_judge_unpatched.bak"
if not os.path.exists(backup):
    shutil.copy2(target, backup)
with open(target, "w", encoding="utf-8") as f:
    f.write(src)
print(f"    [ok] patched {target}")
print(f"    [ok] backup at {backup}")
PY

echo ""
echo "==> Verifying (text-only — does not import the package)"
"$PY_BIN" - "$HELPER_PATH" <<'PY'
import sys

target = sys.argv[1]
with open(target) as f:
    src = f.read()

bad = src.count('model="gpt-4-1106-preview"')
ok  = src.count("model=_VWA_JUDGE_MODEL")
sentinel = "_VWA_JUDGE_MODEL" in src
print(f"    hardcoded gpt-4-1106-preview hits remaining: {bad} (expect 0)")
print(f"    references to _VWA_JUDGE_MODEL:              {ok}  (expect 2)")
print(f"    sentinel present:                            {sentinel}")
if bad != 0 or ok != 2 or not sentinel:
    sys.exit(4)
PY

echo ""
echo "==> Done. Override default with:"
echo "      export VWA_JUDGE_MODEL=gpt-4o      # default"
echo "      export VWA_JUDGE_MODEL=gpt-5.4     # if you want gpt-5.x judge"
echo "      export VWA_JUDGE_MODEL=gpt-4o-mini # if you want cheaper judge"
