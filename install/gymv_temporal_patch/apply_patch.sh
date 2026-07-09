#!/usr/bin/env bash
# ==============================================================================
# apply_patch.sh — apply the Temporal/* (stable-retro) upgrades on top of a
# checked-out ModalMinds/gym-v tree.
#
# Idempotent: safe to re-run. It will:
#   1. (optional) install stable-retro into the active conda env
#   2. extract ROMs/Mega_Drive_Mini_Full_Set.zip and run `python -m retro.import`
#   3. copy the patched gym_v/wrappers/{__init__,observation}.py and
#      gym_v/envs/multi_turn/temporal/retro_env.py into the gym-v source tree
#   4. fix the Airstriker game-name typo in gym_v/envs/__init__.py
#   5. fix the same typo in tests/test_retro_integration.py
#   6. drop the smoke-test script under gym-v/examples/
#   7. run the smoke test
#
# Usage:
#   bash install/gymv_temporal_patch/apply_patch.sh \
#        [GYMV_DIR=/fs/gamma-projects/vlm-robot/gym-v] \
#        [ROM_ZIP=/fs/gamma-projects/vlm-robot/ROMs/Mega_Drive_Mini_Full_Set.zip]
#
# Prerequisites:
#   * conda env `gymv` exists and gym-v is `pip install -e .` into it
#     (i.e. the existing install/install_gymv.sh has already been run)
#   * The Mega_Drive_Mini_Full_Set.zip is on disk somewhere
# ==============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYMV_DIR="${1:-/fs/gamma-projects/vlm-robot/gym-v}"
ROM_ZIP="${2:-/fs/gamma-projects/vlm-robot/ROMs/Mega_Drive_Mini_Full_Set.zip}"
ENV_NAME="${GYMV_ENV:-gymv}"

if [[ ! -d "$GYMV_DIR" ]]; then
    echo "ERROR: gym-v not found at $GYMV_DIR" >&2
    echo "       Run install/install_gymv.sh first, or pass the path as arg 1." >&2
    exit 1
fi

# --- conda activation -------------------------------------------------------
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090,SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- 1. stable-retro --------------------------------------------------------
echo "[1/7] Ensuring stable-retro is installed in '$ENV_NAME' ..."
if ! python -c "import stable_retro" >/dev/null 2>&1; then
    pip install stable-retro
else
    echo "      stable_retro already importable."
fi

# --- 2. ROM extract + import -----------------------------------------------
DATA_DIR="$(python -c 'import stable_retro;print(stable_retro.data.path())')/stable"
NEED_IMPORT=0
for g in AlteredBeast-Genesis-v0 CastleOfIllusion-Genesis-v0 \
         CastlevaniaBloodlines-Genesis-v0 Columns-Genesis-v0 \
         DynamiteHeaddy-Genesis-v0 GoldenAxe-Genesis-v0 \
         KidChameleon-Genesis-v0 MortalKombatII-Genesis-v0 \
         SpaceHarrierII-Genesis-v0 StreetsOfRage2-Genesis-v0 \
         Strider-Genesis-v0 ThunderForceIII-Genesis-v0; do
    [[ -f "$DATA_DIR/$g/rom.md" ]] || NEED_IMPORT=1
done

if [[ "$NEED_IMPORT" == "1" ]]; then
    if [[ ! -f "$ROM_ZIP" ]]; then
        echo "ERROR: ROMs not yet imported and ROM zip not found: $ROM_ZIP" >&2
        echo "       Download from https://archive.org/download/megadrivemini/Mega_Drive_Mini_Full_Set.zip" >&2
        echo "       and pass its path as argument 2." >&2
        exit 1
    fi
    SCRATCH="$(dirname "$ROM_ZIP")/genesis_roms_flat"
    OUTER="$(dirname "$ROM_ZIP")/genesis_roms"
    echo "[2/7] Extracting ROMs ..."
    mkdir -p "$OUTER" "$SCRATCH"
    unzip -q -o "$ROM_ZIP" -d "$OUTER"
    for f in "$OUTER"/Mega_Drive_Mini_ROMs/*.zip; do
        unzip -q -o "$f" -d "$SCRATCH"
    done
    echo "      Importing into stable-retro ..."
    python -m retro.import "$SCRATCH"
else
    echo "[2/7] All 12 commercial ROMs already imported — skipping."
fi

# --- 3. copy patched files --------------------------------------------------
echo "[3/7] Copying patched gym-v files into $GYMV_DIR ..."
install -m 0644 "$PATCH_DIR/gym_v/wrappers/__init__.py" \
                "$GYMV_DIR/gym_v/wrappers/__init__.py"
install -m 0644 "$PATCH_DIR/gym_v/wrappers/observation.py" \
                "$GYMV_DIR/gym_v/wrappers/observation.py"
install -m 0644 "$PATCH_DIR/gym_v/envs/multi_turn/temporal/retro_env.py" \
                "$GYMV_DIR/gym_v/envs/multi_turn/temporal/retro_env.py"

# --- 4. Airstriker registration typo (gym_v/envs/__init__.py) --------------
INIT_PY="$GYMV_DIR/gym_v/envs/__init__.py"
if grep -q '"Airstriker-Genesis"' "$INIT_PY"; then
    echo "[4/7] Fixing Airstriker game-name typo in $INIT_PY ..."
    sed -i 's/"Airstriker-Genesis"/"Airstriker-Genesis-v0"/' "$INIT_PY"
else
    echo "[4/7] Airstriker game-name already correct — skipping."
fi

# --- 5. Airstriker test typo ------------------------------------------------
TEST_PY="$GYMV_DIR/tests/test_retro_integration.py"
if [[ -f "$TEST_PY" ]] && grep -q 'TEST_GAME = "Airstriker-Genesis"$' "$TEST_PY"; then
    echo "[5/7] Fixing Airstriker test typo in $TEST_PY ..."
    sed -i 's/TEST_GAME = "Airstriker-Genesis"$/TEST_GAME = "Airstriker-Genesis-v0"/' "$TEST_PY"
else
    echo "[5/7] Test file already correct or missing — skipping."
fi

# --- 6. smoke test script ---------------------------------------------------
echo "[6/7] Installing examples/temporal_smoketest.py ..."
install -m 0644 "$PATCH_DIR/examples/temporal_smoketest.py" \
                "$GYMV_DIR/examples/temporal_smoketest.py"

# --- 7. run smoke test ------------------------------------------------------
echo "[7/7] Running smoke test ..."
( cd "$GYMV_DIR" && python examples/temporal_smoketest.py )

echo
echo "Patch applied successfully."
echo "  gym-v dir:   $GYMV_DIR"
echo "  conda env:   $ENV_NAME"
echo "  Activate:    conda activate $ENV_NAME"
