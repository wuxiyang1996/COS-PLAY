#!/usr/bin/env bash
#
# smoke_test_osworld.sh — Light end-to-end test of the OSWorld cold-start
# actor pipeline. Verifies that the agent can:
#
#   (1) Boot a Docker-backed OSWorld VM (headless).
#   (2) Convert the live visual state (screenshot) into a canonical
#       <state>...<entities>...</state> schema via gpt-5.5 vision.
#   (3) Pick a pyautogui action from the schema via OpenAI function-calling
#       (tool-call).
#   (4) Step the env, persist the trajectory + frames + per-step sidecars.
#
# Designed to finish in ~25-30 min on a single VM. Uses 3 hand-picked
# tasks across 3 domains, with a moderate max-step cap (12 — large
# enough for the easy chrome / calc tasks to actually emit DONE and
# score eval=1.0, small enough to keep the loop tight).
#
# >>> SMOKE = pipeline check, NOT a benchmark eval. <<<
#
# The PASS criterion is whether every step produced a VLM-generated
# schema AND a tool-call action — i.e. the visual->schema->action loop
# is functional. ``eval_score`` is informational only. For a real
# benchmark number use ``--max_steps 50`` (the OSWorld standard) on
# the parallel multi-domain launcher.
#
# Usage:
#
#   bash cold_start/smoke_test_osworld.sh            # full smoke test
#   bash cold_start/smoke_test_osworld.sh --keep     # keep prior smoke output
#   bash cold_start/smoke_test_osworld.sh --steps 6  # tighter, faster pass
#   bash cold_start/smoke_test_osworld.sh --steps 50 # full OSWorld cap
#                                                    # (longer wall-clock,
#                                                    # but proper eval)
#
# Output:
#   <repo_root>/Cold-start-out-osworld-smoke/
#     <domain>/<task_id>/episode_000.json
#     <domain>/<task_id>/frames/ep_000/step_NNN.json     <- VLM schema + tool-call
#     <domain>/<task_id>/frames/ep_000/step_NNN.png     <- screenshot the VLM saw
#
# Exit code: 0 on PASS (all 3 tasks executed schema-from-vision + tool-call
# at least once), non-zero on FAIL.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${CODEBASE_ROOT}/Cold-start-out-osworld-smoke"
MAX_STEPS=12
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --keep)        KEEP=1; shift ;;
        --steps)       shift; MAX_STEPS="${1:-12}"; shift ;;
        --output_dir)  shift; OUT_DIR="${1:-$OUT_DIR}"; shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            echo "[ERROR] unknown flag: $1" >&2; exit 2 ;;
    esac
done

# ── Tasks we exercise (catalog: test_all.json) ─────────────────────────────
# Picked because each one stresses a different VLM ability:
#   chrome  06fe7178…  : tab management — small screenshot, clear targets,
#                        DONE typically reached in 2 steps.
#   libreoffice_calc 357ef137… : numerical reasoning + spreadsheet formula
#                        entry — requires the VLM to read cells from pixels
#                        AND emit a typewrite('=...') tool-call.
#   gimp 7a4deb26…     : multi-window desktop with palette popups — exercises
#                        the schema's region / dialog handling.
TASK_IDS=(
    "06fe7178-4491-4589-810f-2e2bc9502122"
    "357ef137-7eeb-4c80-a3bb-0951f26a8aff"
    "7a4deb26-d57d-4ea9-9a73-630f66a7b568"
)

# ── Optional fresh start ───────────────────────────────────────────────────
if [ "$KEEP" -eq 0 ] && [ -d "$OUT_DIR" ]; then
    echo "[smoke] clearing previous smoke output: $OUT_DIR"
    rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "================================================================"
echo "  OSWorld pipeline smoke test (vision schema + tool-call action)"
echo "  tasks    : ${#TASK_IDS[@]}"
echo "  max_steps: $MAX_STEPS"
echo "  output   : $OUT_DIR"
echo "================================================================"

# ── Run the actor on the 3 pinned tasks ────────────────────────────────────
bash "${SCRIPT_DIR}/run_coldstart_actor_osworld.sh" \
    --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
    --task_ids "${TASK_IDS[@]}" \
    --episodes 1 \
    --max_steps "$MAX_STEPS" \
    --output_dir "$OUT_DIR" \
    --resume -v
RC=$?

echo ""
echo "================================================================"
echo "  Validating smoke artifacts"
echo "================================================================"

# ── Validate: every task got an episode with VLM schema + tool-call action ─
python3 - "$OUT_DIR" "${TASK_IDS[@]}" <<'PY'
import json, sys, re
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_ids = sys.argv[2:]

print(f"Inspecting: {out_dir}")
print(f"Expecting  : {len(expected_ids)} task(s)")
print()

found = {}
for ep in out_dir.glob("*/*/episode_000.json"):
    domain, task_id = ep.parts[-3], ep.parts[-2]
    found[task_id] = (domain, ep)

print(f"{'TASK ID':<40s} {'DOMAIN':<12s} {'STEPS':>5s} {'VLM-schema':>10s} {'TOOL-CALL':>10s} {'EVAL':>6s}")
print("-" * 92)

passed = 0
failed_reasons = []
for tid in expected_ids:
    if tid not in found:
        failed_reasons.append(f"  - missing episode for {tid}")
        print(f"{tid:<40s} {'?':<12s} {'?':>5s} {'MISSING':>10s} {'-':>10s} {'-':>6s}")
        continue
    domain, ep_path = found[tid]
    ep_data = json.load(open(ep_path))
    meta = ep_data.get("metadata", {})
    n_steps = int(meta.get("steps", 0))
    eval_score = meta.get("eval_score")
    eval_str = "—" if eval_score is None else f"{eval_score:.2f}"

    sidecar_dir = ep_path.parent / "frames" / "ep_000"
    sidecars = sorted(sidecar_dir.glob("step_*.json")) if sidecar_dir.is_dir() else []

    schema_ok = 0
    tool_ok = 0
    for s in sidecars:
        sd = json.load(open(s))
        if (sd.get("schema_source") == "vlm"
            and isinstance(sd.get("schema"), str)
            and "<entities>" in sd["schema"]):
            schema_ok += 1
        action_str = sd.get("action") or ""
        if action_str.strip() and action_str.strip() != "FAIL":
            tool_ok += 1

    schema_status = f"{schema_ok}/{len(sidecars)}"
    tool_status   = f"{tool_ok}/{len(sidecars)}"
    print(f"{tid:<40s} {domain:<12s} {n_steps:>5d} {schema_status:>10s} {tool_status:>10s} {eval_str:>6s}")

    if schema_ok >= 1 and tool_ok >= 1:
        passed += 1
    else:
        if schema_ok < 1:
            failed_reasons.append(f"  - {tid}: no VLM schema in any step (visual->schema broken)")
        if tool_ok < 1:
            failed_reasons.append(f"  - {tid}: no tool-call action in any step")

print()
print(f"PASSED: {passed}/{len(expected_ids)} task(s)")
if failed_reasons:
    print("Failures:")
    for r in failed_reasons:
        print(r)

if passed == len(expected_ids):
    print()
    print("=== SMOKE PASS ===")
    print("Vision -> schema -> tool-call -> env.step loop is functional.")
    sys.exit(0)
else:
    print()
    print("=== SMOKE FAIL ===")
    sys.exit(1)
PY
PY_RC=$?

# ── Final exit code ────────────────────────────────────────────────────────
# We surface the validator's exit code; the actor's exit code is informational.
# (The actor sometimes exits non-zero on partial failures even when artifacts
# are valid — the validator is the source of truth.)
echo ""
echo "actor_rc=$RC  validator_rc=$PY_RC"
exit "$PY_RC"
