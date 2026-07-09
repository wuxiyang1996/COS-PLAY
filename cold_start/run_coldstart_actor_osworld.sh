#!/usr/bin/env bash
#
# run_coldstart_actor_osworld.sh — gpt-5.5 actor cold-start over OSWorld.
#
# Runs ``cold_start/generate_cold_start_actor_osworld.py`` to drive the
# COS-PLAY actor pipeline against the live OSWorld DesktopEnv (a real
# Ubuntu / Windows / macOS guest VM running under Docker / VMware /
# AWS / VirtualBox — there is NO synthetic / offline mode).
#
# At each step the agent:
#
#   1. Captures a multimodal observation: PIL screenshot (decoded from
#      the VM framebuffer), namespaced AT-SPI / UI-Automation XML
#      accessibility tree, recent VM terminal output, and the
#      task instruction.
#   2. Computes the deterministic XML-walked schema via
#      ``osworld_wrapper.heuristic.obs_to_schema`` (free baseline,
#      always emitted as a canonical fallback).
#   3. Calls gpt-5.5 (vision) on the screenshot — with the AT-SPI tree
#      as grounding context — to produce the canonical
#      ``<state>...</state>`` schema following ``vlm_wrapper.schema``.
#   4. Calls gpt-5.5 with the schema + a candidate-action list (a11y
#      click targets + global hotkeys + DONE/FAIL/WAIT) and OpenAI
#      function-calling to choose ONE pyautogui action.
#   5. ``env.step(action_string)``, saves the Episode/Experience trail
#      (with schema, screenshot path, candidate actions, reasoning,
#      eval score on DONE) into
#      ``<codebase_root>/Cold-start-out-osworld/<domain>/<safe_task_id>/``.
#
# Usage:
#
#   # Default: 1 episode of the smoke task on Docker, 75 steps, --resume.
#   # 75 is the new default (was 50) — the May-2026 cold-start audit
#   # showed 46% of episodes truncated at step 50 with eval_score=None,
#   # mostly office/gimp/vlc multi-dialog tasks that need 30-60 steps.
#   bash cold_start/run_coldstart_actor_osworld.sh
#
#   # Eval-grade benchmark run — three tiers (cost ↑ → score ↑):
#   #   medium: reasoning_effort=medium, temperature=0.0, no DONE-nudge,
#   #           max_steps=75. Cheapest published-baseline-comparable run.
#   #   high:   same as medium + reasoning_effort=high. ~2-3x token spend
#   #           on the schema/action calls; +3-5pp pass-rate on multi-step.
#   #   max:    high + max_steps=100. For long-tail GIMP / VLC / multi-app
#   #           workflows. Highest spend, highest published number.
#   bash cold_start/run_coldstart_actor_osworld.sh --eval_mode medium \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#   bash cold_start/run_coldstart_actor_osworld.sh --eval_mode high \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
#   # Advanced steering (opt-in, OSWorld-only — these flags are isolated
#   # in cold_start/osworld_steering.py and do NOT touch the main loop
#   # or other corpora when off). Each costs extra LLM calls but lifts
#   # pass-rate on stuck/long trajectories:
#   #   --enable_memory: every K steps, summarise recent actions and
#   #                    inject as a <memory> block on the next prompt
#   #                    (combat the "lost-in-trajectory" failure mode).
#   #   --enable_reflection: when the agent has 2+ consecutive no-op
#   #                        steps, fire a small reflection LLM call
#   #                        ("why did the last action fail? give 3
#   #                        alternatives") and inject the answer.
#   #   --enable_self_verify: before accepting a DONE emission, verify
#   #                         with one extra screenshot+a11y vision call
#   #                         that the goal is objectively satisfied.
#   bash cold_start/run_coldstart_actor_osworld.sh --eval_mode high \
#       --enable_memory --enable_reflection --enable_self_verify \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
#   # In-context skill retrieval (opt-in). Loads a skill_bank.jsonl
#   # produced by labeling/extract_skillbank_gpt54.py or
#   # skill_transfer_test/extract/, and at the start of every episode
#   # retrieves the top-K skills relevant to the task instruction.
#   # The retrieved protocols are formatted as in-context demonstrations
#   # in the actor's user prompt.
#   bash cold_start/run_coldstart_actor_osworld.sh --eval_mode high \
#       --skill_bank_path skill_transfer_test/skill_bank_local/full_v5/osworld/per_episode/skill_bank.jsonl \
#       --skill_retrieval_top_k 3 \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
#   # List task domains/IDs in the catalog and exit
#   bash cold_start/run_coldstart_actor_osworld.sh --list_tasks
#
#   # Restrict to two domains, cap to 2 tasks each
#   bash cold_start/run_coldstart_actor_osworld.sh \
#       --domains chrome os --tasks_per_domain 2 \
#       --episodes 1 --save_frames -v
#
#   # Quick pipeline check (lower budget — easy tasks still solve, but
#   # office/gimp tasks may run out of steps):
#   bash cold_start/run_coldstart_actor_osworld.sh \
#       --task_ids 06fe7178-4491-4589-810f-2e2bc9502122 --max_steps 6 -v
#
#   # Show all options:
#   bash cold_start/run_coldstart_actor_osworld.sh --help
#
#   # Multi-provider cross-machine eval (Claude / Gemini / Qwen3-VL):
#   #   The same code path supports any OpenRouter-hosted vision+tool
#   #   model — the only thing that changes is ``--model`` (full slash
#   #   id like ``anthropic/claude-sonnet-4.6``) and which ``--eval_mode``
#   #   tier is meaningful for that family. Use the wrapper:
#   #     bash cold_start/run_osworld_multimodel.sh --provider claude-sonnet ...
#   #     bash cold_start/run_osworld_multimodel.sh --provider gemini-pro ...
#   #     bash cold_start/run_osworld_multimodel.sh --provider qwen3-vl ...
#   #   For a 30-second LLM-only credentials/plumbing check before
#   #   booting the VM, run:
#   #     python cold_start/smoke_multimodel.py
#
# Hard-wired modes (NO opt-out):
#   - VM is ALWAYS HEADLESS (Xvfb-backed; no GUI on the host).
#   - The VLM (gpt-5.5 vision) is REQUIRED on every step.
#   - Frames are SAVED BY DEFAULT (PNG + step_NNN.json sidecar). Pass
#     ``--no_save_frames`` to skip when disk pressure matters.
#
# Pre-reqs:
#   - ``osworld`` conda env: ``bash install/install_osworld.sh``
#     (see osworld_wrapper/README.md for the VM backend setup)
#   - Docker daemon running and ``happysixd/osworld-docker`` image pulled
#     (the launcher will warn but not bail if the image is missing —
#      OSWorld will pull it on first use)
#   - ``./docker_vm_data/Ubuntu.qcow2`` (~23 GB) at the launch cwd
#     (DockerVMManager.VMS_DIR is './docker_vm_data' relative to cwd)
#   - OpenAI/OpenRouter API key (REQUIRED — vision is mandatory):
#       OPENAI_API_KEY / OPENROUTER_API_KEY env, or ``api_keys.py`` next
#       to the repo root, or pass ``--api_key`` on the CLI.
#
# Optional environment variables:
#   PYTHONPATH                          extra paths to prepend (the
#                                       codebase root and workspace root
#                                       are added automatically).
#   OSWORLD_CONDA_ENV                   conda env to auto-activate when
#                                       desktop_env / docker imports fail
#                                       (default: ``osworld``).
#   OSWORLD_VM_DATA_DIR                 directory containing
#                                       ``Ubuntu.qcow2`` — the launcher
#                                       chdirs into this so OSWorld's
#                                       relative path lookup hits.
#                                       Default: ``<repo>/docker_vm_data``.

set -uo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"

# ── Parse user args (forwarded to the python launcher) ────────────────────
EXTRA_ARGS=()
CONDA_ENV="${OSWORLD_CONDA_ENV:-osworld}"
VM_DATA_DIR="${OSWORLD_VM_DATA_DIR:-${CODEBASE_ROOT}/docker_vm_data}"
while [ $# -gt 0 ]; do
    case "$1" in
        --conda_env|--conda-env)
            shift; CONDA_ENV="${1:-$CONDA_ENV}"; shift ;;
        --vm_data_dir|--vm-data-dir)
            shift; VM_DATA_DIR="${1:-$VM_DATA_DIR}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Auto-activate the osworld conda env if available ──────────────────────
# The desktop_env stack pins gymnasium~=0.28.1 / torch~=2.5.0 etc., so it
# always lives in its own env. If the user is already in a python that can
# import desktop_env we leave them alone; otherwise we try to switch to
# ``$CONDA_ENV`` (default ``osworld``). Override with --conda_env or
# OSWORLD_CONDA_ENV=<name>.
_python_has_osworld() {
    python3 -c "import desktop_env, docker" >/dev/null 2>&1
}

if ! _python_has_osworld; then
    if command -v conda >/dev/null 2>&1; then
        if conda env list | awk '$1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV"; then
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV"
            echo "[INFO] Activated conda env: $CONDA_ENV"
        else
            echo "[NOTE] conda env '$CONDA_ENV' not found; staying in current python."
            echo "       If imports fail, install via:"
            echo "         bash install/install_osworld.sh"
        fi
    fi
fi

# ── Headless display (some OSWorld renderers / pyautogui need an X server) ─
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb >/dev/null 2>&1 && ! pgrep -x Xvfb >/dev/null 2>&1; then
        Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
        sleep 0.5
    fi
    if command -v Xvfb >/dev/null 2>&1; then
        export DISPLAY="${DISPLAY:-:99}"
    fi
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── API key check ─────────────────────────────────────────────────────────
# Vision is mandatory in this pipeline (VM is headless; the only signal the
# actor has is the gpt-5.5 visual schema). The launcher will still try
# api_keys.py at import time, so we only NOTE here, not FAIL — the python
# launcher itself will hard-fail with a clear message if no key resolves.
LIST_ONLY=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --list_tasks|--list-tasks) LIST_ONLY=1 ;;
    esac
done
if [ "$LIST_ONLY" -eq 0 ] \
    && [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[NOTE] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set."
    echo "       The launcher will try api_keys.py next to the repo root;"
    echo "       otherwise pass --api_key on the CLI."
    echo "       (Vision is REQUIRED on every step — no --no_vision opt-out.)"
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Cold-Start Actor Agent — OSWorld + gpt-5.5"
echo "================================================================"
echo "  Codebase:    ${CODEBASE_ROOT}"
echo "  Workspace:   ${WORKSPACE_ROOT}"
echo "  DISPLAY:     ${DISPLAY:-unset}"
echo "  Python:      $(python3 --version 2>&1)"
echo "  Conda env:   ${CONDA_DEFAULT_ENV:-unknown}"
echo "  VM data dir: ${VM_DATA_DIR}"

# ── Dependency probe ──────────────────────────────────────────────────────
if python3 -c "import openai" >/dev/null 2>&1; then
    :
else
    echo "[ERROR] 'openai' package missing. Install: pip install openai"
    exit 1
fi

SMOKE_OUTPUT="$(python3 - <<PYEOF 2>&1 || true
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, "${CODEBASE_ROOT}")
sys.path.insert(0, "${WORKSPACE_ROOT}")
ok = []
fail = []
for mod, label, required in [
    ("desktop_env", "desktop_env (REQUIRED — install/install_osworld.sh)", True),
    ("desktop_env.desktop_env", "desktop_env.DesktopEnv", True),
    ("env_wrappers.osworld_wrapper", "env_wrappers.osworld_wrapper", True),
    ("osworld_wrapper.heuristic", "osworld_wrapper.heuristic", True),
    ("vlm_wrapper.schema", "vlm_wrapper.schema", True),
    ("docker", "docker (REQUIRED if --provider_name=docker)", False),
    ("pyautogui", "pyautogui (transitive, OSWorld VM-side)", False),
]:
    try:
        __import__(mod)
        ok.append(label)
    except Exception as exc:
        kind = "FAIL" if required else "WARN"
        print(f"{kind} {label}: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
        if required:
            fail.append(label)
print("imports_ok=" + ", ".join(ok))
PYEOF
)"
echo "  ${SMOKE_OUTPUT}" | sed 's/^/  /'

if echo "${SMOKE_OUTPUT}" | grep -q '^FAIL '; then
    echo ""
    echo "[ERROR] Required OSWorld dependencies are missing."
    echo "        Install with: bash install/install_osworld.sh"
    echo "        Then activate the env:  conda activate ${CONDA_ENV}"
    exit 1
fi

# ── Docker / qcow2 preflight (warnings only — OSWorld can self-heal) ──────
DOCKER_OK=0
QCOW2_OK=0
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        DOCKER_OK=1
        if docker image inspect happysixd/osworld-docker >/dev/null 2>&1; then
            echo "  Docker:      OK (happysixd/osworld-docker present)"
        else
            echo "  Docker:      OK (happysixd/osworld-docker NOT pulled — "
            echo "               first VM boot will pull ~360 MB)"
        fi
    else
        echo "  Docker:      [WARN] daemon unreachable — start with"
        echo "               'sudo systemctl start docker' or use --provider_name=vmware"
    fi
else
    echo "  Docker:      [WARN] CLI not found"
fi

if [ -f "${VM_DATA_DIR}/Ubuntu.qcow2" ]; then
    QCOW2_OK=1
    SIZE_GB="$(du -h "${VM_DATA_DIR}/Ubuntu.qcow2" 2>/dev/null | awk '{print $1}')"
    echo "  qcow2:       OK (${VM_DATA_DIR}/Ubuntu.qcow2 — ${SIZE_GB})"
else
    echo "  qcow2:       [WARN] ${VM_DATA_DIR}/Ubuntu.qcow2 missing"
    echo "               First VM boot will download ~12 GB zip → 23 GB qcow2."
    echo "               To pre-stage:"
    echo "                 mkdir -p '${VM_DATA_DIR}' && cd '${VM_DATA_DIR}'"
    echo "                 curl -L -C - -o Ubuntu.qcow2.zip \\"
    echo "                   https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
    echo "                 unzip -o Ubuntu.qcow2.zip && rm -f Ubuntu.qcow2.zip"
fi

[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:     ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
[ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && echo "  API key:     ${OPENAI_API_KEY:0:12}... (OpenAI)"
echo "================================================================"
echo ""

# ── Defaults: when the user passes nothing, run the smoke task with
#               --resume + verbose stepping. Frames are saved by default.
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--resume -v)
fi

# Surface key modes before spawning the python agent. Vision and headless
# are hard-wired ON in the python launcher (NO opt-out).
NO_SAVE_FRAMES=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --no_save_frames|--no-save-frames) NO_SAVE_FRAMES=1 ;;
    esac
done
echo "  Headless:    YES  (mandatory; VM runs Xvfb-backed)"
echo "  Vision:      ON   (mandatory; gpt-5.5 visual grounding every step)"
echo "  Save frames: $([ $NO_SAVE_FRAMES -eq 0 ] && echo 'YES (PNG + step_NNN.json sidecar — default)' \
                                                || echo 'NO  (--no_save_frames)')"
echo "================================================================"
echo ""

# ── Run the python launcher ───────────────────────────────────────────────
# Important: chdir into the parent of the VM data dir so OSWorld's
# DockerVMManager (which looks for ./docker_vm_data relative to cwd) finds
# the qcow2. We chdir to the dir CONTAINING ``docker_vm_data`` so the
# relative path resolves; if VM_DATA_DIR is ``${CODEBASE_ROOT}/docker_vm_data``
# we cd to ${CODEBASE_ROOT}.
VM_PARENT_DIR="$(dirname "${VM_DATA_DIR}")"
mkdir -p "${VM_DATA_DIR}"
cd "${VM_PARENT_DIR}"
echo "  cwd:         ${VM_PARENT_DIR}  (so OSWorld can find docker_vm_data/)"
echo ""

python3 "${SCRIPT_DIR}/generate_cold_start_actor_osworld.py" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# ── Post-run summary ──────────────────────────────────────────────────────
# Honor --output_dir; otherwise default to Cold-start-out-osworld at the repo root.
OUTPUT_DIR="${CODEBASE_ROOT}/Cold-start-out-osworld"
prev_was_outdir=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$prev_was_outdir" -eq 1 ]; then
        OUTPUT_DIR="$arg"
        prev_was_outdir=0
        continue
    fi
    case "$arg" in
        --output_dir|--output-dir) prev_was_outdir=1 ;;
        --output_dir=*) OUTPUT_DIR="${arg#--output_dir=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#--output-dir=}" ;;
    esac
done

echo ""
echo "================================================================"
echo "  Cold-Start Actor (OSWorld) — Post-Run Summary"
echo "================================================================"

if [ -d "$OUTPUT_DIR" ]; then
    TOTAL=0
    for domain_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$domain_dir" ] || continue
        domain="$(basename "$domain_dir")"
        # Skip top-level files like batch_rollout_summary.json
        case "$domain" in
            _* | "frames" | "*.json") continue ;;
        esac
        domain_count=0
        for task_dir in "$domain_dir"*/; do
            [ -d "$task_dir" ] || continue
            task_id="$(basename "$task_dir")"
            count=$(find "$task_dir" -maxdepth 1 -name 'episode_*.json' \
                ! -name 'episode_buffer.json' 2>/dev/null | wc -l)
            domain_count=$((domain_count + count))
            TOTAL=$((TOTAL + count))
            has_buffer="no"; [ -f "$task_dir/episode_buffer.json" ] && has_buffer="yes"
            has_jsonl="no";  [ -f "$task_dir/rollouts.jsonl" ]      && has_jsonl="yes"
            printf "  %-12s %-44s %3d episodes  buffer=%s  jsonl=%s\n" \
                "$domain" "$task_id" "$count" "$has_buffer" "$has_jsonl"
        done
        printf "  %-12s SUBTOTAL: %3d episodes\n" "$domain" "$domain_count"
    done
    echo ""
    echo "  Total episodes: $TOTAL"
    echo "  Output dir:     $OUTPUT_DIR"
    [ -f "$OUTPUT_DIR/batch_rollout_summary.json" ] && \
        echo "  Master summary: $OUTPUT_DIR/batch_rollout_summary.json"
else
    echo "  (no output produced — exit code ${EXIT_CODE})"
fi

echo ""
echo "  Load into trainer:"
echo "    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records"
echo "    eps = load_episodes_from_jsonl('${OUTPUT_DIR}/<domain>/<safe_task_id>/rollouts.jsonl')"
echo "================================================================"

exit ${EXIT_CODE}
