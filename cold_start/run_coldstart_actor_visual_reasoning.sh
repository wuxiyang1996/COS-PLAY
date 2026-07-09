#!/usr/bin/env bash
#
# run_coldstart_actor_visual_reasoning.sh — gpt-5.5 actor cold-start over the
# four visual-reasoning benchmarks declared in ``visual_reasoning_wrapper``:
#
#   - visual_toolbench  (image QA, free-form answers)
#   - tir_bench         (image QA, free-form answers)
#   - video_holmes      (video MCQ, A..F)
#   - siv_bench         (video MCQ, A..E)
#
# For each sample the pipeline:
#   1. Loads the image / samples N frames from the video.
#   2. Calls gpt-5.5 (vision) to produce a canonical <state>...</state>
#      schema using ``vlm_wrapper.schema``.
#   3. Calls gpt-5.5 (actor agent) with the schema + the question + the
#      valid action space and chooses ONE answer via OpenAI function
#      calling.
#   4. Writes a per-sample JSON record (schema + answer + gold + correct)
#      to <codebase_root>/Cold-start-out-visual-reasoning/<run_id>/<benchmark>/.
#
# Default behaviour: 5 test cases per benchmark (20 total), vision ON,
# frames saved to disk for debug, verbose logging.
#
# Usage:
#
#   # All four benchmarks, 5 test cases each (default)
#   bash cold_start/run_coldstart_actor_visual_reasoning.sh
#
#   # Just the image benchmarks, 3 cases each
#   bash cold_start/run_coldstart_actor_visual_reasoning.sh \
#       --benchmarks visual_toolbench tir_bench --num_test_cases 3 -v
#
#   # Cheap dry-run: skip the vision call (still calls the actor LLM
#   # over the question text alone)
#   bash cold_start/run_coldstart_actor_visual_reasoning.sh --no_vision -v
#
#   # Pin a run id (handy for resuming / tagging an experiment)
#   bash cold_start/run_coldstart_actor_visual_reasoning.sh \
#       --run_id smoke_2026_n5 -v
#
# Optional environment variables:
#   OPENAI_API_KEY / OPENROUTER_API_KEY  one of these must be set, OR the
#                                        sibling ``api_keys.py`` will be
#                                        auto-loaded by the python launcher.
#   PYTHONPATH                           extra paths to prepend.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODEBASE_ROOT/.." && pwd)"
PY_LAUNCHER="${SCRIPT_DIR}/generate_cold_start_actor_visual_reasoning.py"

# ── PYTHONPATH ────────────────────────────────────────────────────────────
PYPATH_ADD=("${CODEBASE_ROOT}" "${WORKSPACE_ROOT}")
JOINED_PYPATH="$(IFS=:; echo "${PYPATH_ADD[*]}")"
export PYTHONPATH="${JOINED_PYPATH}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Headless rendering (some video decoders touch SDL on import) ─────────
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export PYGLET_HEADLESS="${PYGLET_HEADLESS:-1}"

# ── Parse user args ───────────────────────────────────────────────────────
EXTRA_ARGS=("$@")

# ── API key check (warn-only; the python launcher auto-loads api_keys.py) ─
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ ! -f "${CODEBASE_ROOT}/api_keys.py" ] \
        && [ ! -f "${CODEBASE_ROOT}/../api_keys.py" ] \
        && [ ! -f "${SCRIPT_DIR}/api_keys.py" ]; then
        echo "[WARNING] Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set,"
        echo "          and no api_keys.py was found alongside the repo."
        echo "          Both VLM and actor calls will fail. See .env.example."
    fi
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Cold-Start Actor — Visual Reasoning Benchmarks (gpt-5.5)"
echo "================================================================"
echo "  Codebase:    ${CODEBASE_ROOT}"
echo "  Launcher:    ${PY_LAUNCHER}"
echo "  Python:      $(python3 --version 2>&1)"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:     ${OPENROUTER_API_KEY:0:12}... (OpenRouter)"
[ -z "${OPENROUTER_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && echo "  API key:     ${OPENAI_API_KEY:0:12}... (OpenAI)"

# ── Smoke-import the benchmark & schema modules so we fail fast ──────────
SMOKE_OUTPUT="$(python3 - <<'PYEOF' 2>&1 || true
import warnings, sys
warnings.filterwarnings("ignore")
ok = []
try:
    from vlm_wrapper.schema import (
        build_system_prompt, build_user_message, parse_schema_output,
    )
    ok.append("vlm_wrapper.schema")
except Exception as exc:
    print(f"FAIL vlm_wrapper.schema: {exc}", file=sys.stderr)
try:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        iter_visual_toolbench_samples,
    )
    ok.append("benchmarks.visual_toolbench")
except Exception as exc:
    print(f"WARN benchmarks.visual_toolbench: {exc}", file=sys.stderr)
try:
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples,
    )
    ok.append("benchmarks.tir_bench")
except Exception as exc:
    print(f"WARN benchmarks.tir_bench: {exc}", file=sys.stderr)
try:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, sample_video_frames,
    )
    ok.append("benchmarks.video_holmes")
except Exception as exc:
    print(f"WARN benchmarks.video_holmes: {exc}", file=sys.stderr)
try:
    from visual_reasoning_wrapper.benchmarks.siv_bench import (
        iter_siv_bench_samples,
    )
    ok.append("benchmarks.siv_bench")
except Exception as exc:
    print(f"WARN benchmarks.siv_bench: {exc}", file=sys.stderr)
print("imports_ok=" + ",".join(ok))
PYEOF
)"
echo "${SMOKE_OUTPUT}" | sed 's/^/  /'
echo "================================================================"
echo ""

# ── Defaults: when the user passes nothing, run the canonical 5-cases-per-
#               benchmark sweep, save frames, and turn on verbose output.
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--num_test_cases 5 --save_frames -v)
fi

# ── Dispatch ──────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python3 "${PY_LAUNCHER}" "${EXTRA_ARGS[@]}"
EXIT_CODE=$?

# ── Post-run summary ──────────────────────────────────────────────────────
OUTPUT_BASE="${CODEBASE_ROOT}/Cold-start-out-visual-reasoning"
prev_was_outdir=0
LATEST_OUTPUT_DIR=""
for arg in "${EXTRA_ARGS[@]}"; do
    if [ "$prev_was_outdir" -eq 1 ]; then
        LATEST_OUTPUT_DIR="$arg"
        prev_was_outdir=0
        continue
    fi
    case "$arg" in
        --output_dir|--output-dir) prev_was_outdir=1 ;;
        --output_dir=*) LATEST_OUTPUT_DIR="${arg#--output_dir=}" ;;
        --output-dir=*) LATEST_OUTPUT_DIR="${arg#--output-dir=}" ;;
    esac
done

if [ -z "${LATEST_OUTPUT_DIR}" ] && [ -L "${OUTPUT_BASE}/latest" ]; then
    LATEST_OUTPUT_DIR="${OUTPUT_BASE}/$(readlink "${OUTPUT_BASE}/latest")"
fi

echo ""
echo "================================================================"
echo "  Visual-Reasoning Actor — Post-Run Summary"
echo "================================================================"
if [ -n "${LATEST_OUTPUT_DIR}" ] && [ -d "${LATEST_OUTPUT_DIR}" ]; then
    TOTAL=0
    for bench_dir in "${LATEST_OUTPUT_DIR}"/*/; do
        [ -d "$bench_dir" ] || continue
        bench="$(basename "$bench_dir")"
        count=$(find "$bench_dir" -maxdepth 1 -name 'sample_*.json' 2>/dev/null | wc -l)
        TOTAL=$((TOTAL + count))
        has_summary="no"; [ -f "$bench_dir/summary.json" ] && has_summary="yes"
        has_jsonl="no";   [ -f "$bench_dir/samples.jsonl" ] && has_jsonl="yes"
        printf "  %-22s %3d samples  summary=%s  jsonl=%s\n" "$bench" "$count" "$has_summary" "$has_jsonl"
    done
    echo ""
    echo "  Total samples: ${TOTAL}"
    echo "  Output dir:    ${LATEST_OUTPUT_DIR}"
    [ -f "${LATEST_OUTPUT_DIR}/batch_summary.json" ] && \
        echo "  Master summary: ${LATEST_OUTPUT_DIR}/batch_summary.json"
else
    echo "  (no output produced — exit code ${EXIT_CODE})"
fi
echo "================================================================"

exit ${EXIT_CODE}
