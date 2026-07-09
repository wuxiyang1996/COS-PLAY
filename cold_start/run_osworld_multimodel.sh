#!/usr/bin/env bash
#
# run_osworld_multimodel.sh — multi-provider OSWorld actor cold-start /
#                              eval launcher (Claude / Gemini / Qwen3-VL /
#                              GPT-5.x), wired through OpenRouter.
#
# Why a separate launcher:
# ------------------------
# The base ``run_coldstart_actor_osworld.sh`` is hard-wired to GPT-5.x
# semantics (reasoning_effort tiers, OpenAI direct + OpenRouter fall-back,
# the gpt-5.x ``tools`` 400-bug workaround in ``_chat_completion``). Most
# of those switches are no-ops for non-OpenAI families on OpenRouter — the
# driver silently drops ``reasoning_effort`` for non-reasoning models and
# transparently routes slash-prefixed model ids through OpenRouter — so we
# don't fork the python driver. We just front it with a thin wrapper that:
#
#   * Maps a friendly ``--provider`` name to the canonical OpenRouter id.
#   * Applies the correct ``--eval_mode`` tier (Anthropic / Google / Qwen
#     handle ``reasoning_effort=high`` differently or not at all, so we
#     pick a tier that's actually meaningful for that family).
#   * Sets a provider-suffixed ``--output_dir`` so concurrent runs across
#     machines don't trample each other.
#   * Forwards every other CLI arg verbatim to the base launcher.
#
# Quick reference (today's stable IDs on OpenRouter, May-2026):
#
#   claude-sonnet     anthropic/claude-sonnet-4.6     (default Claude tier)
#   claude-opus       anthropic/claude-opus-4.7       (frontier Claude)
#   gemini-pro        google/gemini-2.5-pro           (default Gemini tier)
#   gemini-3-pro      google/gemini-3.1-pro-preview   (preview frontier)
#   qwen3-vl          qwen/qwen3-vl-235b-a22b-instruct(default Qwen3-VL)
#   gpt5              gpt-5.4                         (OpenAI direct, our
#                                                      May-2026 baseline)
#   gpt5-or           openai/gpt-5.4                  (gpt-5.4 via OpenRouter
#                                                      — keeps tools+
#                                                      reasoning_effort)
#
# Usage:
#
#   # ------------------------------------------------------------------
#   # MACHINE A — Claude Sonnet 4.6 over the full 250-task eval
#   # ------------------------------------------------------------------
#   bash cold_start/run_osworld_multimodel.sh \
#       --provider claude-sonnet \
#       --eval_mode high \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
#   # ------------------------------------------------------------------
#   # MACHINE B — Gemini 2.5 Pro
#   # ------------------------------------------------------------------
#   bash cold_start/run_osworld_multimodel.sh \
#       --provider gemini-pro \
#       --eval_mode high \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
#   # ------------------------------------------------------------------
#   # MACHINE C — Qwen3-VL 235B-A22B Instruct
#   # ------------------------------------------------------------------
#   bash cold_start/run_osworld_multimodel.sh \
#       --provider qwen3-vl \
#       --eval_mode high \
#       --task_catalog /workspace/OSWorld/evaluation_examples/test_all.json \
#       --episodes 1 --resume -v
#
# All three runs land in distinct output dirs so you can rsync them back
# to a single host and aggregate with ``cold_start/load_rollouts.py
# --treat-null-as-zero`` for an honest pass@1 number per provider.
#
# Cost / behaviour notes per provider:
#
#   Claude Sonnet 4.6 / Opus 4.7
#     - Tool-calling + vision: rock-solid via OpenRouter.
#     - Does NOT honour ``reasoning_effort`` — silently dropped by the
#       driver. ``--eval_mode {medium,high,max}`` only changes
#       ``temperature`` / ``done_nudge`` / ``max_steps`` for Claude.
#     - Sonnet is the cost-perf sweet spot; Opus 4.7 spends ~5x but
#       gains ~3-5pp on long-tail multi-app tasks (anecdotal).
#
#   Gemini 2.5 Pro / 3.1 Pro Preview
#     - Tool-calling + vision: works, but ~5% of vision calls return
#       ``finish_reason=error`` with empty content (safety / RPC blip).
#       The driver retries on its own fall-back path; no action needed.
#     - Like Claude, ``reasoning_effort`` is dropped.
#     - 3.1-pro-preview is currently in preview rotation — pricing may
#       change without notice.
#
#   Qwen3-VL 235B-A22B Instruct
#     - Tool-calling + vision: works on the *instruct* variant. The
#       *thinking* variant (``qwen/qwen3-vl-235b-a22b-thinking``) is
#       NOT recommended — strict ``tool_choice`` is rejected in thinking
#       mode (``InvalidParameter ... in thinking mode``). The driver's
#       ``_maybe_disable_thinking_kwargs`` helper auto-injects
#       ``enable_thinking=False`` which the instruct provider treats as a
#       no-op.
#     - Smaller variants (``qwen3-vl-32b-instruct``, ``qwen3-vl-8b-*``)
#       have flaky tool_choice support across providers — stick to the
#       235B-a22b-instruct unless you specifically want a small student.
#
#   GPT-5.4 (OpenAI direct)
#     - The ``--api_key`` route forces OpenAI direct so ``gpt-5.4`` is
#       valid. The driver's ``_chat_completion`` already strips
#       ``reasoning_effort`` from any *tool-bearing* call to gpt-5.x on
#       OpenAI direct (workaround for the May-2026 HTTP-400 bug); schema
#       calls still get ``reasoning_effort=high``.
#
# Pre-reqs:
#   - api_keys.py at the workspace root with ``openrouter_api_key`` set
#     (and ``openai_api_key`` if you want the gpt5 provider on the same
#     machine). The base launcher auto-loads it.
#   - The same OSWorld stack as the base launcher (DesktopEnv, Docker
#     image, qcow2 — see ``install/install_osworld.sh``).
#
# Environment variables (advanced):
#   OSWORLD_PROVIDER_OUTPUT_TAG    suffix appended to ``--output_dir``
#                                  (default: the resolved model id with
#                                  ``/`` → ``__`` so it's filesystem-safe).
#   OSWORLD_MULTIMODEL_FORCE_OPENROUTER=1
#                                  Strip any ``--api_key`` from the
#                                  forwarded args so the base launcher
#                                  always picks OpenRouter (only relevant
#                                  if api_keys.py also has openai_api_key).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/run_coldstart_actor_osworld.sh"

if [ ! -x "${BASE_LAUNCHER}" ] && [ ! -f "${BASE_LAUNCHER}" ]; then
    echo "[ERROR] base launcher missing at ${BASE_LAUNCHER}" >&2
    exit 1
fi

# ── Provider → model id table ────────────────────────────────────────────
provider_to_model() {
    case "${1}" in
        claude-sonnet|claude|sonnet)        echo "anthropic/claude-sonnet-4.6" ;;
        claude-opus|opus)                    echo "anthropic/claude-opus-4.7" ;;
        gemini-pro|gemini|gemini-2.5|gemini-2.5-pro)
                                             echo "google/gemini-2.5-pro" ;;
        gemini-3-pro|gemini-3.1|gemini-3.1-pro)
                                             echo "google/gemini-3.1-pro-preview" ;;
        qwen3-vl|qwen|qwen3vl|qwen3-vl-235b)
                                             echo "qwen/qwen3-vl-235b-a22b-instruct" ;;
        gpt5|gpt5.4|gpt-5.4)                 echo "gpt-5.4" ;;
        gpt5-or|gpt5-openrouter)             echo "openai/gpt-5.4" ;;
        *)                                   echo "" ;;
    esac
}

# ── Default eval_mode per provider (only applied if user didn't set one) ─
#
# Default flipped 2026-05-03: ``low`` is the new baseline tier across
# all providers, picked for the cross-model OSWorld teacher comparison
# use case where pass-rate parity (within ±2-3 pp of ``medium``) at
# ~2x cost / wall-clock savings is preferred. Operators chasing the
# leaderboard headline number can still pass ``--eval_mode high`` /
# ``max`` explicitly. For Anthropic / Google / Qwen3-VL on OpenRouter
# the ``reasoning_effort`` knob is silently dropped by the driver, so
# the tier only changes temperature / done_nudge / max_steps for
# those families — but we still pick ``low`` for consistency in
# logged metadata.
provider_default_eval_mode() {
    case "${1}" in
        claude-sonnet|claude|sonnet|claude-opus|opus)
            echo "low" ;;
        gemini-pro|gemini|gemini-2.5|gemini-2.5-pro|gemini-3-pro|gemini-3.1|gemini-3.1-pro)
            echo "low" ;;
        qwen3-vl|qwen|qwen3vl|qwen3-vl-235b)
            echo "low" ;;
        gpt5|gpt5.4|gpt-5.4|gpt5-or|gpt5-openrouter)
            echo "low" ;;
        *)
            echo "low" ;;
    esac
}

# ── Parse our wrapper-only flags out of "$@", forward the rest ──────────
PROVIDER=""
USER_SET_EVAL_MODE=0
USER_SET_MODEL=0
USER_SET_OUTPUT_DIR=0
FORCE_OR="${OSWORLD_MULTIMODEL_FORCE_OPENROUTER:-0}"
FORWARD_ARGS=()

print_help() {
    sed -n '1,120p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    cat <<EOF

Wrapper-only flags:
  --provider <name>         REQUIRED. One of:
                              claude-sonnet | claude-opus
                              gemini-pro | gemini-3-pro
                              qwen3-vl
                              gpt5 | gpt5-or
  --list_providers          Print the provider → model-id table and exit.
  --print_resolved          Print the fully-expanded base-launcher cmd
                            without running it (dry-run).
  --help / -h               This help.

Every other arg is forwarded verbatim to ${BASE_LAUNCHER}.
EOF
}

while [ $# -gt 0 ]; do
    case "${1}" in
        --provider)
            shift; PROVIDER="${1:-}"; shift ;;
        --provider=*)
            PROVIDER="${1#--provider=}"; shift ;;
        --list_providers|--list-providers)
            cat <<EOF
Provider               Model id                                       Default --eval_mode
---------------------  ---------------------------------------------  -------------------
claude-sonnet          anthropic/claude-sonnet-4.6                    low
claude-opus            anthropic/claude-opus-4.7                      low
gemini-pro             google/gemini-2.5-pro                          low
gemini-3-pro           google/gemini-3.1-pro-preview                  low
qwen3-vl               qwen/qwen3-vl-235b-a22b-instruct               low
gpt5                   gpt-5.4 (OpenAI direct via --api_key)          low
gpt5-or                openai/gpt-5.4 (via OpenRouter)                low
EOF
            exit 0 ;;
        --print_resolved|--print-resolved)
            DRY_RUN=1; shift ;;
        --eval_mode|--eval-mode)
            USER_SET_EVAL_MODE=1
            FORWARD_ARGS+=("${1}"); shift
            FORWARD_ARGS+=("${1}"); shift ;;
        --eval_mode=*|--eval-mode=*)
            USER_SET_EVAL_MODE=1
            FORWARD_ARGS+=("${1}"); shift ;;
        --model)
            USER_SET_MODEL=1
            FORWARD_ARGS+=("${1}"); shift
            FORWARD_ARGS+=("${1}"); shift ;;
        --model=*)
            USER_SET_MODEL=1
            FORWARD_ARGS+=("${1}"); shift ;;
        --output_dir|--output-dir)
            USER_SET_OUTPUT_DIR=1
            FORWARD_ARGS+=("${1}"); shift
            FORWARD_ARGS+=("${1}"); shift ;;
        --output_dir=*|--output-dir=*)
            USER_SET_OUTPUT_DIR=1
            FORWARD_ARGS+=("${1}"); shift ;;
        --api_key|--api-key)
            if [ "${FORCE_OR}" = "1" ]; then
                shift; shift
            else
                FORWARD_ARGS+=("${1}"); shift
                FORWARD_ARGS+=("${1}"); shift
            fi ;;
        --api_key=*|--api-key=*)
            if [ "${FORCE_OR}" = "1" ]; then
                shift
            else
                FORWARD_ARGS+=("${1}"); shift
            fi ;;
        --help|-h)
            print_help; exit 0 ;;
        *)
            FORWARD_ARGS+=("${1}"); shift ;;
    esac
done

if [ -z "${PROVIDER}" ]; then
    echo "[ERROR] --provider is required (e.g. --provider claude-sonnet). Run with --help." >&2
    exit 2
fi

MODEL_ID="$(provider_to_model "${PROVIDER}")"
if [ -z "${MODEL_ID}" ]; then
    echo "[ERROR] unknown --provider '${PROVIDER}'. Run --list_providers." >&2
    exit 2
fi

# Inject --model unless the user already set one
if [ "${USER_SET_MODEL}" -eq 0 ]; then
    FORWARD_ARGS+=(--model "${MODEL_ID}")
fi

# Inject --eval_mode unless the user already set one
if [ "${USER_SET_EVAL_MODE}" -eq 0 ]; then
    DEFAULT_TIER="$(provider_default_eval_mode "${PROVIDER}")"
    FORWARD_ARGS+=(--eval_mode "${DEFAULT_TIER}")
fi

# Inject a provider-tagged --output_dir unless the user set one
if [ "${USER_SET_OUTPUT_DIR}" -eq 0 ]; then
    TAG="${OSWORLD_PROVIDER_OUTPUT_TAG:-}"
    if [ -z "${TAG}" ]; then
        # filesystem-safe slug of the model id
        TAG="$(echo "${MODEL_ID}" | tr '/.' '__')"
    fi
    CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    OUT_DIR="${CODEBASE_ROOT}/Cold-start-out-osworld-${TAG}"
    FORWARD_ARGS+=(--output_dir "${OUT_DIR}")
fi

echo "================================================================"
echo "  multimodel launcher"
echo "    provider:       ${PROVIDER}"
echo "    model id:       ${MODEL_ID}"
[ "${USER_SET_EVAL_MODE}" -eq 0 ] && echo "    eval_mode:      ${DEFAULT_TIER:-high}  (default for ${PROVIDER})"
[ "${USER_SET_OUTPUT_DIR}" -eq 0 ] && echo "    output_dir:     ${OUT_DIR}"
echo "    forwarding to:  ${BASE_LAUNCHER}"
echo "================================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "Resolved command:"
    printf '  %q ' "${BASE_LAUNCHER}" "${FORWARD_ARGS[@]}"
    echo
    exit 0
fi

exec "${BASE_LAUNCHER}" "${FORWARD_ARGS[@]}"
