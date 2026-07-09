#!/bin/bash
# =============================================================================
# Wire the local 35B-A3B vLLM endpoint into API_func per-model routing.
# =============================================================================
# Source (don't execute) this file from the same shell that runs the trainer:
#
#     source scripts/use_35b_judge.sh
#     bash scripts/run_2048.sh
#
# Why this exists
# ---------------
# As of 2026-05-03 ``BACKBONE_JUDGE_MODEL`` defaults to
# ``Qwen/Qwen3.5-35B-A3B`` (same weights as the crafter/harness teacher),
# so ``LLMJudgeConfig`` and ``orchestrator.JudgeConfig`` already pick up
# the right model name at process start with no env var.  But
# ``API_func.ask_vllm`` still needs to know WHICH endpoint serves the
# 35B model — without ``VLLM_BASE_URL_MAP`` every 35B call would land
# on the default :8000 endpoint (which serves the 9B actor) and either
# silently return 9B completions or fail with a model-mismatch.  This
# script sets that map.
#
# Assumptions
# -----------
#   1. The 9B actor backbone is already running on :8000 (the trainer
#      auto-launches one via scripts/run_coevolution.py).
#   2. The 35B-A3B server has been (or will be) started via
#      `bash inference/serve_qwen35_35b_a3b.sh` with PORT=8001.
#
# What it does
# ------------
#   * Sets VLLM_BASE_URL_MAP so `API_func._candidate_vllm_urls`
#     dispatches 35B-A3B requests to :8001 while the rest of the stack
#     keeps using :8000 (the actor / skill-bank backbone).
#   * Re-asserts VLM_AGENT_BACKBONE_JUDGE_MODEL=Qwen/Qwen3.5-35B-A3B
#     (already the default; setting it explicitly makes the choice
#     auditable in process env / wandb config dumps).
#
# Methodology caveat
# ------------------
# The 35B judge shares Qwen3.5 pretraining with the 9B actor — i.e. the
# judge is not "off-distribution" relative to the trained model, which
# can mask self-preference bias.  For paper / formal eval runs export
# ``VLM_AGENT_BACKBONE_JUDGE_MODEL=gpt-5.5`` AFTER sourcing this script
# (or skip sourcing entirely and rely on the gpt-5.5 OpenRouter route),
# OR run a 5% disagreement-rate spot-check against gpt-5.5 — see
# implementation_notes/coevolution-cross-domain-integration.md
# §"Judge family bias".
# =============================================================================

# Refuse to run as a script — env vars need to be exported into the
# caller's shell (caller must `source` this file).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: this file must be SOURCED, not executed."
    echo "       Run:   source ${BASH_SOURCE[0]}"
    exit 1
fi

ACTOR_PORT="${ACTOR_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
ACTOR_URL="http://localhost:${ACTOR_PORT}/v1"
JUDGE_URL="http://localhost:${JUDGE_PORT}/v1"

# Re-assert the default explicitly so the choice is captured in any
# wandb run-config / env dump that scrapes the live env.  Without this
# line `BACKBONE_JUDGE_MODEL` would still resolve to 35B-A3B (it's the
# common/models.py default) but the env var wouldn't be set.
export VLM_AGENT_BACKBONE_JUDGE_MODEL="Qwen/Qwen3.5-35B-A3B"
export VLLM_BASE_URL_MAP="Qwen/Qwen3.5-9B=${ACTOR_URL},Qwen/Qwen3.5-35B-A3B=${JUDGE_URL}"

# Optional reachability sanity check — non-fatal so we don't break sourcing.
if command -v curl >/dev/null 2>&1; then
    for url_label in "actor:${ACTOR_URL}" "judge:${JUDGE_URL}"; do
        label="${url_label%%:*}"
        url="${url_label#*:}"
        if curl -s -m 2 -o /dev/null "${url}/models"; then
            echo "[use_35b_judge] reachable: ${label}=${url}"
        else
            echo "[use_35b_judge] NOT reachable: ${label}=${url}  (start it before running the trainer)"
        fi
    done
fi

echo "[use_35b_judge] exported VLM_AGENT_BACKBONE_JUDGE_MODEL=${VLM_AGENT_BACKBONE_JUDGE_MODEL}"
echo "[use_35b_judge] exported VLLM_BASE_URL_MAP=${VLLM_BASE_URL_MAP}"
