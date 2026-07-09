#!/bin/bash
# =============================================================================
# Qwen3.5-35B-A3B inference-only vLLM server
# =============================================================================
# Spins up a single vLLM OpenAI-compatible endpoint for Qwen/Qwen3.5-35B-A3B
# (35B-total / 3B-active MoE) suitable for evaluation, baselines, or as a
# teacher / opponent.  This script does NOT load any LoRA adapters and is
# meant to run *outside* the GRPO training loop (Qwen3.5-9B + LoRA stays the
# trained model — see scripts/run_coevolution.py).
#
# When to use it:
#   • Live judge for skill evaluation + promotion gates (E0 / E1 / E2)
#     during co-evolution training — REQUIRED whenever the skill_eval
#     /orchestrator code paths fire, because both default to
#     ``Qwen/Qwen3.5-35B-A3B`` (the project-wide judge backbone, see
#     ``common/models.py`` and tests/test_backbone_model.py).
#   • Post-training eval: kill the trainer, run this on the freed 8 H200s.
#   • Side-by-side baseline: launch on a separate machine and point your
#     eval harness at http://<host>:8001/v1.
#   • Teacher labeling: use as the oracle for `inference/run_*_eval.sh` by
#     setting `VLLM_BASE_URL` / `MODEL` in those scripts.
#
# Recommended live-training launch (35B judge alongside 9B trainer):
#
#       # GPU 0-3 → 9B actor (training), GPU 4-7 → 35B judge:
#       CUDA_VISIBLE_DEVICES=4,5,6,7 TENSOR_PARALLEL=4 PORT=8001 \
#           bash inference/serve_qwen35_35b_a3b.sh &
#       source scripts/use_35b_judge.sh   # exports VLLM_BASE_URL_MAP
#       bash scripts/run_2048.sh
#
#     The skill-evaluation judge (skill_agents/skill_evaluation) and
#     orchestrator promotion gates (orchestrator.JudgeConfig) both
#     pick up ``Qwen/Qwen3.5-35B-A3B`` from BACKBONE_JUDGE_MODEL by
#     default — no env override required for the model name.  The only
#     env var the trainer NEEDS is ``VLLM_BASE_URL_MAP`` (set by
#     ``scripts/use_35b_judge.sh``) so the 35B requests dispatch to
#     this server's :8001 instead of falling back to the 9B :8000.
#     Routing is implemented in API_func._candidate_vllm_urls; contract
#     is locked by tests/test_api_func_routing.py and
#     tests/test_backbone_model.py::TestSkillEvalJudgeWiring.
#
# GPU layout (defaults, 8x H200 / 141GB):
#   TENSOR_PARALLEL=8, EXPERT_PARALLEL=on  → ~5 GB weights/GPU + huge KV cache.
#   Override TENSOR_PARALLEL=4 if you only have 4 GPUs free during training.
#
# ======================== USAGE ==============================================
#
#   # Default: TP=8 + expert-parallel on GPUs 0-7, port 8001
#   bash inference/serve_qwen35_35b_a3b.sh
#
#   # Run on GPUs 4-7 only (e.g. while GRPO is training on 0-3):
#   GPUS="4,5,6,7" TENSOR_PARALLEL=4 \
#       bash inference/serve_qwen35_35b_a3b.sh
#
#   # Custom port + expose externally:
#   PORT=9000 HOST=0.0.0.0 bash inference/serve_qwen35_35b_a3b.sh
#
#   # FP8 weights (2x weight-memory savings, ~no quality loss):
#   QUANTIZATION=fp8 bash inference/serve_qwen35_35b_a3b.sh
#
#   # Multimodal mode (image+text inputs, vision tower enabled):
#   MULTIMODAL=1 bash inference/serve_qwen35_35b_a3b.sh
#
#   # Disable speculative decoding (debug perf):
#   SPECULATIVE=none bash inference/serve_qwen35_35b_a3b.sh
#
# =============================================================================

set -e

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Conda
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-game-ai-agent}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /workspace/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[serve-35b] Activated conda env: $CONDA_ENV - $(python --version 2>&1)"

# ---------------------------------------------------------------------------
# HF cache (shared with training)
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HUB_CACHE"

# ---------------------------------------------------------------------------
# vLLM 0.20+ workarounds (mirrors trainer/coevolution/vllm_server.py
# _launch_wave so managed and unmanaged servers behave identically).
# ---------------------------------------------------------------------------
# DeepGEMM warmup hard-fails on bf16 MoE/dense weights when the optional
# `deep_gemm` package isn't installed (vLLM unconditionally probes the
# FP8 fast path during engine init).  Disabling DeepGEMM is a no-op for
# bf16 inference and lets the engine come up cleanly.  Override by
# setting VLLM_USE_DEEP_GEMM=1 in the env if you've installed deep_gemm.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"   # 1=enable, 0=disable
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
QUANTIZATION="${QUANTIZATION:-}"          # e.g. fp8, awq_marlin, gptq_marlin
MULTIMODAL="${MULTIMODAL:-0}"             # 1 to keep the vision tower
SPECULATIVE="${SPECULATIVE:-mtp}"         # mtp | none
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# ---------------------------------------------------------------------------
# Build vllm command
# ---------------------------------------------------------------------------
VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-prefix-caching
    --enable-chunked-prefill
    --trust-remote-code
    --dtype auto
)

# MoE expert-parallel reduces per-GPU weight memory: each rank holds a
# disjoint slice of experts instead of replicating them.  Highly
# recommended for Qwen3.5-MoE families when TP > 1.
if [ "$EXPERT_PARALLEL" = "1" ] && [ "$TENSOR_PARALLEL" -gt 1 ]; then
    VLLM_CMD+=( --enable-expert-parallel )
fi

if [ "$MULTIMODAL" != "1" ]; then
    VLLM_CMD+=( --language-model-only )
fi

if [ -n "$REASONING_PARSER" ]; then
    VLLM_CMD+=( --reasoning-parser "$REASONING_PARSER" )
fi

if [ -n "$QUANTIZATION" ]; then
    VLLM_CMD+=( --quantization "$QUANTIZATION" )
fi

if [ "$SPECULATIVE" = "mtp" ] && [ "$NUM_SPEC_TOKENS" -gt 0 ]; then
    SPEC_JSON=$(printf '{"method":"mtp","num_speculative_tokens":%d}' \
                       "$NUM_SPEC_TOKENS")
    VLLM_CMD+=( --speculative_config "$SPEC_JSON" )
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Qwen3.5-35B-A3B  (inference-only)"
echo "============================================"
echo "  Model:           $MODEL"
echo "  Bind:            http://${HOST}:${PORT}/v1"
echo "  GPUs:            $GPUS  (TP=${TENSOR_PARALLEL}, EP=${EXPERT_PARALLEL})"
echo "  Multimodal:      $MULTIMODAL  (0 = text-only / vision tower skipped)"
echo "  Quantization:    ${QUANTIZATION:-bf16}"
echo "  Speculative:     $SPECULATIVE  (${NUM_SPEC_TOKENS} tok)"
echo "  Reasoning:       $REASONING_PARSER"
echo "============================================"
echo ""
echo "[serve-35b] Command:"
printf '  %q ' "${VLLM_CMD[@]}"
echo
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" exec "${VLLM_CMD[@]}"
