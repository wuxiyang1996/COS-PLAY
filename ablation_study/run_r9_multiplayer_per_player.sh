#!/usr/bin/env bash
# ======================================================================
#  R9 multiplayer per-player donor sweep (avalon × per_role + diplomacy
#  × per_power, both vs gpt-5.4 opponents).  Uses the leave-one-out
#  donor banks (shared cross-domain skill bank).
#
#  Two vLLM servers (one per game) on separate GPUs run in parallel,
#  and the two eval clients run in parallel.  Total wall-clock target
#  ≈ 90-110 minutes.
#
#  Env overrides:
#    AVALON_GPU       (default 0)
#    AVALON_PORT      (default 8040)
#    DIPL_GPU         (default 1)
#    DIPL_PORT        (default 8041)
#    AVALON_EPISODES  (default 50)   5 roles × 10 ep
#    DIPL_EPISODES    (default 70)   7 powers × 10 ep
#    OPPONENT_MODEL   (default gpt-5.4)
#    TEMPERATURE      (default 0.4)
#    SEED             (default 42)
#    OUTPUT_BASE      (default ablation_study/output/r9_cross_game/per_player_<ts>)
# ======================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate game-ai-agent

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── OpenRouter API key for gpt-5.4 opponent ────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f /workspace/keys.py ]; then
    KEY="$(python3 -c "
import re,sys
with open('/workspace/keys.py') as f:
    for line in f:
        m=re.search(r'(openrouter_api_key|open_router_api_key|OPENROUTER_API_KEY)\\s*=\\s*[\"\\'](.*?)[\"\\']', line)
        if m: print(m.group(2)); sys.exit(0)
" 2>/dev/null)"
    [ -n "${KEY}" ] && export OPENROUTER_API_KEY="${KEY}"
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "[r9-pp] WARN: OPENROUTER_API_KEY not set — gpt-5.4 opponent will fail" >&2
fi

# ── Config ────────────────────────────────────────────────────────────
AVALON_GPU="${AVALON_GPU:-0}"
AVALON_PORT="${AVALON_PORT:-8040}"
DIPL_GPU="${DIPL_GPU:-1}"
DIPL_PORT="${DIPL_PORT:-8041}"
AVALON_EPISODES="${AVALON_EPISODES:-50}"
DIPL_EPISODES="${DIPL_EPISODES:-70}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
TEMPERATURE="${TEMPERATURE:-0.4}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"

AVALON_ADAPTER="${PROJECT_ROOT}/runs/Qwen3-8B_avalon_20260322_200424/best/adapters/decision/action_taking"
DIPL_ADAPTER="${PROJECT_ROOT}/runs/Qwen3-8B_diplomacy_20260322_234548/best/adapters/decision/action_taking"
AVALON_LORA_NAME="qwen3-8b-avalon-best"
DIPL_LORA_NAME="qwen3-8b-diplomacy-best"

DONOR_DIR="${PROJECT_ROOT}/ablation_study/output/r9_cross_game/donor_banks"
AVALON_BANK="${DONOR_DIR}/donor_avalon.jsonl"
DIPL_BANK="${DONOR_DIR}/donor_diplomacy.jsonl"

for p in "${AVALON_ADAPTER}" "${DIPL_ADAPTER}" "${AVALON_BANK}" "${DIPL_BANK}"; do
    if [ ! -e "${p}" ]; then echo "[r9-pp] ERROR: missing ${p}" >&2; exit 1; fi
done

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/ablation_study/output/r9_cross_game/per_player_${TS}}"
mkdir -p "${OUTPUT_BASE}/avalon_donor" "${OUTPUT_BASE}/diplomacy_donor"
TOP_LOG="${OUTPUT_BASE}/run.log"
exec > >(tee -a "${TOP_LOG}") 2>&1

echo "══════════════════════════════════════════════════════════════"
echo "  R9 multiplayer per-player donor sweep — ${TS}"
echo "══════════════════════════════════════════════════════════════"
echo "  Avalon:    GPU ${AVALON_GPU}  port ${AVALON_PORT}  ep=${AVALON_EPISODES} (5 role × 10)  vs ${OPPONENT_MODEL}"
echo "  Diplomacy: GPU ${DIPL_GPU}    port ${DIPL_PORT}    ep=${DIPL_EPISODES} (7 power × 10) vs ${OPPONENT_MODEL}"
echo "  Donor bank: ${DONOR_DIR}/donor_{avalon,diplomacy}.jsonl"
echo "  Temperature ${TEMPERATURE}, seed ${SEED}"
echo "  Output:    ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"

# ── Launch vLLM servers ────────────────────────────────────────────────
launch_vllm() {
    local gpu="$1"; local port="$2"; local lora_name="$3"; local adapter="$4"; local logfile="$5"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        python -m vllm.entrypoints.openai.api_server \
            --model "${BASE_MODEL}" \
            --host 127.0.0.1 \
            --port "${port}" \
            --tensor-parallel-size 1 \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.85 \
            --dtype auto \
            --trust-remote-code \
            --enable-lora \
            --lora-modules "${lora_name}=${adapter}" \
            --max-lora-rank 16 \
            > "${logfile}" 2>&1 &
    echo $!
}

wait_health() {
    local port="$1"; local pid="$2"; local name="$3"
    local MAX_WAIT=600 WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "[r9-pp] ${name} vLLM ready (waited ${WAITED}s) pid=${pid}"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[r9-pp] ERROR: ${name} vLLM exited unexpectedly"
            return 1
        fi
        sleep 5; WAITED=$((WAITED + 5))
    done
    echo "[r9-pp] ERROR: ${name} vLLM did not become healthy in ${MAX_WAIT}s"
    return 1
}

echo "[r9-pp] $(date -u +%FT%TZ) ▶ Launching vLLM (avalon + diplomacy) in parallel..."
AV_PID="$(launch_vllm "${AVALON_GPU}" "${AVALON_PORT}" "${AVALON_LORA_NAME}" "${AVALON_ADAPTER}" "${OUTPUT_BASE}/avalon_donor/vllm.log")"
DI_PID="$(launch_vllm "${DIPL_GPU}"    "${DIPL_PORT}"    "${DIPL_LORA_NAME}"    "${DIPL_ADAPTER}"    "${OUTPUT_BASE}/diplomacy_donor/vllm.log")"

cleanup() {
    set +e
    for pid in "${AV_PID:-}" "${DI_PID:-}" "${AV_EVAL_PID:-}" "${DI_EVAL_PID:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill -INT "${pid}" 2>/dev/null || true
        fi
    done
    sleep 5
    for pid in "${AV_PID:-}" "${DI_PID:-}" "${AV_EVAL_PID:-}" "${DI_EVAL_PID:-}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

wait_health "${AVALON_PORT}" "${AV_PID}" "avalon" || { echo "[r9-pp] FATAL: avalon vLLM"; exit 1; }
wait_health "${DIPL_PORT}"   "${DI_PID}" "diplomacy" || { echo "[r9-pp] FATAL: diplomacy vLLM"; exit 1; }

# ── Launch eval clients in parallel ────────────────────────────────────
echo
echo "[r9-pp] $(date -u +%FT%TZ) ▶ Launching evals in parallel..."

# Avalon: per-role rotation, 50 ep × gpt-5.4
(
    export VLLM_BASE_URL="http://127.0.0.1:${AVALON_PORT}/v1"
    export VLLM_API_KEY="EMPTY"
    python -m scripts.run_qwen3_avalon_matched \
        --model "${AVALON_LORA_NAME}" \
        --episodes "${AVALON_EPISODES}" \
        --per_role \
        --opponent_model "${OPPONENT_MODEL}" \
        --temperature "${TEMPERATURE}" \
        --seed "${SEED}" \
        --bank "${AVALON_BANK}" \
        --output_dir "${OUTPUT_BASE}/avalon_donor/eval_out" \
        --verbose \
        > "${OUTPUT_BASE}/avalon_donor/cell.log" 2>&1
    echo $? > "${OUTPUT_BASE}/avalon_donor/exit_code"
) &
AV_EVAL_PID=$!
echo "[r9-pp] avalon eval pid=${AV_EVAL_PID}"

# Diplomacy: per-power cycling, 70 ep × gpt-5.4
(
    export VLLM_BASE_URL="http://127.0.0.1:${DIPL_PORT}/v1"
    export VLLM_API_KEY="EMPTY"
    python -m scripts.run_diplomacy_discrete_eval \
        --model "${DIPL_LORA_NAME}" \
        --opponent_model "${OPPONENT_MODEL}" \
        --episodes "${DIPL_EPISODES}" \
        --per_power \
        --temperature "${TEMPERATURE}" \
        --seed "${SEED}" \
        --bank "${DIPL_BANK}" \
        --unchosen_strategy hold \
        --output_dir "${OUTPUT_BASE}/diplomacy_donor/eval_out" \
        --verbose \
        > "${OUTPUT_BASE}/diplomacy_donor/cell.log" 2>&1
    echo $? > "${OUTPUT_BASE}/diplomacy_donor/exit_code"
) &
DI_EVAL_PID=$!
echo "[r9-pp] diplomacy eval pid=${DI_EVAL_PID}"

# ── Wait for both evals ────────────────────────────────────────────────
wait "${AV_EVAL_PID}" || true
AV_RC="$(cat "${OUTPUT_BASE}/avalon_donor/exit_code" 2>/dev/null || echo -1)"
echo "[r9-pp] $(date -u +%FT%TZ) ✓ avalon eval done (exit ${AV_RC})"

wait "${DI_EVAL_PID}" || true
DI_RC="$(cat "${OUTPUT_BASE}/diplomacy_donor/exit_code" 2>/dev/null || echo -1)"
echo "[r9-pp] $(date -u +%FT%TZ) ✓ diplomacy eval done (exit ${DI_RC})"

# ── Aggregate per-role / per-power stats ───────────────────────────────
echo
echo "[r9-pp] ▶ Aggregating..."
python3 - <<EOF
import json, glob, statistics, math, os
from collections import defaultdict

OUT = "${OUTPUT_BASE}"

def stats(vals):
    if not vals: return None
    n = len(vals); m = statistics.mean(vals)
    s = statistics.stdev(vals) if n>1 else 0
    ci = 1.96*s/math.sqrt(n) if n>1 else 0
    return dict(n=n, mean=m, std=s, ci95=ci, min=min(vals), max=max(vals))

# --- Avalon per-role -----------------------------------------------------
av_eps = sorted(glob.glob(f"{OUT}/avalon_donor/eval_out/avalon/*/episode_*.json"))
av_by_role = defaultdict(list)
av_wins = defaultdict(int)
for p in av_eps:
    d = json.load(open(p))
    m = d.get("metadata") or {}
    role = m.get("role_name", "?")
    side = m.get("role_side", "?")
    gv = m.get("good_victory")
    won = (gv is True and side == "good") or (gv is False and side == "evil")
    av_by_role[role].append(dict(side=side, gv=gv, won=won, reward=m.get("total_reward")))
    if won: av_wins[role] += 1

av_table = {}
for role, eps in av_by_role.items():
    n = len(eps)
    wr = sum(1 for e in eps if e["won"]) / n if n else 0
    s = stats([e["reward"] for e in eps if isinstance(e["reward"], (int,float))])
    av_table[role] = dict(n=n, win_rate=wr, reward=s)

# --- Diplomacy per-power -------------------------------------------------
di_eps = sorted(glob.glob(f"{OUT}/diplomacy_donor/eval_out/diplomacy/*/episode_*.json"))
di_by_power = defaultdict(list)
for p in di_eps:
    d = json.load(open(p))
    m = d.get("metadata") or {}
    pw = m.get("controlled_power")
    fsc = m.get("final_sc_rewards") or {}
    if pw and pw in fsc:
        raw = fsc[pw] * 18.0
        elim = raw < 0
        di_by_power[pw].append(dict(sc=0.0 if elim else raw, elim=elim))

di_table = {}
for pw, eps in di_by_power.items():
    s = stats([e["sc"] for e in eps])
    elim_n = sum(1 for e in eps if e["elim"])
    di_table[pw] = dict(n=len(eps), sc=s, elim_n=elim_n)

result = dict(avalon=av_table, diplomacy=di_table,
              n_avalon=len(av_eps), n_diplomacy=len(di_eps))
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

# Markdown
md = ["# R9 Donor — Per-Role / Per-Power (vs gpt-5.4)"]
md += ["", "## Avalon — per role (donor bank = union of other 5 games' best banks)"]
md += ["", "| Role | n | win_rate | mean reward ± 95% CI | min · max |"]
md += ["|---|---:|---:|---|---|"]
for role in sorted(av_table):
    r = av_table[role]
    s = r["reward"] or {}
    mean_s = f"{s.get('mean',0):.2f} ± {s.get('ci95',0):.2f}" if s else "n/a"
    mm = f"{s.get('min',0):.0f} · {s.get('max',0):.0f}" if s else "n/a"
    md += [f"| {role} | {r['n']} | {r['win_rate']:.3f} | {mean_s} | {mm} |"]
md += ["", "## Diplomacy — per power (raw SC count, max=18, elim→0)"]
md += ["", "| Power | n | mean SC ± 95% CI | min · max | elim/n |"]
md += ["|---|---:|---|---|---:|"]
POWERS = ["AUSTRIA","ENGLAND","FRANCE","GERMANY","ITALY","RUSSIA","TURKEY"]
for pw in POWERS:
    if pw not in di_table: continue
    r = di_table[pw]; s = r["sc"] or {}
    mean_s = f"{s.get('mean',0):.2f} ± {s.get('ci95',0):.2f}" if s else "n/a"
    mm = f"{s.get('min',0):.0f} · {s.get('max',0):.0f}" if s else "n/a"
    md += [f"| {pw} | {r['n']} | {mean_s} | {mm} | {r['elim_n']}/{r['n']} |"]
with open(f"{OUT}/summary.md", "w") as f:
    f.write("\n".join(md) + "\n")

print("Aggregation written to:", f"{OUT}/summary.json", "+ summary.md")
print()
print("\n".join(md))
EOF

echo
echo "[r9-pp] $(date -u +%FT%TZ) ✓ ALL DONE. Output: ${OUTPUT_BASE}"
echo "  avalon exit:    ${AV_RC}"
echo "  diplomacy exit: ${DI_RC}"
