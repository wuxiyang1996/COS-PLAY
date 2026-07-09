#!/usr/bin/env bash
# Candy Crush domain-ablation entry point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-runs/domain_ablation/candy_crush}"
LOG_DIR="${LOG_DIR:-logs/domain_ablation/candy_crush}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

RUN_ROOT="${RUN_ROOT}" LOG_DIR="${LOG_DIR}" \
    bash scripts/run_candy_crush_contract_ablation.sh

