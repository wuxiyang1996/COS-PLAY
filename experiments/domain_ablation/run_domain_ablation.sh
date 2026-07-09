#!/usr/bin/env bash
# Unified entry point for ALFWorld, WebShop, and Candy Crush ablations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOMAIN="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain) DOMAIN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --domain {all|alfworld|webshop|candy_crush}"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

run_domain() {
    local domain="$1"
    echo ""
    echo "=============================================================="
    echo "  Domain ablation: ${domain}"
    echo "=============================================================="
    case "${domain}" in
        candy_crush)
            bash "${SCRIPT_DIR}/run_candy_crush_ablation.sh"
            ;;
        webshop)
            bash "${SCRIPT_DIR}/run_webshop_ablation.sh"
            ;;
        alfworld)
            bash "${SCRIPT_DIR}/run_alfworld_ablation.sh"
            ;;
        *)
            echo "Unknown domain: ${domain}" >&2
            exit 2
            ;;
    esac
}

if [[ "${DOMAIN}" == "all" ]]; then
    run_domain candy_crush
    run_domain webshop
    run_domain alfworld
else
    run_domain "${DOMAIN}"
fi

