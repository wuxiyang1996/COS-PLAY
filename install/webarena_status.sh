#!/usr/bin/env bash
#
# webarena_status.sh — one-shot status report for the WebArena install.
#
# Shows: tarball download progress, loaded Docker images, running
# containers, HTTP health of each canonical port, and whether the
# cold_start/webarena_env.sh file is present and source-able. Safe
# to run while the install scripts are still going.
#
# (Also probes the legacy VWA classifieds container + env file. VWA
# was dropped 2026-05-03 — see legacy/visualwebarena/README.md — so
# these probes will normally show ---- / missing on a clean install.
# They remain here as a courtesy for anyone manually reviving VWA
# from the legacy archive.)
#
# Usage:
#   bash install/webarena_status.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WA_DATA_DIR="${WA_DATA_DIR:-/workspace/webarena_data}"
VWA_DATA_DIR="${VWA_DATA_DIR:-/workspace/visualwebarena_data}"
HOST="${WA_HOST:-localhost}"

echo "================================================================"
echo "  WebArena / VisualWebArena install status"
echo "================================================================"

# ── Tarball download progress ─────────────────────────────────────────────
echo ""
echo "[1/5] Tarball downloads (${WA_DATA_DIR})"
declare -A EXPECTED_GB=(
    ["shopping_final_0712.tar"]=63
    ["shopping_admin_final_0719.tar"]=9
    ["postmill-populated-exposed-withimg.tar"]=50
    ["gitlab-populated-final-port8023.tar"]=72
    ["wikipedia_en_all_maxi_2022-05.zim"]=89
)
TOTAL_HAVE=0; TOTAL_NEED=0
for f in "${!EXPECTED_GB[@]}"; do
    expected=${EXPECTED_GB[$f]}
    TOTAL_NEED=$((TOTAL_NEED + expected))
    if [ -f "${WA_DATA_DIR}/${f}" ]; then
        size_gb=$(du -m "${WA_DATA_DIR}/${f}" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
        size_mb=$(du -m "${WA_DATA_DIR}/${f}" 2>/dev/null | awk '{print $1}')
        TOTAL_HAVE=$(awk -v a=$TOTAL_HAVE -v b=$size_gb 'BEGIN{printf "%.1f", a+b}')
        pct=$(awk -v have=$size_gb -v need=$expected 'BEGIN{printf "%d", (have/need)*100}')
        printf "  %-44s %5s GB / %2s GB  (%s%%)\n" "$f" "$size_gb" "$expected" "$pct"
    else
        printf "  %-44s     -        / %2s GB  (0%%)\n" "$f" "$expected"
    fi
done
echo "  ----"
printf "  %-44s %5s GB / %s GB total\n" "TOTAL" "$TOTAL_HAVE" "$TOTAL_NEED"

# ── Loaded Docker images ──────────────────────────────────────────────────
echo ""
echo "[2/5] Docker images loaded"
for img in shopping_final_0712 shopping_admin_final_0719 postmill-populated-exposed-withimg \
           gitlab-populated-final-port8023 ghcr.io/kiwix/kiwix-serve:3.3.0 webarena-homepage; do
    if docker image inspect "$img" >/dev/null 2>&1; then
        sz=$(docker image inspect "$img" --format '{{.Size}}' 2>/dev/null \
            | awk '{printf "%.1f GB", $1/1024/1024/1024}')
        printf "  %-44s OK   %s\n" "$img" "$sz"
    else
        printf "  %-44s ----\n" "$img"
    fi
done

# ── Running containers ────────────────────────────────────────────────────
echo ""
echo "[3/5] Running containers"
for c in shopping shopping_admin forum gitlab kiwix33 homepage classifieds_app classifieds_db; do
    if docker ps --format '{{.Names}}' | grep -qx "$c"; then
        port=$(docker ps --filter "name=^${c}$" --format '{{.Ports}}' | head -1)
        printf "  %-20s UP    %s\n" "$c" "$port"
    elif docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
        st=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null)
        printf "  %-20s %s\n" "$c" "$st"
    else
        printf "  %-20s ----\n" "$c"
    fi
done

# ── HTTP health probes ────────────────────────────────────────────────────
echo ""
echo "[4/5] HTTP health (curl http://${HOST}:<port>)"
for port_label in "7770:shopping" "7780:shopping_admin" "9999:reddit" "8023:gitlab" \
                  "8888:wikipedia" "4399:wa_homepage" "9980:vwa_classifieds" "3000:wa_map"; do
    port="${port_label%%:*}"
    label="${port_label##*:}"
    code=$(curl --max-time 5 -s -o /dev/null -w "%{http_code}" "http://${HOST}:${port}" 2>/dev/null)
    [ -z "$code" ] && code="000"
    case "$code" in
        2*|3*) status="OK  (HTTP $code)" ;;
        000)   status="DOWN" ;;
        *)     status="HTTP $code" ;;
    esac
    printf "  :%-5s %-22s %s\n" "$port" "$label" "$status"
done

# ── Env files ─────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Env files (used by run_coldstart_actor_browsergym.sh)"
for f in "${CODEBASE_ROOT}/cold_start/webarena_env.sh" \
         "${CODEBASE_ROOT}/cold_start/visualwebarena_env.sh"; do
    if [ -f "$f" ]; then
        nvars=$(grep -c '^export ' "$f" || echo 0)
        printf "  %-50s OK  (%s exports)\n" "${f##*/}" "$nvars"
    else
        printf "  %-50s missing\n" "${f##*/}"
    fi
done

echo ""
echo "================================================================"
