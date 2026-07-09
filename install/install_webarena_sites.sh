#!/usr/bin/env bash
#
# install_webarena_sites.sh — bring up the WebArena self-hosted sites.
#
# Downloads the official WebArena Docker image tarballs from CMU's
# metis.lti.cs.cmu.edu mirror (with archive.org fallback), `docker load`s
# them, starts containers on the canonical ports, and writes
# ``cold_start/webarena_env.sh`` exporting ``WA_*`` so the actor pipeline
# auto-discovers the running stack.
#
# Sites brought up:
#   shopping       (Magento OneStopShop)        :7770
#   shopping_admin (Magento CMS / admin)        :7780
#   reddit/forum   (Postmill clone)             :9999
#   gitlab         (populated GitLab CE)        :8023
#   wikipedia      (kiwix-serve + 89 GB .zim)   :8888
#   homepage       (static site listing all)    :4399
#
# NOT installed automatically:
#   map (3000)     OpenStreetMap tile + nominatim + OSRM stack — requires
#                  ~180 GB of OSM data and a separate setup. WebArena's
#                  official path is to spin up a second AMI for it, see
#                  https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#map
#                  Tasks that hit the map subdomain (~80 of 812) will fail
#                  until a Map backend is configured (set WA_MAP=http://<ip>:3000).
#
# Total disk: ~283 GB downloads + ~350 GB after `docker load` (deduplicated).
# Total time: 30 min (1 Gbps) – 6 h (100 Mbps) for the downloads, plus
# 10–20 min to load each tarball + ~10 min for first-boot Magento/GitLab.
#
# Usage:
#   bash install/install_webarena_sites.sh                 # everything
#   bash install/install_webarena_sites.sh shopping        # just one site
#   bash install/install_webarena_sites.sh \
#        shopping shopping_admin reddit gitlab wiki homepage
#
# Env vars / overrides:
#   WA_DATA_DIR       Where tarballs and the kiwix .zim live
#                     (default: /workspace/webarena_data)
#   WA_HOST           Hostname/IP that the *running browser* should use
#                     to reach the sites. Defaults to "localhost", which
#                     is correct when BrowserGym + this stack run on the
#                     same machine. Set to the public IP or DNS otherwise.
#   WA_SKIP_DOWNLOAD  =1 → skip wget; assume tarballs already exist.
#   WA_SKIP_LOAD      =1 → skip docker load; assume images are loaded.
#   WA_PARALLEL_DOWNLOADS  Number of concurrent wget jobs (default: 2).
#                     The CMU mirror caps each connection at ~50–100 MB/s,
#                     so going wider doesn't always help.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WA_DATA_DIR="${WA_DATA_DIR:-/workspace/webarena_data}"
WA_HOST="${WA_HOST:-localhost}"
WA_PARALLEL_DOWNLOADS="${WA_PARALLEL_DOWNLOADS:-2}"
ENV_FILE="${CODEBASE_ROOT}/cold_start/webarena_env.sh"

mkdir -p "$WA_DATA_DIR"

# ── Service catalogue (one row per site) ──────────────────────────────────
# Each row: NAME|IMAGE|TAR|PORT|URL_VAR|CMU_URL|ARCHIVE_FALLBACK
# (The archive.org fallback is an _item_ URL, not a direct file URL —
# we treat it as a hint for the user, not an automated mirror.)
SITES=(
    "shopping|shopping_final_0712|shopping_final_0712.tar|7770|WA_SHOPPING|http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar|https://archive.org/details/webarena-env-shopping-image"
    "shopping_admin|shopping_admin_final_0719|shopping_admin_final_0719.tar|7780|WA_SHOPPING_ADMIN|http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar|https://archive.org/details/webarena-env-shopping-admin-image"
    "reddit|postmill-populated-exposed-withimg|postmill-populated-exposed-withimg.tar|9999|WA_REDDIT|http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar|https://archive.org/details/webarena-env-forum-image"
    "gitlab|gitlab-populated-final-port8023|gitlab-populated-final-port8023.tar|8023|WA_GITLAB|http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar|https://archive.org/details/webarena-env-gitlab-image"
    "wiki|kiwix-serve|wikipedia_en_all_maxi_2022-05.zim|8888|WA_WIKIPEDIA|http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim|https://archive.org/details/webarena-env-wiki-image"
    "homepage|webarena-homepage|webarena-homepage.tar|4399|WA_HOMEPAGE|||"
)

# ── User-selected subset (default: all) ───────────────────────────────────
if [ $# -gt 0 ]; then
    REQUESTED="$*"
else
    REQUESTED="shopping shopping_admin reddit gitlab wiki homepage"
fi

_is_requested() {
    [[ " $REQUESTED " == *" $1 "* ]]
}

_get_field() {
    local row="$1"; local idx="$2"
    echo "$row" | awk -F'|' -v i="$idx" '{print $i}'
}

# ── Step 1: download tarballs (parallel, resumable) ───────────────────────
download_tarballs() {
    [ "${WA_SKIP_DOWNLOAD:-0}" = "1" ] && { echo "[SKIP] downloads (WA_SKIP_DOWNLOAD=1)"; return 0; }
    echo "==> Downloading WebArena image tarballs to ${WA_DATA_DIR}"
    echo "    (resumable — re-running this script picks up where it left off)"
    local pending=()
    for row in "${SITES[@]}"; do
        local name; name="$(_get_field "$row" 1)"
        local tar;  tar="$(_get_field "$row" 3)"
        local url;  url="$(_get_field "$row" 6)"
        _is_requested "$name" || continue
        [ -z "$url" ] && continue   # homepage is built locally, no tar
        local target="${WA_DATA_DIR}/${tar}"
        # Skip if already downloaded and size matches the upstream HEAD
        if [ -f "$target" ]; then
            local local_sz remote_sz
            local_sz=$(stat -c '%s' "$target" 2>/dev/null || echo 0)
            remote_sz=$(curl --max-time 10 -sSLI "$url" 2>/dev/null \
                | tr -d '\r' | awk -F': ' '/^[Cc]ontent-[Ll]ength/ {print $2}' | tail -1)
            if [ -n "$remote_sz" ] && [ "$local_sz" = "$remote_sz" ]; then
                echo "    [skip] ${tar} already complete (${local_sz} bytes)"
                continue
            fi
        fi
        pending+=("$row")
    done
    [ ${#pending[@]} -eq 0 ] && { echo "    Nothing to download."; return 0; }

    # Parallel batch download with wget -c (resumable)
    echo "    Downloading ${#pending[@]} tarball(s), $WA_PARALLEL_DOWNLOADS at a time..."
    printf '%s\n' "${pending[@]}" | \
        xargs -n1 -P "$WA_PARALLEL_DOWNLOADS" -I{} bash -c '
            row="$1"
            tar=$(echo "$row" | awk -F"|" "{print \$3}")
            url=$(echo "$row" | awk -F"|" "{print \$6}")
            target="'"$WA_DATA_DIR"'/$tar"
            echo "    [start] $tar  <- $url"
            wget --progress=dot:giga --tries=10 --timeout=120 -c -O "$target" "$url" 2>&1 \
                | grep --line-buffered -oE "[0-9]+%" | tail -1
            echo "    [done]  $tar  ($(du -h "$target" | cut -f1))"
        ' _ {}
}

# ── Step 2: docker load ───────────────────────────────────────────────────
load_images() {
    [ "${WA_SKIP_LOAD:-0}" = "1" ] && { echo "[SKIP] docker load (WA_SKIP_LOAD=1)"; return 0; }
    echo "==> Loading WebArena Docker images"
    for row in "${SITES[@]}"; do
        local name; name="$(_get_field "$row" 1)"
        local img;  img="$(_get_field "$row" 2)"
        local tar;  tar="$(_get_field "$row" 3)"
        _is_requested "$name" || continue
        [ "$name" = "wiki" ] && {
            # wiki uses an upstream image, not a loadable tar
            docker pull ghcr.io/kiwix/kiwix-serve:3.3.0 >/dev/null 2>&1 \
                && echo "    [pull] ghcr.io/kiwix/kiwix-serve:3.3.0" \
                || echo "    [WARN] kiwix-serve pull failed"
            continue
        }
        [ "$name" = "homepage" ] && continue   # built locally below
        if docker image inspect "$img" >/dev/null 2>&1; then
            echo "    [skip] image $img already loaded"
            continue
        fi
        local target="${WA_DATA_DIR}/${tar}"
        [ -f "$target" ] || { echo "    [WARN] $tar not found in $WA_DATA_DIR (skipping load)"; continue; }
        echo "    [load] $tar -> $img"
        docker load --input "$target" 2>&1 | sed 's/^/      /' || {
            echo "      [ERROR] docker load failed for $tar"
        }
    done
}

# ── Step 3: start containers ──────────────────────────────────────────────
start_containers() {
    echo "==> Starting WebArena containers"
    for row in "${SITES[@]}"; do
        local name; name="$(_get_field "$row" 1)"
        local img;  img="$(_get_field "$row" 2)"
        local port; port="$(_get_field "$row" 4)"
        _is_requested "$name" || continue

        local cname="$name"
        case "$name" in
            reddit)        cname="forum" ;;
            shopping_admin) cname="shopping_admin" ;;
            wiki)          cname="kiwix33" ;;
        esac

        if docker ps -a --format '{{.Names}}' | grep -qx "$cname"; then
            local status; status=$(docker inspect -f '{{.State.Status}}' "$cname" 2>/dev/null)
            if [ "$status" = "running" ]; then
                echo "    [skip] $cname already running"
                continue
            else
                echo "    [start] $cname (was $status)"
                docker start "$cname" >/dev/null
                continue
            fi
        fi

        case "$name" in
            shopping)
                docker run --name shopping -p "${port}:80" -d "$img" >/dev/null
                ;;
            shopping_admin)
                docker run --name shopping_admin -p "${port}:80" -d "$img" >/dev/null
                ;;
            reddit)
                docker run --name forum -p "${port}:80" -d "$img" >/dev/null
                ;;
            gitlab)
                docker run --name gitlab -d -p "${port}:8023" --shm-size=10g \
                    "$img" /opt/gitlab/embedded/bin/runsvdir-start >/dev/null
                ;;
            wiki)
                docker run -d --name=kiwix33 \
                    --volume="${WA_DATA_DIR}:/data" \
                    -p "${port}:80" \
                    ghcr.io/kiwix/kiwix-serve:3.3.0 \
                    wikipedia_en_all_maxi_2022-05.zim >/dev/null
                ;;
            homepage)
                # Build a tiny static homepage from the WebArena repo template.
                _build_homepage_image
                docker run --name homepage -d -p "${port}:4399" webarena-homepage >/dev/null
                ;;
        esac
        echo "    [up]   $cname on :${port}"
    done

    echo ""
    echo "    Waiting 60s for Magento/GitLab boot..."
    sleep 60
}

# ── Step 4: post-boot configuration (Magento + GitLab URLs) ───────────────
configure_services() {
    echo "==> Configuring WebArena services"
    if docker ps --format '{{.Names}}' | grep -qx shopping; then
        echo "    [shopping] setting base URL → http://${WA_HOST}:7770"
        docker exec shopping /var/www/magento2/bin/magento setup:store-config:set \
            --base-url="http://${WA_HOST}:7770" 2>&1 | sed 's/^/      /' || true
        docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \
            "UPDATE core_config_data SET value='http://${WA_HOST}:7770/' WHERE path='web/secure/base_url';" 2>&1 | sed 's/^/      /' || true
        docker exec shopping /var/www/magento2/bin/magento cache:flush 2>&1 | sed 's/^/      /' || true
    fi
    if docker ps --format '{{.Names}}' | grep -qx shopping_admin; then
        echo "    [shopping_admin] setting base URL → http://${WA_HOST}:7780"
        docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set \
            --base-url="http://${WA_HOST}:7780" 2>&1 | sed 's/^/      /' || true
        docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \
            "UPDATE core_config_data SET value='http://${WA_HOST}:7780/' WHERE path='web/secure/base_url';" 2>&1 | sed 's/^/      /' || true
        docker exec shopping_admin php /var/www/magento2/bin/magento config:set \
            admin/security/password_is_forced 0 2>&1 | sed 's/^/      /' || true
        docker exec shopping_admin php /var/www/magento2/bin/magento config:set \
            admin/security/password_lifetime 0 2>&1 | sed 's/^/      /' || true
        docker exec shopping_admin /var/www/magento2/bin/magento cache:flush 2>&1 | sed 's/^/      /' || true
    fi
    if docker ps --format '{{.Names}}' | grep -qx gitlab; then
        echo "    [gitlab] setting external_url → http://${WA_HOST}:8023 (this can take a few minutes)"
        docker exec gitlab sed -i "s|^external_url.*|external_url 'http://${WA_HOST}:8023'|" \
            /etc/gitlab/gitlab.rb 2>&1 | sed 's/^/      /' || true
        docker exec gitlab gitlab-ctl reconfigure 2>&1 | tail -3 | sed 's/^/      /' || true
    fi
}

_build_homepage_image() {
    if docker image inspect webarena-homepage >/dev/null 2>&1; then
        return 0
    fi
    local tmp; tmp=$(mktemp -d)
    cat > "${tmp}/Dockerfile" <<'DOCKER'
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask
COPY index.html /app/templates/index.html
COPY app.py /app/app.py
EXPOSE 4399
CMD ["python", "/app/app.py"]
DOCKER
    cat > "${tmp}/app.py" <<'PYEOF'
from flask import Flask, render_template
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4399)
PYEOF
    cat > "${tmp}/index.html" <<HTML
<!doctype html>
<html><head><meta charset="utf-8"><title>WebArena Homepage</title>
<style>body{font-family:sans-serif;margin:2em;}a{display:block;margin:.5em 0;}</style>
</head><body>
<h1>WebArena Sites</h1>
<a href="http://${WA_HOST}:7770">Shopping (OneStopShop)</a>
<a href="http://${WA_HOST}:7780">Shopping Admin (CMS)</a>
<a href="http://${WA_HOST}:9999">Reddit / Forum</a>
<a href="http://${WA_HOST}:8023">GitLab</a>
<a href="http://${WA_HOST}:8888">Wikipedia</a>
<a href="http://${WA_HOST}:3000">Map (not auto-installed)</a>
</body></html>
HTML
    echo "    [build] webarena-homepage from ${tmp}"
    docker build -t webarena-homepage "$tmp" >/dev/null 2>&1
    rm -rf "$tmp"
}

# ── Step 5: write env file ────────────────────────────────────────────────
write_env_file() {
    echo "==> Writing $ENV_FILE"
    cat > "$ENV_FILE" <<EOF
# Auto-generated by install/install_webarena_sites.sh on $(date)
# Source this file to activate the WebArena URL endpoints, e.g.:
#   source $ENV_FILE
# The actor pipeline launcher does this automatically when --tasks
# browsergym/webarena.* is passed.
export WA_HOMEPAGE="http://${WA_HOST}:4399"
export WA_SHOPPING="http://${WA_HOST}:7770"
export WA_SHOPPING_ADMIN="http://${WA_HOST}:7780"
export WA_REDDIT="http://${WA_HOST}:9999"
export WA_GITLAB="http://${WA_HOST}:8023"
export WA_WIKIPEDIA="http://${WA_HOST}:8888/viewer#wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
# Map is not auto-installed. Set WA_MAP after standing up your own
# OpenStreetMap + Nominatim + OSRM stack (see README), e.g.:
# export WA_MAP="http://<map-host>:3000"
export WA_MAP="\${WA_MAP:-http://${WA_HOST}:3000}"
EOF
    echo "    Done. ${ENV_FILE}"
}

# ── Step 6: smoke test ────────────────────────────────────────────────────
smoke_test() {
    echo "==> Smoke testing WebArena endpoints (HTTP HEAD)"
    for row in "${SITES[@]}"; do
        local name; name="$(_get_field "$row" 1)"
        local port; port="$(_get_field "$row" 4)"
        _is_requested "$name" || continue
        local code
        code=$(curl --max-time 15 -s -o /dev/null -w "%{http_code}" "http://${WA_HOST}:${port}" 2>&1 || echo "000")
        printf "    %-15s :%-5s -> HTTP %s\n" "$name" "$port" "$code"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  WebArena self-hosted sites installer"
echo "================================================================"
echo "  Data dir:   $WA_DATA_DIR"
echo "  Host:       $WA_HOST  (the URL the BrowserGym agent will hit)"
echo "  Sites:      $REQUESTED"
echo "  Disk free:  $(df -h "$WA_DATA_DIR" | awk 'NR==2{print $4}')"
echo "================================================================"

download_tarballs
load_images
start_containers
configure_services
write_env_file
smoke_test

echo ""
echo "================================================================"
echo "  WebArena install complete"
echo "================================================================"
echo "  Activate with:    source ${ENV_FILE}"
echo "  Then run tasks:   bash cold_start/run_coldstart_actor_browsergym.sh \\"
echo "                        --tasks browsergym/webarena.0 \\"
echo "                        --episodes 1 --max_steps 12 --save_frames -v"
echo "================================================================"
