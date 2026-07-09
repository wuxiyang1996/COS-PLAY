# cold_start/visualwebarena_env.sh — VisualWebArena URL endpoints.
#
# This file is the same shape as the one written by
# ``install/install_visualwebarena_sites.sh``. As of 2026-05-03 the
# install IS complete on this machine — both the Classifieds stack
# (:9980) and a VWA-specific homepage (:4400) are running, so all 200
# tasks in ``cold_start/task_samples/browsergym_visualwebarena_200.txt``
# are runnable end-to-end.
#
# Provenance of the two added containers:
#
#   1. ``classifieds`` + ``classifieds_db`` — brought up by
#      ``install/install_visualwebarena_sites.sh`` (OSClass + MySQL
#      seeded from ``osclass_craigslist.sql``). Listens on :9980.
#
#   2. ``vwa_homepage`` — a manual one-liner running the
#      ``webarena-homepage`` image with VWA's
#      ``environment_docker/webarena-homepage/`` directory mounted at
#      ``/app:ro``. That directory ships with all
#      ``static/input_images/<site>/task_<id>/input_<i>.png`` assets
#      VWA tasks need. Maps host :4400 → container :4399 to avoid
#      colliding with the shared WebArena :4399 homepage.
#
#   Bring-up command (idempotent — re-run safely):
#       VWA_APP=/workspace/visualwebarena_data/vwa-homepage/environment_docker/webarena-homepage
#       docker run -d --name vwa_homepage \
#           -v "$VWA_APP:/app:ro" -p 4400:4399 \
#           --restart unless-stopped \
#           webarena-homepage
#
# Source this from your shell or rely on
# ``cold_start/run_coldstart_actor_browsergym.sh`` to source it
# automatically when ``--tasks browsergym/visualwebarena.*`` is passed.

# ── Required by ``browsergym.visualwebarena`` (VWA_*-prefixed names) ─────
export VWA_HOMEPAGE="http://localhost:4400"
export VWA_SHOPPING="http://localhost:7770"
export VWA_REDDIT="http://localhost:9999"
export VWA_WIKIPEDIA="http://localhost:8888/viewer#wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export VWA_CLASSIFIEDS="http://localhost:9980"
export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"

# ── Judge model for ``llm_fuzzy_match`` / ``llm_ua_match`` ───────────────
# The upstream ``visualwebarena`` package hardcodes the deprecated
# ``gpt-4-1106-preview`` (2024-Q1) which most current OpenAI keys can no
# longer reach. ``install/patch_vwa_judge_model.sh`` rewires those calls
# to read ``VWA_JUDGE_MODEL`` instead; without this override 18/200 tasks
# in the pinned subset would fail at ``env.step()`` when the evaluator
# tries to score string_match / page_image_query refs.
export VWA_JUDGE_MODEL="${VWA_JUDGE_MODEL:-gpt-4o}"

# Optional full-stack reset endpoint (used by VisualWebArenaInstance
# .full_reset() — ours is unset so the upstream code skips the call).
# export VWA_FULL_RESET="http://localhost:7565"

# ── Required by upstream ``visualwebarena`` Python pkg (un-prefixed) ─────
# The upstream module reads these eagerly at import time. Keeping them
# in sync with the VWA_* vars above is what makes ``import
# visualwebarena.browser_env.env_config`` succeed without a KeyError.
export DATASET="visualwebarena"
export HOMEPAGE="${VWA_HOMEPAGE}"
export SHOPPING="${VWA_SHOPPING}"
export REDDIT="${VWA_REDDIT}"
export WIKIPEDIA="${VWA_WIKIPEDIA}"
export CLASSIFIEDS="${VWA_CLASSIFIEDS}"
export CLASSIFIEDS_RESET_TOKEN="${VWA_CLASSIFIEDS_RESET_TOKEN}"
