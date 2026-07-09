"""Stub WebShop Flask server — same templates, fake products, zero install.

Why this exists
---------------
The real WebShop install pulls in pyserini (Lucene/Java 11), spaCy, an
old PyTorch, and a 100 MB Google-Drive dataset.  Before paying that cost
we want to answer the **single make-or-break question** for the bridge:

    "Does Chromium's AXTree (as exposed via BrowserGym) pick up clean
     button / link / textbox / radio roles + bids from WebShop's HTML,
     or does it collapse everything into role='generic'?"

The HTML templates we need to test against live in
``webshop_wrapper/templates/`` (a verbatim copy of
``princeton-nlp/WebShop/web_agent_site/templates/``).  Since AXTree
quality depends entirely on the rendered DOM — not on which products
populate the page — we can serve those templates with a 5-product
in-memory dataset and still get a representative answer.

If this stub passes the smoke (>=4 of 5 expected interactive roles
detected per page, no `generic`-only collapse), then the real
``server.py`` boots the actual WebShop Flask server and everything
downstream (BrowserGym tools registry, anti-thrash, schema heads) just
works.  If it fails, we know to abandon WebShop *before* the half-day
install.

Usage::

    python -m webshop_wrapper.stub_app --port 3000
    # then in another shell:
    python -m webshop_wrapper.smoke_axtree --base-url http://localhost:3000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, redirect, request, url_for

_TEMPLATE_DIR = str(Path(__file__).resolve().parent / "templates")
_STATIC_DIR = str(Path(__file__).resolve().parent / "static")


# --------------------------------------------------------------------------- #
# Fake catalogue
#
# Five products spanning the price range the search-first heuristic in
# ``cold_start/generate_cold_start_actor_browsergym.py`` actually
# exercises (cheap → expensive, with options → without).  Keys mirror
# WebShop's real product schema (``MainImage``, ``Title``, ``Price``,
# ``Rating``, ``options`` dict-of-lists, ``Description``,
# ``BulletPoints``, ``Reviews``) so the templates render unmodified.
# --------------------------------------------------------------------------- #
_PRODUCTS: list[dict[str, Any]] = [
    {
        "asin": "B07X1Y2Z3A",
        "Title": "Stainless Steel Insulated Water Bottle, 32oz, Matte Black",
        "Price": "$19.99",
        "Rating": "4.6 / 5",
        "MainImage": "data:image/svg+xml,%3Csvg/%3E",
        "category": "kitchen",
        "query": "water bottle",
        "product_category": "Bottles",
        "options": {"size": ["20oz", "32oz", "40oz"], "color": ["black", "blue", "silver"]},
        "option_to_image": {},
        "Description": "Double-wall vacuum insulation keeps drinks cold 24h, hot 12h.",
        "BulletPoints": ["BPA-free", "Leak-proof lid", "Fits in car cup holders"],
        "Reviews": [{"title": "Great bottle", "body": "Keeps ice all day."}],
        "Attributes": ["insulated", "stainless steel", "32oz"],
    },
    {
        "asin": "B05K9L8M7N",
        "Title": "Wireless Noise-Cancelling Over-Ear Headphones, 30-hour Battery",
        "Price": "$129.50",
        "Rating": "4.4 / 5",
        "MainImage": "data:image/svg+xml,%3Csvg/%3E",
        "category": "electronics",
        "query": "noise cancelling headphones",
        "product_category": "Headphones",
        "options": {"color": ["black", "white"]},
        "option_to_image": {},
        "Description": "Bluetooth 5.2 wireless headphones with active noise cancellation.",
        "BulletPoints": ["30-hour battery", "Foldable", "Built-in mic"],
        "Reviews": [{"title": "Solid ANC", "body": "Better than my last pair."}],
        "Attributes": ["wireless", "noise cancelling", "over-ear"],
    },
    {
        "asin": "B04A3B2C1D",
        "Title": "Mechanical Keyboard, RGB Backlit, Brown Switches",
        "Price": "$79.00",
        "Rating": "4.7 / 5",
        "MainImage": "data:image/svg+xml,%3Csvg/%3E",
        "category": "electronics",
        "query": "mechanical keyboard",
        "product_category": "Keyboards",
        "options": {"layout": ["TKL", "Full-size"], "switch": ["red", "brown", "blue"]},
        "option_to_image": {},
        "Description": "Hot-swappable mechanical keyboard with per-key RGB.",
        "BulletPoints": ["USB-C detachable cable", "PBT keycaps", "N-key rollover"],
        "Reviews": [{"title": "Tactile", "body": "Brown switches feel great."}],
        "Attributes": ["mechanical", "rgb", "hot-swappable"],
    },
    {
        "asin": "B03E5F6G7H",
        "Title": "Cotton Bath Towels, Set of 4, 600 GSM",
        "Price": "$34.95",
        "Rating": "4.3 / 5",
        "MainImage": "data:image/svg+xml,%3Csvg/%3E",
        "category": "home",
        "query": "bath towels",
        "product_category": "Towels",
        "options": {"color": ["white", "grey", "navy"]},
        "option_to_image": {},
        "Description": "100% cotton, 600 GSM, machine washable.",
        "BulletPoints": ["Set of 4", "Quick-dry", "Lint-free"],
        "Reviews": [{"title": "Soft", "body": "Plush and absorbent."}],
        "Attributes": ["cotton", "600 gsm"],
    },
    {
        "asin": "B02I8J9K0L",
        "Title": "Stainless Steel Water Bottle, 20oz, Bright Blue",
        "Price": "$14.49",
        "Rating": "4.2 / 5",
        "MainImage": "data:image/svg+xml,%3Csvg/%3E",
        "category": "kitchen",
        "query": "water bottle",
        "product_category": "Bottles",
        "options": {"size": ["20oz"], "color": ["blue"]},
        "option_to_image": {},
        "Description": "Single-wall stainless bottle, dishwasher safe.",
        "BulletPoints": ["BPA-free", "Carabiner cap"],
        "Reviews": [{"title": "Decent", "body": "Cheap and works."}],
        "Attributes": ["stainless steel", "20oz"],
    },
]

_PRODUCT_BY_ASIN: dict[str, dict[str, Any]] = {p["asin"]: p for p in _PRODUCTS}

# Five fixed goals (one per product), so smoke runners can hit
# /fixed_0 .. /fixed_4 deterministically.
_GOALS: list[dict[str, Any]] = [
    {
        "asin": p["asin"],
        "category": p["category"],
        "query": p["query"],
        "product_category": p["product_category"],
        "instruction_text": (
            f"i am looking for a {p['query']}, and price lower than 200.00 dollars"
        ),
        "attributes": p["Attributes"],
        "price_upper": 200.0,
        "goal_options": {k: v[0] for k, v in p["options"].items()},
    }
    for p in _PRODUCTS
]


# --------------------------------------------------------------------------- #
# Tiny in-memory search (no Lucene, no rank_bm25 dep) — substring scoring.
# Sufficient because AXTree quality does not depend on result ranking.
# --------------------------------------------------------------------------- #
def _search(keywords: list[str]) -> list[dict[str, Any]]:
    needle = " ".join(keywords).lower()
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in _PRODUCTS:
        haystack = " ".join(
            [p["Title"], p["query"], p["category"], p["product_category"]]
        ).lower()
        score = sum(1 for tok in needle.split() if tok and tok in haystack)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return list(_PRODUCTS)
    return [p for _, p in scored]


# --------------------------------------------------------------------------- #
# Trivial reward: 1.0 if purchased asin matches goal asin, else 0.0.
# Real WebShop has a richer attribute-matching reward (in
# ``web_agent_site/engine/goal.py``); the bridge will call that one in
# full mode.  The stub keeps it simple — reward calibration is not what
# this spike is checking.
# --------------------------------------------------------------------------- #
def _reward(goal: dict[str, Any], purchased_asin: str, options: dict[str, Any]) -> float:
    if purchased_asin != goal["asin"]:
        return 0.0
    matched_opts = sum(
        1 for k, v in goal["goal_options"].items() if options.get(k) == v
    )
    total_opts = max(1, len(goal["goal_options"]))
    return 0.5 + 0.5 * (matched_opts / total_opts)


# --------------------------------------------------------------------------- #
# Flask app — same routes as web_agent_site/app.py so URL-based hooks
# from the real BrowserGym task work unmodified.
# --------------------------------------------------------------------------- #
def create_app() -> Flask:
    Path(_STATIC_DIR).mkdir(exist_ok=True)
    css = Path(_STATIC_DIR) / "style.css"
    if not css.exists():
        css.write_text("/* stub */\n")

    app = Flask(__name__, template_folder=_TEMPLATE_DIR, static_folder=_STATIC_DIR)
    app.url_map.strict_slashes = False

    user_sessions: dict[str, dict[str, Any]] = {}

    def _ensure_session(session_id: str) -> dict[str, Any]:
        if session_id in user_sessions:
            return user_sessions[session_id]
        if session_id.startswith("fixed_"):
            idx = int(session_id.split("_")[-1]) % len(_GOALS)
            goal = _GOALS[idx]
        else:
            goal = _GOALS[0]
        user_sessions[session_id] = {"goal": goal, "done": False}
        return user_sessions[session_id]

    @app.route("/")
    def home():
        return redirect(url_for("index", session_id="abc"))

    @app.route("/<session_id>", methods=["GET", "POST"])
    def index(session_id):
        sess = _ensure_session(session_id)
        if request.method == "POST" and "search_query" in request.form:
            keywords = request.form["search_query"].lower().split(" ")
            return redirect(
                url_for("search_results", session_id=session_id, keywords=keywords, page=1)
            )
        from flask import render_template
        return render_template(
            "search_page.html",
            session_id=session_id,
            instruction_text=sess["goal"]["instruction_text"],
        )

    @app.route(
        "/search_results/<session_id>/<keywords>/<page>",
        methods=["GET", "POST"],
    )
    def search_results(session_id, keywords, page):
        from ast import literal_eval
        from flask import render_template
        sess = _ensure_session(session_id)
        try:
            kw = literal_eval(keywords)
            if isinstance(kw, str):
                kw = kw.split(" ")
        except (ValueError, SyntaxError):
            kw = keywords.split(" ")
        page_i = int(page) if str(page).isdigit() else 1
        results = _search(kw)
        per_page = 10
        products = results[(page_i - 1) * per_page : page_i * per_page]
        return render_template(
            "results_page.html",
            session_id=session_id,
            products=products,
            keywords=kw,
            page=page_i,
            total=len(results),
            instruction_text=sess["goal"]["instruction_text"],
        )

    @app.route(
        "/item_page/<session_id>/<asin>/<keywords>/<page>/<options>",
        methods=["GET", "POST"],
    )
    def item_page(session_id, asin, keywords, page, options):
        from ast import literal_eval
        from flask import render_template
        sess = _ensure_session(session_id)
        try:
            opts = literal_eval(options) if isinstance(options, str) else dict(options)
        except (ValueError, SyntaxError):
            opts = {}
        product_info = dict(_PRODUCT_BY_ASIN.get(asin, _PRODUCTS[0]))
        product_info["goal_instruction"] = sess["goal"]["instruction_text"]
        return render_template(
            "item_page.html",
            session_id=session_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=asin,
            options=opts,
            instruction_text=sess["goal"]["instruction_text"],
            show_attrs=False,
        )

    @app.route(
        "/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>",
        methods=["GET", "POST"],
    )
    def item_sub_page(session_id, asin, keywords, page, sub_page, options):
        from ast import literal_eval
        from flask import render_template
        sess = _ensure_session(session_id)
        try:
            opts = literal_eval(options) if isinstance(options, str) else dict(options)
        except (ValueError, SyntaxError):
            opts = {}
        product_info = dict(_PRODUCT_BY_ASIN.get(asin, _PRODUCTS[0]))
        product_info["goal_instruction"] = sess["goal"]["instruction_text"]
        template_map = {
            "Description": "description_page.html",
            "Features": "features_page.html",
            "Reviews": "review_page.html",
            "Attributes": "attributes_page.html",
        }
        template = template_map.get(sub_page, "description_page.html")
        return render_template(
            template,
            session_id=session_id,
            product_info=product_info,
            keywords=keywords,
            page=page,
            asin=asin,
            options=opts,
            instruction_text=sess["goal"]["instruction_text"],
        )

    @app.route("/done/<session_id>/<asin>/<options>", methods=["GET", "POST"])
    def done(session_id, asin, options):
        from ast import literal_eval
        from flask import render_template
        sess = _ensure_session(session_id)
        try:
            opts = literal_eval(options) if isinstance(options, str) else {}
        except (ValueError, SyntaxError):
            opts = {}
        reward = _reward(sess["goal"], asin, opts)
        sess["done"] = True
        sess["reward"] = reward
        purchased = _PRODUCT_BY_ASIN.get(asin, _PRODUCTS[0])
        return render_template(
            "done_page.html",
            session_id=session_id,
            reward=reward,
            asin=asin,
            options=opts,
            reward_info={"r_match": reward},
            query=purchased["query"],
            category=purchased["category"],
            product_category=purchased["product_category"],
            goal_attrs=sess["goal"]["attributes"],
            purchased_attrs=purchased["Attributes"],
            goal=sess["goal"],
            mturk_code="STUB000000",
        )

    # Side-channel for the bridge: read session reward via JSON without
    # parsing the done page's HTML. Real WebShop does not expose this;
    # the full-mode bridge in ``reward.py`` reaches into the WebShop
    # process via its own user_sessions dict.
    #
    # Auto-creates fixed-* sessions so a probe can read goal text before
    # the first browser navigation has happened (the BrowserGym task's
    # setup() calls this endpoint *before* page.goto).
    @app.route("/__bridge/session/<session_id>")
    def bridge_session(session_id):
        if session_id.startswith("fixed_"):
            _ensure_session(session_id)
        sess = user_sessions.get(session_id, {})
        return json.dumps({
            "done": sess.get("done", False),
            "reward": sess.get("reward", 0.0),
            "goal": sess.get("goal", {}),
        })

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebShop stub Flask server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("FLASK_ENV", "production")
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
