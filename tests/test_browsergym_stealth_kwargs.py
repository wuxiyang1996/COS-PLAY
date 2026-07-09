"""Regression tests for the Playwright stealth + Google-consent
pre-injection helper in
``cold_start/generate_cold_start_actor_browsergym.py``.

Cross-refs:
  * ``legacy/visualwebarena/vwa-improvement-plan.md`` §12.6.1 —
    Google cookie wall flagged as the next-step blocker for
    AssistantBench tasks (all 181 episodes scored 0 because the
    agent never made it past consent.google.com).

The fix lives in ``_build_pw_stealth_kwargs`` + ``_payload_likely_hits_google``
and pre-seeds:
  - a realistic Chromium UA string
  - the ``--disable-blink-features=AutomationControlled`` launch arg
    (defeats Google's /sorry/index CAPTCHA)
  - the ``SOCS`` and ``CONSENT`` cookies on ``.google.com`` (skips
    the consent dialog entirely, only when the payload is likely
    to land on a Google property)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DRV_PATH = Path(__file__).resolve().parents[1] / "cold_start" / "generate_cold_start_actor_browsergym.py"
spec = importlib.util.spec_from_file_location("drv_browsergym", DRV_PATH)
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)


# ---------------------------------------------------------------------------
# _payload_likely_hits_google
# ---------------------------------------------------------------------------

class TestPayloadHitsGoogle:
    """Heuristic should be tight: only fire when the env will load Google."""

    @pytest.mark.parametrize("payload", [
        "browsergym/assistantbench.test.0",
        "browsergym/assistantbench.test.92",
        "browsergym/assistantbench.validation.5",
        "browsergym/assistantbench.dev.10",
    ])
    def test_assistantbench_always_matches(self, payload):
        assert drv._payload_likely_hits_google(payload) is True

    @pytest.mark.parametrize("payload", [
        "https://www.google.com/",
        "https://google.com",
        "http://www.google.co.uk/search?q=test",
        "https://scholar.google.com/scholar?q=foo",
    ])
    def test_openended_google_url_matches(self, payload):
        assert drv._payload_likely_hits_google(payload) is True

    @pytest.mark.parametrize("payload", [
        "browsergym/miniwob.click-button",
        "browsergym/webarena.42",
        "browsergym/workarena.servicenow-form-fill.0",
        "browsergym/openended",
    ])
    def test_other_browsergym_suites_do_not_match(self, payload):
        assert drv._payload_likely_hits_google(payload) is False

    @pytest.mark.parametrize("payload", [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://example.com/",
        "https://semanticscholar.org/author/Yann-LeCun",
        "https://duckduckgo.com/",
    ])
    def test_non_google_urls_do_not_match(self, payload):
        assert drv._payload_likely_hits_google(payload) is False

    def test_empty_payload(self):
        assert drv._payload_likely_hits_google("") is False

    def test_none_payload(self):
        assert drv._payload_likely_hits_google(None) is False  # type: ignore[arg-type]

    def test_url_with_google_substring_in_path_does_not_match(self):
        # "google" appears in the path, NOT in the domain — must not match.
        assert drv._payload_likely_hits_google(
            "https://example.com/about/google-clone"
        ) is False


# ---------------------------------------------------------------------------
# _build_pw_stealth_kwargs
# ---------------------------------------------------------------------------

class TestBuildPwStealthKwargs:
    """The kwargs builder should always set UA + stealth flag, and
    conditionally seed cookies."""

    def test_returns_two_dicts(self):
        cm, ctx = drv._build_pw_stealth_kwargs("browsergym/miniwob.click-button")
        assert isinstance(cm, dict)
        assert isinstance(ctx, dict)

    def test_chromium_kwargs_does_not_pass_args(self):
        # IMPORTANT: BrowserGym's BrowserEnv hard-rejects ``args=`` overrides
        # in pw_chromium_kwargs (see /workspace/BrowserGym/.../core/env.py
        # line ~278: "# will raise an Exception if above args are overriden").
        # Stealth Chromium flags must therefore stay OUT of pw_chromium_kwargs;
        # they are applied via ``_apply_stealth_init_script`` post-reset.
        for payload in (
            "browsergym/miniwob.click-button",
            "browsergym/assistantbench.test.0",
            "browsergym/webarena.42",
            "https://example.com/",
        ):
            cm, _ = drv._build_pw_stealth_kwargs(payload)
            assert "args" not in cm, (
                f"chromium kwargs must not set 'args' (collides with BrowserGym): {cm}"
            )

    def test_stealth_init_script_defined(self):
        # The JS-level stealth replacement for the rejected --disable-blink-
        # features flag must exist as a module constant for
        # _apply_stealth_init_script to inject post-reset.
        assert hasattr(drv, "_STEALTH_INIT_SCRIPT")
        s = drv._STEALTH_INIT_SCRIPT
        assert isinstance(s, str)
        assert "navigator" in s and "webdriver" in s, (
            "stealth init script must redefine navigator.webdriver"
        )

    def test_user_agent_set_universally(self):
        for payload in (
            "browsergym/miniwob.click-button",
            "browsergym/assistantbench.test.0",
            "https://example.com/",
        ):
            _, ctx = drv._build_pw_stealth_kwargs(payload)
            assert "user_agent" in ctx
            assert "Chrome/" in ctx["user_agent"]
            assert "HeadlessChrome" not in ctx["user_agent"]

    def test_cookies_seeded_for_assistantbench(self):
        _, ctx = drv._build_pw_stealth_kwargs("browsergym/assistantbench.test.0")
        assert "storage_state" in ctx
        cookies = ctx["storage_state"]["cookies"]
        names = {c["name"] for c in cookies}
        assert "SOCS" in names
        assert "CONSENT" in names

    def test_cookies_have_correct_domain(self):
        _, ctx = drv._build_pw_stealth_kwargs("browsergym/assistantbench.test.0")
        for cookie in ctx["storage_state"]["cookies"]:
            assert cookie["domain"] == ".google.com"
            assert cookie["path"] == "/"
            assert cookie["expires"] > 1_700_000_000  # ~2024-11; well in the past for any near-term run

    def test_cookies_seeded_for_google_url(self):
        _, ctx = drv._build_pw_stealth_kwargs("https://www.google.com/")
        assert "storage_state" in ctx

    def test_no_cookies_for_miniwob(self):
        _, ctx = drv._build_pw_stealth_kwargs("browsergym/miniwob.click-button")
        assert "storage_state" not in ctx

    def test_no_cookies_for_webarena(self):
        _, ctx = drv._build_pw_stealth_kwargs("browsergym/webarena.42")
        assert "storage_state" not in ctx

    def test_no_cookies_for_workarena(self):
        _, ctx = drv._build_pw_stealth_kwargs("browsergym/workarena.foo.bar")
        assert "storage_state" not in ctx

    def test_no_cookies_for_non_google_url(self):
        _, ctx = drv._build_pw_stealth_kwargs("https://example.com/")
        assert "storage_state" not in ctx

    def test_kwargs_are_independent_copies(self):
        # Each call should return a fresh storage_state list — mutating the
        # output of one call must not pollute future calls.
        _, ctx1 = drv._build_pw_stealth_kwargs("browsergym/assistantbench.test.0")
        ctx1["storage_state"]["cookies"].append({"name": "POISON", "value": "x"})
        _, ctx2 = drv._build_pw_stealth_kwargs("browsergym/assistantbench.test.0")
        names2 = {c["name"] for c in ctx2["storage_state"]["cookies"]}
        assert "POISON" not in names2


# ---------------------------------------------------------------------------
# _CONSENT_ACCEPT_KEYWORDS extension
# ---------------------------------------------------------------------------

class TestConsentKeywordCoverage:
    """Updated keyword list should now cover both accept- and reject-style
    dismissals across major locales."""

    def test_covers_english_reject(self):
        kws = " ".join(drv._CONSENT_ACCEPT_KEYWORDS)
        assert "reject all" in kws
        assert "decline" in kws

    def test_covers_english_accept(self):
        kws = " ".join(drv._CONSENT_ACCEPT_KEYWORDS)
        assert "accept all" in kws
        assert "i agree" in kws

    def test_reject_keywords_listed_before_accept(self):
        # Tiebreak: prefer reject (privacy-preserving) over accept when both
        # are visible. _detect_consent_button_bid uses enumerate-rank, so
        # reject must come first.
        kws = drv._CONSENT_ACCEPT_KEYWORDS
        reject_idx = next(i for i, k in enumerate(kws) if "reject" in k)
        accept_idx = next(i for i, k in enumerate(kws) if "accept" in k)
        assert reject_idx < accept_idx, (
            f"reject keyword at idx={reject_idx} must beat accept at idx={accept_idx}"
        )

    @pytest.mark.parametrize("locale_kw", [
        "tout refuser", "alle ablehnen", "rifiuta tutto",
        "rechazar todo", "全部拒否", "거부",
    ])
    def test_locale_reject_variants(self, locale_kw):
        assert locale_kw in drv._CONSENT_ACCEPT_KEYWORDS, (
            f"locale reject keyword {locale_kw!r} missing"
        )


# ---------------------------------------------------------------------------
# Placeholder-verbatim rejection in _validate_action_string
# ---------------------------------------------------------------------------

class TestPlaceholderRejection:
    """A tired LLM at max_steps sometimes copies the candidate-list hint
    (``send_msg_to_user("<your answer here>")``) verbatim. This must
    fail validation so the harness re-prompts or falls back to a noop,
    rather than submitting the literal placeholder as the final answer
    (guaranteed reward=0 against any real reference)."""

    def test_send_msg_placeholder_rejected(self):
        assert drv._validate_action_string(drv._SEND_MSG_PLACEHOLDER) is False

    def test_report_infeasible_placeholder_rejected(self):
        assert drv._validate_action_string(drv._REPORT_INFEASIBLE_PLACEHOLDER) is False

    @pytest.mark.parametrize("evil_action", [
        'send_msg_to_user("<your answer here>")',
        'send_msg_to_user("<YOUR ANSWER HERE>")',          # case-insensitive
        'send_msg_to_user("<your   answer   here>")',      # whitespace-tolerant
        'send_msg_to_user("<concise answer>")',
        'send_msg_to_user("<final answer>")',
        'report_infeasible("<reason this task cannot be answered>")',
    ])
    def test_all_placeholder_variants_rejected(self, evil_action):
        assert drv._validate_action_string(evil_action) is False, (
            f"placeholder {evil_action!r} should fail validation"
        )

    @pytest.mark.parametrize("good_action", [
        'send_msg_to_user("42")',
        'send_msg_to_user("Yann LeCun")',
        'send_msg_to_user("CrossFit East River, Avea Pilates")',
        'send_msg_to_user("14.2")',
        'report_infeasible("All search engines blocked Playwright with CAPTCHA")',
    ])
    def test_real_answers_still_pass(self, good_action):
        assert drv._validate_action_string(good_action) is True, (
            f"real answer {good_action!r} should pass validation"
        )

    def test_placeholder_inside_otherwise_valid_action(self):
        # Even a syntactically valid action that happens to embed the
        # placeholder must be rejected.
        assert drv._validate_action_string(
            'send_msg_to_user("Answer is <your answer here>")'
        ) is False


# ---------------------------------------------------------------------------
# CAPTCHA fallback prompt guidance
# ---------------------------------------------------------------------------

class TestCaptchaFallbackPrompt:
    """The system prompt's TERMINAL ACTIONS section must steer the agent
    toward direct-URL navigation when search engines anti-bot it,
    rather than the previously suggested ``duckduckgo.com`` (which is
    now also blocked)."""

    def test_prompt_warns_about_ddg_homepage(self):
        prompt = drv._ACTOR_SYSTEM_PROMPT
        assert "static-pages/418" in prompt or "html.duckduckgo.com/html" in prompt, (
            "prompt must mention DDG's anti-bot endpoint(s) so the LLM avoids them"
        )

    def test_prompt_recommends_search_web(self):
        """May-2026 update: ``search_web`` (server-side, intercepted by
        the harness) is now the primary search affordance; the legacy
        ``goto("https://html.duckduckgo.com/html/...")`` fallback is
        still mentioned in the prompt's anti-CAPTCHA paragraph but
        ``search_web`` should be flagged as the *preferred* path."""
        prompt = drv._ACTOR_SYSTEM_PROMPT
        assert "search_web" in prompt, (
            "prompt must mention ``search_web`` as the primary search affordance"
        )
        # Both 'preferred' and 'STRONGLY prefer' are valid wordings;
        # confirm the prompt explicitly recommends search_web over the
        # legacy goto-based search-engine fallbacks.
        assert (
            "STRONGLY prefer" in prompt
            or "preferred" in prompt.lower()
            or "primary" in prompt.lower()
        ), "prompt must explicitly recommend search_web over the goto fallbacks"

    def test_prompt_recommends_direct_site_navigation(self):
        prompt = drv._ACTOR_SYSTEM_PROMPT
        # The named-source examples should be present; we don't pin the
        # exact URL, just confirm the pattern is taught.
        assert "wikipedia" in prompt.lower()
        assert "tripadvisor" in prompt.lower()


# ---------------------------------------------------------------------------
# goto in structured action enum
# ---------------------------------------------------------------------------

class TestGotoStructuredSlot:
    """The ``goto`` action must be reachable via the structured slot so
    that low-effort models can switch search engines without having to
    construct the action_string by hand."""

    def test_goto_in_enum(self):
        tools = drv._build_action_tools(["click(\"a1\")"])
        enum_types = (
            tools[0]["function"]["parameters"]["properties"]["action_type"]["enum"]
        )
        assert "goto" in enum_types, (
            f"goto must be in structured action enum, got: {enum_types}"
        )

    def test_url_param_exists(self):
        tools = drv._build_action_tools(["click(\"a1\")"])
        props = tools[0]["function"]["parameters"]["properties"]
        assert "url" in props
        assert "goto" in props["url"]["description"].lower()

    def test_structured_goto_with_url(self):
        out = drv._structured_to_action_string(
            {"action_type": "goto", "url": "https://html.duckduckgo.com/html/?q=test"}
        )
        assert out == 'goto("https://html.duckduckgo.com/html/?q=test")'

    def test_structured_goto_with_text_fallback(self):
        # ``text`` is the secondary slot for the URL — some LLM call
        # sites reuse it for any string payload.
        out = drv._structured_to_action_string(
            {"action_type": "goto", "text": "https://en.wikipedia.org/"}
        )
        assert out == 'goto("https://en.wikipedia.org/")'

    def test_structured_goto_missing_url(self):
        out = drv._structured_to_action_string({"action_type": "goto"})
        assert out is None

    def test_structured_goto_escapes_quotes(self):
        out = drv._structured_to_action_string(
            {"action_type": "goto", "url": 'https://example.com/?q="hello"'}
        )
        assert out is not None
        assert '\\"' in out
        # Must round-trip through the validator
        assert drv._validate_action_string(out) is True
