"""Regression tests for the BrowserGym actor's terminal-action support.

Background — the May-1 2026 AssistantBench run produced **0/181**
nonzero rewards because the actor never emitted ``send_msg_to_user``,
the ONLY action AssistantBench uses to score (it extracts the message
argument and matches it against the reference answer with F1 / exact
match). Five places in the actor were missing terminal-action
plumbing:

  1. ``_BROWSERGYM_ACTION_RE`` validator regex — rejected
     ``send_msg_to_user(...)`` and ``report_infeasible(...)``.
  2. ``_build_action_tools`` enum — didn't list either action type.
  3. ``_structured_to_action_string`` — no atype handler for the
     terminal pair, so the structured fallback returned ``None``.
  4. ``_build_candidate_actions`` — never seeded a placeholder, so
     the action LLM literally never saw the option.
  5. ``_ACTOR_SYSTEM_PROMPT`` — no instruction on when / how / what
     payload to emit.

These tests cover all five paths. See
``legacy/visualwebarena/vwa-improvement-plan.md`` §12 for the full
story.
"""
from __future__ import annotations

import sys

import pytest


sys.path.insert(0, "/workspace/Multi-hop-Reasoning-VLM-Agent")


from cold_start.generate_cold_start_actor_browsergym import (  # type: ignore  # noqa: E402
    _ACTOR_SYSTEM_PROMPT,
    _BROWSERGYM_ACTION_RE,
    _REPORT_INFEASIBLE_PLACEHOLDER,
    _SEND_MSG_PLACEHOLDER,
    _build_action_tools,
    _build_candidate_actions,
    _is_information_extraction_task,
    _structured_to_action_string,
)


# ---------------------------------------------------------------------------
# 1. Validator regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,expected",
    [
        # Must accept canonical AssistantBench / WebArena terminal actions
        ('send_msg_to_user("42")', True),
        ('send_msg_to_user("San Francisco")', True),
        ('send_msg_to_user("a, b, c")', True),
        ('send_msg_to_user("multi\\nline\\nanswer")', True),
        ('report_infeasible("page not loadable")', True),
        ('report_infeasible("information not on the available sites")', True),
        # Pre-existing actions still match
        ('click("a17")', True),
        ('fill("a23", "hello")', True),
        ('go_back()', True),
        # Garbage / unknown actions still rejected
        ('say("hi")', False),
        ('stop()', False),
        ('terminate()', False),
        # Empty argument is rejected (regex requires .+ inside parens)
        ('send_msg_to_user()', False),
        ('report_infeasible()', False),
    ],
)
def test_validator_regex_accepts_terminal_actions(action, expected):
    matched = bool(_BROWSERGYM_ACTION_RE.match(action))
    assert matched is expected, (
        f"action={action!r} regex match={matched} expected={expected}"
    )


# ---------------------------------------------------------------------------
# 2. Tool enum exposes the terminal action types
# ---------------------------------------------------------------------------

def test_action_tool_enum_includes_terminal_types():
    tools = _build_action_tools(['click("a1")'])
    assert len(tools) == 1
    fn = tools[0]["function"]
    enum = fn["parameters"]["properties"]["action_type"]["enum"]
    assert "send_msg_to_user" in enum
    assert "report_infeasible" in enum


def test_action_tool_has_answer_param():
    """The structured-fallback path passes ``answer`` to
    ``_structured_to_action_string`` — the function-calling schema must
    expose this slot or the LLM has nowhere to put the answer text
    when it picks ``action_type=send_msg_to_user``."""
    tools = _build_action_tools(['click("a1")'])
    props = tools[0]["function"]["parameters"]["properties"]
    assert "answer" in props
    assert props["answer"]["type"] == "string"


# ---------------------------------------------------------------------------
# 3. Structured-fallback handles both terminal types
# ---------------------------------------------------------------------------

def test_structured_send_msg_with_answer_field():
    out = _structured_to_action_string(
        {"action_type": "send_msg_to_user", "answer": "42"}
    )
    assert out == 'send_msg_to_user("42")'


def test_structured_send_msg_falls_back_to_text_field():
    """Some prompt templates reuse ``text`` for any free-form payload —
    accept it as a synonym for ``answer`` so the LLM doesn't need to
    relearn the field name per task type."""
    out = _structured_to_action_string(
        {"action_type": "send_msg_to_user", "text": "some answer"}
    )
    assert out == 'send_msg_to_user("some answer")'


def test_structured_send_msg_escapes_quotes_and_backslashes():
    out = _structured_to_action_string(
        {"action_type": "send_msg_to_user", "answer": 'a "b" c\\d'}
    )
    # ``"`` -> ``\"``, ``\`` -> ``\\``
    assert out == 'send_msg_to_user("a \\"b\\" c\\\\d")'


def test_structured_send_msg_empty_answer_returns_none():
    """Refusing empty payloads is intentional — better to fall through
    to the ``noop()`` recovery than to emit ``send_msg_to_user("")``,
    which would terminate the episode with a definitively-wrong
    answer."""
    assert _structured_to_action_string(
        {"action_type": "send_msg_to_user"}
    ) is None
    assert _structured_to_action_string(
        {"action_type": "send_msg_to_user", "answer": ""}
    ) is None


def test_structured_report_infeasible_with_answer():
    out = _structured_to_action_string(
        {"action_type": "report_infeasible", "answer": "site offline"}
    )
    assert out == 'report_infeasible("site offline")'


# ---------------------------------------------------------------------------
# 4. _is_information_extraction_task heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task_id,goal,expected",
    [
        # ── Tier 1: AssistantBench task ids ALWAYS trigger ────────
        # (covers the 11 % of AB goals like "Compute X" or "Find the
        # number of Y" that aren't grammatically questions).
        ("assistantbench.test.10", "anything", True),
        ("browsergym/assistantbench.test.42", "x", True),
        ("assistantbench.test.7",
         "Compute the average annual temperature in Arizona", True),
        ("assistantbench.test.99", "Find primer sequences for D13S317", True),

        # ── Tier 2: page-state suites only trigger on QUESTION grammar ──
        # AB-style "Find/Compute" leads on VWA are almost always
        # navigation tasks (find a listing) and MUST NOT trigger.
        ("visualwebarena.92", "Find a TV listing in Maryland", False),
        ("webarena.10", "Add 3 items to the shopping cart", False),
        ("miniwob.click-button.0", "Click the OK button", False),
        # …but a genuinely-question-form VWA goal DOES trigger
        ("visualwebarena.183",
         "What is the price of the cheapest TV?", True),
        ("webarena.42",
         "How many subscribers does /r/news have?", True),

        # ── Tier 3: unknown task-id → fall back to goal grammar ───
        (None, "What is the capital of France?", True),
        (None, "Who founded Microsoft?", True),
        (None, "How many planets are in the solar system?", True),
        (None, "Why did the chicken cross the road?", True),
        (None, "Which city has the largest population?", True),
        # Trailing ``?`` triggers even without a wh- lead
        (None, "The number of states in the US?", True),
        # Imperative leads NO LONGER trigger without a QA task-id
        # (avoids the VWA "Find a listing" collision).
        (None, "Find the GDP of Japan in 2022", False),
        (None, "Tell me the population of Tokyo.", False),
        (None, "Identify the CEO of OpenAI", False),
        (None, "Calculate 17 × 23", False),
        # Side-effect goals never trigger
        (None, "Click the login button and submit", False),
        (None, "Post a comment in the forum", False),
        # Empty / None inputs are safe
        (None, None, False),
        (None, "", False),
    ],
)
def test_information_extraction_heuristic(task_id, goal, expected):
    assert _is_information_extraction_task(task_id=task_id, goal=goal) is expected


# ---------------------------------------------------------------------------
# 5. _build_candidate_actions seeds terminals for QA tasks (and only QA)
# ---------------------------------------------------------------------------

def test_candidates_seed_terminals_for_assistantbench():
    obs = {"goal": "What is the capital of France?"}
    strings, meta = _build_candidate_actions(
        obs=obs, registry=None,
        task_id="assistantbench.test.10",
        goal="What is the capital of France?",
    )
    assert _SEND_MSG_PLACEHOLDER in strings, (
        f"send_msg_to_user placeholder missing from candidates: {strings}"
    )
    assert _REPORT_INFEASIBLE_PLACEHOLDER in strings


def test_candidates_do_not_seed_terminals_for_vwa_navigation_task():
    """Side-effect tasks (clicks, page state) score on URL/DOM, not on
    a returned message — seeding the terminals here would just waste a
    candidate slot and could mislead the agent into stopping early."""
    strings, _ = _build_candidate_actions(
        obs={}, registry=None,
        task_id="visualwebarena.92",
        goal="Find a TV listing in Maryland",
    )
    assert _SEND_MSG_PLACEHOLDER not in strings


def test_candidates_seed_terminals_when_only_goal_is_question():
    """No assistantbench task id but a question-form goal → still
    seed (catches webarena information_seeking and VWA info_seek
    tasks)."""
    strings, _ = _build_candidate_actions(
        obs={}, registry=None,
        task_id="webarena.42",
        goal="How many subscribers does the /r/news subreddit have?",
    )
    assert _SEND_MSG_PLACEHOLDER in strings


def test_candidates_remain_compatible_when_task_id_and_goal_omitted():
    """Backwards-compatibility: callers that don't pass the new
    keyword args should still get a working candidate list (just with
    no terminal seeding)."""
    strings, meta = _build_candidate_actions(obs={}, registry=None)
    # Global navigation actions are still present
    assert "go_back()" in strings
    assert "noop()" in strings
    # No accidental terminal seeding
    assert _SEND_MSG_PLACEHOLDER not in strings


# ---------------------------------------------------------------------------
# 6. System prompt teaches the model when to terminate
# ---------------------------------------------------------------------------

def test_system_prompt_documents_terminal_action():
    p = _ACTOR_SYSTEM_PROMPT
    assert "send_msg_to_user" in p
    assert "report_infeasible" in p
    assert "AssistantBench" in p, (
        "Prompt should call out the benchmark by name so the model "
        "associates the rule with QA-style tasks specifically."
    )
    # It should warn that NOT calling the terminal action means
    # reward=0 — that's the *causal* mental model we want the LLM
    # to internalise.
    assert "max_steps" in p or "reward=0" in p.lower() or "without it" in p.lower()
