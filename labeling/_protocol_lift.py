"""Protocol lift — prose → typed-hops.

Implements the design locked in
[`implementation_notes/legacy/protocol-lift-design.md`](../implementation_notes/legacy/protocol-lift-design.md).
The lift transforms cold-start `protocol.steps: List[str]` (natural-language
prose) into `protocol: List[Dict]` carrying:

  * `op`            — abstract verb ∈ §4.1 taxonomy ∪ {"EXEC"}
  * `payload`       — `${slot}` placeholders for bind-time resolution
  * `slot_types`    — slot_name → schema-canonical ontology label
  * `preconditions` — typed list (free-form pass-through for v0)
  * `effects_add`   — per-hop effect predicates mined from success_criteria
  * `effects_del`   — per-hop effect predicates mined from abort_criteria
  * `evidence_role` — passes through from skill.evidence_role
  * `notes`         — verbatim original prose (diffable)
  * `lift_mode`     — "first" | "rescued" | "fallback_exec"  (coverage metric)

The contract roll-up (`SkillContract.effects_add` / `effects_del`) is the
union of per-hop predicate `type` strings.

Pure deterministic. No LLM calls. Idempotent. Run once or N times — the
output is byte-identical given the same inputs.

PLAN-HARNESS §21, harness/README.md §21.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# §4.1 Verb taxonomy (gymv-only, v0).
#
# Empirically validated at 92.5 % coverage on `run_20260430_030637`
# (74 / 80 prose steps). Lemma triggers are matched against the
# subordinator-stripped, downstream-walked prose.
# ---------------------------------------------------------------------------

# Each row: (verb, bucket, lemma_triggers, evidence_role, slot_signature).
# `slot_signature` is a dict[name → schema_canonical ontology] — the lift
# uses it to type-check slot binding at decoration time.
VERB_TABLE: List[Tuple[str, str, FrozenSet[str], str, Dict[str, str]]] = [
    # env-mutating
    ("SELECT", "env", frozenset({"select", "choose", "pick"}), "COMMIT",
        {"target": "selectable_entity"}),
    ("SWAP", "env", frozenset({"swap", "exchange"}), "COMMIT",
        {"lhs": "selectable_entity", "rhs": "selectable_entity"}),
    ("SLIDE", "env", frozenset({"slide", "shift", "swipe"}), "COMMIT",
        {"direction": "enum"}),
    ("MOVE", "env", frozenset({"move", "drift", "navigate", "walk", "step"}),
        "COMMIT",
        {"target": "tracked_entity", "direction": "enum"}),
    ("ROTATE", "env", frozenset({"rotate", "turn", "spin"}), "COMMIT",
        {"target": "tracked_entity", "dir": "enum"}),
    ("DROP", "env", frozenset({"drop", "land"}), "COMMIT",
        {"target": "tracked_entity"}),
    ("PLACE", "env", frozenset({"place", "position", "set", "lock"}), "COMMIT",
        {"target": "tracked_entity", "anchor": "container_entity"}),
    ("APPROACH", "env", frozenset({"approach", "head", "begin"}), "COMMIT",
        {"target": "navigable_region"}),
    ("EXECUTE", "env",
        frozenset({"execute", "perform", "apply", "do", "merge",
                   "compress", "clear"}),
        "COMMIT", {}),
    # gather
    ("READ", "gather", frozenset({"read", "report"}), "GATHER",
        {"target": "goal_indicator"}),
    ("INSPECT", "gather",
        frozenset({"inspect", "examine", "look", "scan", "observe",
                   "identify", "list", "enumerate", "find", "assess"}),
        "GATHER", {"target": "container_entity"}),
    ("TRACK", "gather", frozenset({"track", "monitor"}), "GATHER",
        {"target": "tracked_entity"}),
    # reason
    ("COMPARE", "reason", frozenset({"compare"}), "REASON",
        {"lhs": "any", "rhs": "any"}),
    ("EVALUATE", "reason",
        frozenset({"determine", "evaluate", "decide", "score",
                   "rank", "compute"}),
        "REASON", {"subject": "any", "criterion": "any"}),
    ("SIMULATE", "reason", frozenset({"simulate", "predict"}), "REASON",
        {"move": "any", "base_state": "any"}),
    ("PREFER", "reason", frozenset({"prefer", "favor"}), "REASON",
        {"chosen": "any", "alternatives": "any"}),
    ("PENALIZE", "reason", frozenset({"penalize", "avoid", "discount"}),
        "REASON", {"subject": "any", "criterion": "any"}),
    ("VERIFY", "reason",
        frozenset({"verify", "check", "confirm", "ensure"}),
        "VERIFY", {"predicate": "effect_predicate"}),
    # control
    ("STOP", "control", frozenset({"stop", "abort", "terminate"}), "PASS", {}),
    ("CONTINUE", "control", frozenset({"continue", "resolve"}), "PASS", {}),
    ("KEEP", "control", frozenset({"keep", "maintain", "let"}), "PASS",
        {"invariant": "effect_predicate"}),
]

# Build a fast inverted index lemma → (verb, bucket, role, slots).
_LEMMA_INDEX: Dict[str, Tuple[str, str, str, Dict[str, str]]] = {}
for _verb, _bucket, _lemmas, _role, _slots in VERB_TABLE:
    for _l in _lemmas:
        # Last-write wins on overlap. The §4.1 table is overlap-free by
        # construction; this assertion is a guard against future edits.
        if _l in _LEMMA_INDEX:
            raise AssertionError(
                f"verb taxonomy has overlapping lemma {_l!r}: "
                f"{_LEMMA_INDEX[_l][0]} vs {_verb}"
            )
        _LEMMA_INDEX[_l] = (_verb, _bucket, _role, _slots)

VERB_TAXONOMY: FrozenSet[str] = frozenset(v[0] for v in VERB_TABLE)


# Leading subordinators / connectives / articles / quantifiers. Stripped
# before the first-word match. `during / for / while / unless / until`
# etc. set up a subordinate clause; the real verb is downstream.
_SUBORDINATORS: FrozenSet[str] = frozenset({
    "if", "when", "after", "before", "during", "while", "for",
    "because", "unless", "until", "as", "though", "given",
    "the", "a", "an", "this", "that", "any", "no",
    "with", "without", "in", "on", "of", "by", "to", "at",
    # Quantifiers that commonly lead "Each turn, …" / "Every step, …".
    "each", "every", "all", "some",
    # Adverbs that often lead prose steps (don't carry lemma signal).
    "soon", "then", "now", "also", "again", "still", "always", "never",
    "first", "next", "finally", "early", "late",
})

# Direction tokens — extracted by regex so SLIDE / MOVE / ROTATE can
# bind their direction slot. Includes adverbial variants (`upward`,
# `downward`, …) that are common in cold-start prose.
_DIRECTION_TOKENS: Dict[str, str] = {
    # canonical
    "up": "up", "down": "down", "left": "left", "right": "right",
    "north": "north", "south": "south", "east": "east", "west": "west",
    "cw": "cw", "ccw": "ccw",
    "clockwise": "cw", "counterclockwise": "ccw",
    # adverbial variants normalise to canonical
    "upward": "up", "downward": "down",
    "leftward": "left", "rightward": "right",
    "upwards": "up", "downwards": "down",
    "leftwards": "left", "rightwards": "right",
}

# ---------------------------------------------------------------------------
# §5 Effect predicates (gymv-only, v0).
#
# Predicates are mined from success_criteria / abort_criteria phrases.
# Each row: (predicate, lemma_triggers, default_args).
# ---------------------------------------------------------------------------

EFFECT_PREDICATE_TYPES: Tuple[str, ...] = (
    "entity_value_increased",
    "entity_value_decreased",
    "entity_count_changed",
    "entity_appeared",
    "entity_disappeared",
    "attribute_changed",
    "cumulative_reward_increased",
    "phase_transitioned",
)

# Predicate-trigger ⇒ predicate mapping. Order matters (more specific first):
# cumulative_reward_increased and phase_transitioned are checked before
# entity-typed predicates and the catch-all `attribute_changed` so that
# game-rule outcomes ("merges were applied" → reward, in 2048) win over
# the generic "the board differs" → attribute_changed catch-all.
#
# Triggers are matched literally (lower-cased substring); they should be
# narrow enough that bare phrasings like "or one merge" inside a
# disjunction do NOT trigger reward (which would mean over-claiming).
# See `_phase2_report.md` §3 for the empirical motivation behind each
# Day-4 addition.
_PREDICATE_TRIGGERS: List[Tuple[str, Tuple[str, ...]]] = [
    # cumulative reward
    ("cumulative_reward_increased",
        # Direct reward / score phrasings.
        ("reward increases", "reward improves", "score increases",
         "score improves", "score is higher", "score higher than",
         "scoring", "earn points", "earn score", "earns points",
         "earns score", "points awarded", "reward goes up",
         "score goes up",
         # 2048-style merges always award score; gate on "applied" /
         # "resolved" / "valid" so the disjunctive phrasing
         # "tile movement or one merge" does NOT trigger.
         "valid merges", "valid merge", "merges were applied",
         "merges are applied", "merges applied", "merge resolved",
         "merges resolved", "merges produce", "merge produces",
         # Tetris / candy-crush — line / match clears award score.
         "small line clear", "small line clears",
         "line clear award", "line-clear award",
         "match awards", "match awarded")),
    # phase
    ("phase_transitioned",
        ("game over", "gameover", "phase changes", "phase transitions",
         "game ends",
         # Tetris top-out is the canonical phase transition.
         "top out", "top-out", "topping out", "topped out",
         # Mario-style failure modes the prose actually uses.
         "non-recoverable state")),
    # entity appeared / disappeared
    ("entity_appeared",
        ("appears", "spawns", "is created", "shows up", "becomes visible",
         "is visible", "newly visible", "new terrain", "new section",
         "new piece", "new pieces")),
    ("entity_disappeared",
        ("disappears", "is removed", "vanishes", "becomes invisible",
         "is cleared")),
    # entity count changed
    ("entity_count_changed",
        ("count changes", "fewer", "more of", "decrement", "increment",
         "tile count", "block count",
         "lines clear", "lines cleared", "line cleared",
         "row clears", "rows clear", "row cleared", "rows cleared",
         "candy clears", "candies clear", "match clears", "matches clear",
         # Holes / cavities are tracked by count in tetris prose.
         "hole count", "holes increase", "holes decrease",
         "no lines are cleared", "no line is cleared")),
    # value increased
    ("entity_value_increased",
        ("highest tile", "value increases", "increases by", "grows by",
         "score goes up", "lines increase", "level up",
         # Generic "increased from N to M" used in tetris prose.
         "increases from")),
    # value decreased
    ("entity_value_decreased",
        ("decreases by", "shrinks", "drops by", "value drops",
         # Candy-crush "moves remaining has decreased"; tetris column
         # heights "decreased from N to M".
         "has decreased", "count has decreased", "count decreased",
         "decreases from", "decreased from")),
    # generic attribute changed (last because it's the catch-all)
    ("attribute_changed",
        ("position changes", "moves to", "shifts to", "state changes",
         "attribute changes", "board changes", "differs from",
         "differs by", "different from",
         # Cold-start prose phrasings discovered in Phase-2 corpus.
         "remains the same", "remains approximately", "stays the same",
         "is preserved")),
]


# ---------------------------------------------------------------------------
# Schema index — per-game vocabulary mined from cold-start episodes.
# ---------------------------------------------------------------------------

@dataclass
class GameSchemaIndex:
    """Per-game vocabulary distilled from `metadata.schema_canonical`."""

    game: str
    entity_labels: FrozenSet[str] = field(default_factory=frozenset)
    label_to_ontology: Dict[str, str] = field(default_factory=dict)
    affordances_by_ontology: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    @classmethod
    def empty(cls, game: str) -> "GameSchemaIndex":
        return cls(game=game)


_RX_ENTITY = re.compile(
    r"^([a-zA-Z_]\w*)\[type=([\w\-]+)(?:,\s*([^\]]+))?\]\s*$",
    re.M,
)


def _parse_schema_block_entities(text: str) -> Iterable[Tuple[str, str]]:
    """Yield `(label, ontology)` pairs from a schema_canonical block."""

    for m in _RX_ENTITY.finditer(text):
        attrs = (m.group(3) or "").strip()
        d: Dict[str, str] = {}
        for kv in attrs.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k.strip()] = v.strip()
        label = d.get("label")
        ontology = d.get("ontology")
        if label and ontology:
            yield label, ontology


# Day-5: per-game label whitelists to extend the auto-mined schema
# vocabulary. Cold-start `schema_canonical` blocks are produced by a
# VLM and only enumerate entities the prompt nudged the model toward —
# tetris's schema, for instance, surfaces `board`, `next_pieces`,
# `current_piece` but NOT `holes`, `stack_height`, `filled_cells`,
# even though the lifted protocols' success_criteria reference all of
# them by name (`"Hole count increases from 3 to 4"`).
#
# The Day-4B deterministic schema producers
# (`harness.gym_schema_producer`) DO emit those entities at runtime, so
# the predicate evaluator can decide them. The miss is only on the
# **lift side**: `_first_entity_label` looks the phrase up against the
# schema_index, doesn't find `holes`, and the predicate ends up with
# `args={}` → undecidable at runtime.
#
# Whitelist keyed by ``(corpus, game)``; merged into the auto-mined
# vocabulary. Add new envs as their producers ship.
_SCHEMA_INDEX_LABEL_WHITELIST: Dict[Tuple[str, str], Dict[str, str]] = {
    # corpus, game → label → ontology. Only canonical (producer-emitted)
    # labels listed; the Day-5 word-set matcher binds prose phrases like
    # ``"hole count"`` / ``"no lines are cleared"`` to the canonical
    # label without needing aliases in the whitelist.
    ("env_wrappers", "tetris"): {
        "holes": "goal_indicator",
        "stack_height": "goal_indicator",
        "filled_cells": "goal_indicator",
        "level": "goal_indicator",
        "lines_cleared": "goal_indicator",
        "score": "goal_indicator",
    },
    ("env_wrappers", "twenty_forty_eight"): {
        # The auto-mine already gets these from the cold-start schema;
        # listing them defensively in case a future producer changes
        # output without re-running the schema-index build.
        "highest_tile": "goal_indicator",
        "score": "goal_indicator",
        "empty_cells": "navigable_region",
    },
    ("env_wrappers", "candy_crush"): {
        "score": "goal_indicator",
        "moves_remaining": "goal_indicator",
        "moves remaining": "goal_indicator",
    },
    ("env_wrappers", "super_mario"): {
        "mario": "selectable_entity",
        "scroll_x": "goal_indicator",
        "lives": "goal_indicator",
        "life": "goal_indicator",
    },
}


def build_schema_index_for_game(
    actions_root: Optional[Path],
    *,
    corpus: str,
    game: str,
    max_episodes: int = 3,
    max_steps_per_episode: int = 3,
) -> GameSchemaIndex:
    """Read up to `max_episodes` cold-start episodes and distill the
    per-game schema vocabulary.

    The slot-type-checker uses `entity_labels` + `label_to_ontology` to
    determine whether a prose-extracted entity reference can bind to a
    typed slot. Returns an empty index when `actions_root` is None or
    no episodes exist (the lift then degrades gracefully — slot_types
    are recorded as `"unknown"`, predicates are not type-checked).

    The Day-5 per-game whitelist (``_SCHEMA_INDEX_LABEL_WHITELIST``) is
    folded in on top of whatever the cold-start corpus produced, so
    runtime-only entities the producer emits (tetris ``holes``,
    ``stack_height``, …) bind cleanly during prose-mining without
    waiting for the cold-start labeler to be re-run.
    """

    label_to_ontology: Dict[str, str] = {}
    if actions_root is not None:
        src = actions_root / corpus / game
        if src.exists():
            for ep_path in sorted(src.glob("episode_*.json"))[:max_episodes]:
                try:
                    data = json.loads(ep_path.read_text())
                except Exception:                                  # noqa: BLE001
                    continue
                for step in (data.get("experiences") or [])[:max_steps_per_episode]:
                    md = step.get("metadata") or {}
                    sc = md.get("schema_canonical") or ""
                    for label, ontology in _parse_schema_block_entities(sc):
                        label_to_ontology.setdefault(label, ontology)

    # Day-5: overlay the per-game whitelist. Cold-start labels win on
    # collision — the whitelist is the *fallback* vocabulary, not an
    # override.
    whitelist = _SCHEMA_INDEX_LABEL_WHITELIST.get((corpus, game), {})
    for label, ontology in whitelist.items():
        label_to_ontology.setdefault(label, ontology)

    return GameSchemaIndex(
        game=game,
        entity_labels=frozenset(label_to_ontology.keys()),
        label_to_ontology=dict(label_to_ontology),
        affordances_by_ontology={},  # not yet used; reserved for v1
    )


# ---------------------------------------------------------------------------
# Tokenisation + classifier
# ---------------------------------------------------------------------------

_RX_NON_WORD = re.compile(r"[^a-z0-9\-]+")


def _tokenize(prose: str) -> List[str]:
    """Lowercase ASCII tokenise, drop punctuation, normalise hyphens."""

    raw = prose.lower().replace("’", "'").replace("/", " ")
    raw = _RX_NON_WORD.sub(" ", raw)
    return [t for t in raw.split() if t]


def _strip_subordinators(tokens: List[str]) -> List[str]:
    out = list(tokens)
    while out and out[0] in _SUBORDINATORS:
        out.pop(0)
    return out


def classify_prose_step(
    prose: str,
) -> Tuple[str, str, str, Dict[str, str], str]:
    """Classify one prose step.

    Returns `(verb, bucket, role, slot_signature, lift_mode)` where
    `lift_mode ∈ {"first", "rescued", "fallback_exec"}`.
    """

    tokens = _strip_subordinators(_tokenize(prose))
    if not tokens:
        return ("EXEC", "env", "COMMIT", {}, "fallback_exec")

    head = tokens[0]
    if head in _LEMMA_INDEX:
        verb, bucket, role, slots = _LEMMA_INDEX[head]
        return (verb, bucket, role, dict(slots), "first")

    for tok in tokens[1:]:
        if tok in _LEMMA_INDEX:
            verb, bucket, role, slots = _LEMMA_INDEX[tok]
            return (verb, bucket, role, dict(slots), "rescued")

    return ("EXEC", "env", "COMMIT", {}, "fallback_exec")


# ---------------------------------------------------------------------------
# Slot population — entity references and direction enums.
# ---------------------------------------------------------------------------

def extract_payload_slots(
    prose: str,
    slot_signature: Dict[str, str],
    schema_index: GameSchemaIndex,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build `(payload, slot_types)` for one hop.

    Strategy is shallow in v0:

    * For entity-typed slots (`selectable_entity`, `tracked_entity`,
      `container_entity`, `navigable_region`, `goal_indicator`) we scan
      the prose for any cold-start entity label whose ontology matches
      the slot type. The first match wins.
    * For `enum`-typed slots (direction) we extract the first direction
      token from a fixed vocabulary (§_DIRECTION_TOKENS).
    * Anything we can't bind from the prose becomes a `${slot}`
      placeholder that the actor is expected to resolve at run-time.
    """

    payload: Dict[str, Any] = {}
    slot_types: Dict[str, str] = {}

    if not slot_signature:
        return payload, slot_types

    tokens = _tokenize(prose)
    token_set = set(tokens)

    for slot_name, slot_type in slot_signature.items():
        slot_types[slot_name] = slot_type
        if slot_type == "enum":
            picked: Optional[str] = None
            for t in tokens:
                if t in _DIRECTION_TOKENS:
                    picked = _DIRECTION_TOKENS[t]
                    break
            payload[slot_name] = picked or f"${{{slot_name}}}"
            continue
        if slot_type in {"any", "effect_predicate"}:
            payload[slot_name] = f"${{{slot_name}}}"
            continue
        # Entity-typed slot — look for a label whose ontology matches.
        # Labels are snake_case (`highest_tile`); prose uses spaces
        # (`highest tile`). Compare via tokenised label so multi-word
        # labels work, and accept *all* tokens of the label appearing
        # in the prose so we don't accidentally bind `tile` to
        # `tile_2` when the prose only mentions `tile`.
        match: Optional[str] = None
        for lbl in schema_index.entity_labels:
            ont = schema_index.label_to_ontology.get(lbl)
            if ont != slot_type:
                continue
            lbl_tokens = _tokenize(lbl)
            if lbl_tokens and all(t in token_set for t in lbl_tokens):
                match = lbl
                break
        payload[slot_name] = match or f"${{{slot_name}}}"

    return payload, slot_types


# ---------------------------------------------------------------------------
# Effect mining
# ---------------------------------------------------------------------------

def _phrase_in(haystack: str, needles: Iterable[str]) -> bool:
    h = haystack.lower()
    return any(n in h for n in needles)


def mine_effects(
    success_criteria: List[str],
    abort_criteria: List[str],
    schema_index: GameSchemaIndex,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return `(effects_add, effects_del)` mined from criteria.

    `effects_add` ⇐ success_criteria (the things that must hold for the
    skill to count as having worked); `effects_del` ⇐ abort_criteria
    (the things that must NOT hold). Each predicate row is a dict
    `{type, args}` where `type` is one of EFFECT_PREDICATE_TYPES.
    """

    def _mine(criteria: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for phrase in criteria:
            for predicate, triggers in _PREDICATE_TRIGGERS:
                if not _phrase_in(phrase, triggers):
                    continue
                # Resolve entity-label arg if the phrase mentions one.
                lbl = _first_entity_label(phrase, schema_index)
                args: Dict[str, Any] = {}
                if predicate.startswith("entity_") and lbl:
                    args["entity_label"] = lbl
                if predicate == "phase_transitioned":
                    args["to"] = "gameover"
                key = (predicate, json.dumps(args, sort_keys=True))
                if key in seen:
                    break
                seen.add(key)
                out.append({"type": predicate, "args": args, "from_phrase": phrase})
                break  # one predicate per phrase is enough
        return out

    return _mine(success_criteria), _mine(abort_criteria)


def _first_entity_label(
    phrase: str, schema_index: GameSchemaIndex
) -> Optional[str]:
    """Find the first cold-start entity label mentioned in `phrase`.

    Cold-start entity labels are snake-case (`highest_tile`, `tile_2`,
    `active_piece_S`); cold-start prose uses spaces (`highest tile`).
    We match on the de-snake-cased label so `highest tile` in prose
    finds the `highest_tile` schema entity.

    Day-5 generalisations:

    1. Tolerate singular ↔ plural drift (`"Hole count"` → `holes`).
    2. Word-set match for multi-word labels (`lines_cleared` matches
       ``"no lines are cleared by the placement"`` because every
       word of the label appears in the phrase — substring matching
       wouldn't because of the intervening "are").
    3. Iteration is *sorted* and stable, longest-label-wins on ties so
       producer-canonical labels are preferred (`lines_cleared` over
       bare `lines`, `holes` over `hole`).
    """

    p = phrase.lower()
    p_words = set(_PHRASE_WORD_RX.findall(p))
    candidates: List[str] = []
    for lbl in sorted(schema_index.entity_labels):
        if _label_matches_phrase(lbl, p, p_words):
            candidates.append(lbl)
    if not candidates:
        return None
    # Prefer the longest match so `lines_cleared` wins over `lines`,
    # and `holes` wins over `hole` (which both appear in the tetris
    # whitelist).
    return max(candidates, key=lambda x: (len(x), x))


_PHRASE_WORD_RX = re.compile(r"[a-z0-9]+")


def _label_matches_phrase(label: str, phrase_lower: str, phrase_words: set) -> bool:
    """Return True if any of these is true:

    1. The label (or its underscore→space form) is a substring of the
       phrase.
    2. The label has a singular/plural fold that's a substring.
    3. The label is multi-word (`lines_cleared`, `highest tile`) and
       *every* token (after singular/plural fold) appears as a word in
       the phrase. This handles real cold-start prose like ``"no
       lines are cleared by the placement"`` where the label's words
       are present but not adjacent.
    """

    lbl = label.lower()
    forms = [
        lbl,
        lbl.replace("_", " "),
        lbl.rstrip("s") if lbl.endswith("s") else lbl,
        (lbl.replace("_", " ").rstrip("s") if lbl.endswith("s")
         else lbl.replace("_", " ")),
    ]
    if any(f and f in phrase_lower for f in forms):
        return True
    # Word-set match for multi-word labels.
    label_tokens = [t for t in lbl.replace("_", " ").split() if t]
    if len(label_tokens) <= 1:
        return False
    folded_tokens = []
    for tok in label_tokens:
        if tok.endswith("s") and len(tok) > 2:
            # Both singular and plural forms count as a hit.
            folded_tokens.append({tok, tok.rstrip("s")})
        else:
            folded_tokens.append({tok})
    return all(any(tok in phrase_words for tok in alts) for alts in folded_tokens)


# ---------------------------------------------------------------------------
# The lift itself
# ---------------------------------------------------------------------------

@dataclass
class LiftStats:
    n_hops: int = 0
    n_first: int = 0
    n_rescued: int = 0
    n_fallback_exec: int = 0
    verbs: Dict[str, int] = field(default_factory=dict)
    n_effects_add: int = 0
    n_effects_del: int = 0

    @property
    def fallback_exec_pct(self) -> float:
        return (self.n_fallback_exec / self.n_hops) if self.n_hops else 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "n_hops": self.n_hops,
            "n_first": self.n_first,
            "n_rescued": self.n_rescued,
            "n_fallback_exec": self.n_fallback_exec,
            "fallback_exec_pct": round(self.fallback_exec_pct, 4),
            "verbs": dict(self.verbs),
            "n_effects_add": self.n_effects_add,
            "n_effects_del": self.n_effects_del,
        }


def lift_protocol_to_typed_hops(
    skill: Dict[str, Any],
    *,
    schema_index: GameSchemaIndex,
    stats: Optional[LiftStats] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], List[str], List[str]]:
    """Lift one skill's `protocol` from prose → typed hops.

    Returns `(typed_protocol_or_None, contract_eff_add, contract_eff_del)`.

    `None` means "no lift performed" — either because the protocol is
    already a list-of-dicts (idempotent skip) or because there are no
    prose steps to lift. The caller should leave the row's `protocol`
    field unchanged in that case.

    Mutating the contract roll-up is the caller's responsibility.
    """

    protocol_blob = skill.get("protocol")

    # Idempotency: if `protocol` is already list-of-dicts and every dict
    # carries an `op` field in our taxonomy ∪ {"EXEC"}, leave alone.
    if isinstance(protocol_blob, list):
        if all(
            isinstance(h, dict) and str(h.get("op", "")).upper()
            in (VERB_TAXONOMY | {"EXEC"})
            for h in protocol_blob
        ):
            return None, [], []
        # else fall through — list-of-dicts but un-lifted (e.g. shape-
        # only `_wrap_protocol_steps` output): we re-lift from `notes`.
        prose_steps: List[str] = []
        for hop in protocol_blob:
            if isinstance(hop, dict):
                note = hop.get("notes") or hop.get("note")
                if note:
                    prose_steps.append(str(note))
        success_crit: List[str] = []
        abort_crit: List[str] = []
    elif isinstance(protocol_blob, dict):
        prose_steps = [str(s) for s in (protocol_blob.get("steps") or []) if s]
        success_crit = [str(s) for s in (protocol_blob.get("success_criteria") or []) if s]
        abort_crit = [str(s) for s in (protocol_blob.get("abort_criteria") or []) if s]
    else:
        return None, [], []

    if not prose_steps:
        return None, [], []

    role = (skill.get("evidence_role") or "COMMIT").upper()
    eff_add, eff_del = mine_effects(success_crit, abort_crit, schema_index)

    typed: List[Dict[str, Any]] = []
    for prose in prose_steps:
        verb, bucket, hop_role, slot_sig, lift_mode = classify_prose_step(prose)
        payload, slot_types = extract_payload_slots(prose, slot_sig, schema_index)
        # Per-hop effects: only env-mutating verbs carry effect bodies in
        # v0 (gather / reason / control hops are observational and have
        # no env-side delta to assert). Rolling effects up onto each
        # env-mutating hop matches what the per-step success_fn will
        # actually evaluate against consecutive schemas.
        per_hop_eff_add = list(eff_add) if bucket == "env" else []
        per_hop_eff_del = list(eff_del) if bucket == "env" else []
        # Promote the skill's evidence_role onto the hop unless the verb
        # itself overrides (e.g. INSPECT always means GATHER).
        eff_role = hop_role if hop_role != "PASS" else role
        typed.append({
            "op": verb,
            "payload": payload,
            "slot_types": slot_types,
            "preconditions": [],
            "effects_add": per_hop_eff_add,
            "effects_del": per_hop_eff_del,
            "evidence_role": eff_role,
            "notes": prose,
            "lift_mode": lift_mode,
        })
        if stats is not None:
            stats.n_hops += 1
            if lift_mode == "first":
                stats.n_first += 1
            elif lift_mode == "rescued":
                stats.n_rescued += 1
            else:
                stats.n_fallback_exec += 1
            stats.verbs[verb] = stats.verbs.get(verb, 0) + 1
            stats.n_effects_add += len(per_hop_eff_add)
            stats.n_effects_del += len(per_hop_eff_del)

    contract_add = sorted({e["type"] for h in typed for e in h["effects_add"]})
    contract_del = sorted({e["type"] for h in typed for e in h["effects_del"]})
    return typed, contract_add, contract_del


__all__ = [
    "VERB_TABLE",
    "VERB_TAXONOMY",
    "EFFECT_PREDICATE_TYPES",
    "GameSchemaIndex",
    "LiftStats",
    "build_schema_index_for_game",
    "classify_prose_step",
    "extract_payload_slots",
    "lift_protocol_to_typed_hops",
    "mine_effects",
]
