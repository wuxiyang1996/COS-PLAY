"""
Step 0 — Predicate naming standard and reliability registry.

Defines a flat string namespace for all predicate types (UI, HUD, world)
and maintains per-predicate reliability scores used to filter unstable
predicates before effect computation.
"""

from __future__ import annotations

from typing import Dict, Optional, Set


# ── Default reliability by namespace ─────────────────────────────────

_DEFAULT_RELIABILITY: Dict[str, float] = {
    "ui": 1.0,
    "hud": 0.9,
    "world": 0.7,
    "event": 1.0,
}


# ── Canonical predicate examples (for documentation / validation) ────

UI_PREDICATES = frozenset({
    "ui.menu_open",
    "ui.inventory_open",
    "ui.in_dialog",
    "ui.map_open",
    "ui.loading",
})

HUD_PREDICATES = frozenset({
    "hud.hp_low",
    "hud.hp_increased",
    "hud.gold_increased",
    "hud.ammo_low",
})

WORLD_PREDICATE_PREFIXES = frozenset({
    "world.door_open",
    "world.enemy_dead",
    "world.item_visible",
})

EVENT_PREDICATE_PREFIXES = frozenset({
    "event.craft_confirm",
    "event.item_pickup",
    "event.quest_complete",
})


def predicate_namespace(pred: str) -> Optional[str]:
    """Extract the namespace prefix from a predicate string.

    >>> predicate_namespace("ui.menu_open")
    'ui'
    >>> predicate_namespace("world.door_open:door3")
    'world'
    >>> predicate_namespace("unknown_thing")
    """
    dot = pred.find(".")
    if dot < 1:
        return None
    return pred[:dot]


class PredicateVocab:
    """Registry of known predicates with per-predicate reliability scores.

    Reliability scores control which predicates are trusted enough for
    effect computation.  Defaults are assigned by namespace; individual
    predicates can be overridden.
    """

    def __init__(self) -> None:
        self._reliability: Dict[str, float] = {}

    # ── Registration ─────────────────────────────────────────────────

    def register(self, pred: str, reliability: Optional[float] = None) -> None:
        """Register a predicate, optionally with a custom reliability."""
        if reliability is not None:
            self._reliability[pred] = reliability
        elif pred not in self._reliability:
            ns = predicate_namespace(pred)
            self._reliability[pred] = _DEFAULT_RELIABILITY.get(ns or "", 0.5)

    def register_many(self, preds: Set[str]) -> None:
        for p in preds:
            self.register(p)

    # ── Queries ──────────────────────────────────────────────────────

    def reliability(self, pred: str) -> float:
        """Return reliability for *pred*, falling back to namespace default."""
        if pred in self._reliability:
            return self._reliability[pred]
        ns = predicate_namespace(pred)
        return _DEFAULT_RELIABILITY.get(ns or "", 0.5)

    def is_reliable(self, pred: str, threshold: float) -> bool:
        return self.reliability(pred) >= threshold

    def filter_reliable(self, preds: Set[str], threshold: float) -> Set[str]:
        """Return subset of *preds* whose reliability >= *threshold*."""
        return {p for p in preds if self.is_reliable(p, threshold)}

    @property
    def all_predicates(self) -> Set[str]:
        return set(self._reliability.keys())

    def is_ui(self, pred: str) -> bool:
        return predicate_namespace(pred) == "ui"

    def to_dict(self) -> Dict[str, float]:
        return dict(self._reliability)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> PredicateVocab:
        vocab = cls()
        vocab._reliability = dict(d)
        return vocab


def normalize_event(raw_event: str) -> str:
    """Normalize a raw UI event string into the ``event.*`` namespace.

    Strips whitespace, lowercases, and replaces spaces with underscores.
    Prepends ``event.`` if not already namespaced.

    >>> normalize_event("Craft Confirm")
    'event.craft_confirm'
    >>> normalize_event("event.item_pickup")
    'event.item_pickup'
    """
    cleaned = raw_event.strip().lower().replace(" ", "_")
    if not cleaned.startswith("event."):
        cleaned = f"event.{cleaned}"
    return cleaned
