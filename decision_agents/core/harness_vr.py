"""Visual-reasoning harness — read-only image, scratchpad-mutating ops.

The image is fixed for the episode; the actor incrementally builds
evidence in a scratchpad through ``LOOK / CROP / COUNT / COMPARE /
READ_TEXT / SEGMENT / RETRIEVE / NOTE`` ops, and finally commits with
``ANSWER(text)`` which terminates the episode.

This is where the legacy ``inner_mdp`` operators relocate (see the
"Migration of inner-MDP operators" table in
``decision_agents/README.md``):

* ``GROUND(slot)``   → :meth:`VRHarness.step` ``LOOK(region)``
* ``RETRIEVE(q)``    → :meth:`VRHarness.step` ``RETRIEVE(q)``
* ``CONCLUDE(text)`` → :meth:`VRHarness.step` ``NOTE(text)``
* ``EXECUTE(answer)``→ :meth:`VRHarness.step` ``ANSWER(text)``

Phase 8.0 (perception plumbing): the perception ops (``LOOK / CROP /
READ_TEXT / COUNT / SEGMENT``) now optionally call injected
:class:`~decision_agents.core.perception.RegionDetector`,
:class:`~decision_agents.core.perception.Segmenter`, and
:class:`~decision_agents.core.perception.OCREngine` backends, going
through a per-episode :class:`~decision_agents.core.perception.EvidenceCache`.
Each successful tool call mints/updates an :class:`Entity` and pushes
it to ``info["schema_delta"]`` so :meth:`ActorAgent._merge_schema_delta`
folds it into the next step's schema.  When no backends are bound
(unit tests, smoke runs without an image) the harness falls back to
the original scratchpad-only behaviour — the migration is strictly
additive.

Side effects also continue to mutate the
:class:`~decision_agents.actor_agent.InnerScratchpad` the harness was
bound to via :meth:`bind_actor`.  When no bind has happened the
harness keeps a private scratchpad of its own.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decision_agents.core.harness import (
    ACTION_KIND_PRIMITIVE,
    ACTION_KIND_VR_ANSWER,
    ACTION_KIND_VR_LOOK,
    ACTION_KIND_VR_NOTE,
    ACTION_KIND_VR_RETRIEVE,
    Harness,
    HarnessState,
    parse_op_call,
)
from decision_agents.core.multimodal import VisualInput
from decision_agents.core.perception import (
    Detection,
    EvidenceCache,
    OCREngine,
    OCRResult,
    RegionDetector,
    Segmentation,
    Segmenter,
    crop_image_bytes,
)
from decision_agents.core.perception.cache import hash_image_bytes
from decision_agents.schema_parser import Entity, StateSchema

_LOGGER = logging.getLogger(__name__)

MAX_VALID_ACTIONS_IN_PROMPT: int = 16

# Detector confidence floor for COUNT() — keep slightly lower than the
# default LOOK threshold so dim instances still tally.
COUNT_THRESHOLD: float = 0.20
COUNT_TOP_K: int = 32


# ──────────────────────────────────────────────────────────────────────
# VR action vocabulary
# ──────────────────────────────────────────────────────────────────────

# Op tags surfaced to the LLM.  Kept short for prompt economy; the
# multi-strategy action parser tolerates the LLM swapping case or
# dropping the closing paren.  Order matters: this is the prompt's
# "numbered selection" order.
VR_OPS: Tuple[str, ...] = (
    "LOOK",
    "CROP",
    "COUNT",
    "COMPARE",
    "READ_TEXT",
    "SEGMENT",
    "RETRIEVE",
    "NOTE",
    "ANSWER",
)


# ──────────────────────────────────────────────────────────────────────
# Lightweight scratchpad fallback
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _LocalScratchpad:
    """Drop-in stand-in for :class:`InnerScratchpad` when no actor is bound.

    Mirrors the field names so the harness's mutation code works
    against either object — any future fields added to
    :class:`InnerScratchpad` should be mirrored here too.
    """

    grounded_slots: Dict[str, str] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# VR harness
# ──────────────────────────────────────────────────────────────────────


class VRHarness(Harness):
    """Read-only image + perception/reasoning-op vocabulary.

    Parameters
    ----------
    image
        :class:`VisualInput` for the fixed scene the actor reasons over.
        May be ``None`` for offline tests; the harness still runs but
        no image is rendered into the actor's prompt and perception ops
        no-op into the scratchpad.
    question
        Free-form question text.  Surfaced into the actor's task
        string and (optionally) the schema's ``goal``.
    gold_answer
        Optional reference answer.  When set, ``ANSWER(text)`` returns
        ``+1`` reward iff ``text.strip().lower() == gold.strip().lower()``;
        otherwise reward is ``0``.  ``None`` disables scoring.
    max_steps
        Hard cap on outer steps before the harness force-terminates
        with ``done=True``.  Prevents runaway VR rollouts when the
        actor never emits ``ANSWER``.
    candidate_args
        Optional dict mapping op → free-form arg suggestions used to
        pre-fill ``valid_actions`` (e.g. ``{"COUNT": ["cube", "sphere"]}``).
        When omitted, ``valid_actions`` enumerates entity-keyed ops
        from the schema and a generic ``ANSWER("<text>")`` placeholder.
    detector / segmenter / ocr
        Optional perception backends.  When provided, the matching ops
        actually look at the image and emit ``schema_delta`` entries.
        When omitted, the ops fall back to scratchpad-only behaviour
        (Phase 7 semantics) so existing tests keep passing.  Construct
        with :class:`~decision_agents.core.perception.MockRegionDetector`
        / etc. for deterministic CI.
    cache
        Optional :class:`~decision_agents.core.perception.EvidenceCache`.
        Auto-created when at least one backend is bound; cleared on
        every :meth:`reset`.
    detect_threshold
        LOOK confidence floor; detections below this are dropped before
        the entity is minted.
    """

    def __init__(
        self,
        *,
        image: Optional[VisualInput] = None,
        question: str = "",
        gold_answer: Optional[str] = None,
        max_steps: int = 8,
        candidate_args: Optional[Dict[str, List[str]]] = None,
        detector: Optional[RegionDetector] = None,
        segmenter: Optional[Segmenter] = None,
        ocr: Optional[OCREngine] = None,
        cache: Optional[EvidenceCache] = None,
        detect_threshold: float = 0.30,
    ) -> None:
        self.image = image
        self.question = question
        self.gold_answer = gold_answer
        self.max_steps = max(1, int(max_steps))
        self.candidate_args = candidate_args or {}

        # Episode-local state.
        self._t: int = 0
        self._done: bool = False
        self._last_obs: Any = ""
        self._answer: Optional[str] = None
        self._eid_counter: int = 0

        # Side-effect targets — bind via :meth:`bind_actor`.  Default
        # to a private scratchpad so tests that don't bind still work.
        self.scratchpad: Any = _LocalScratchpad()
        self._memory: Optional[Any] = None
        self._tracker: Optional[Any] = None

        # Phase 8.0 perception plumbing.
        self.detector = detector
        self.segmenter = segmenter
        self.ocr = ocr
        self.detect_threshold = float(detect_threshold)
        # Lazily auto-create a cache when at least one backend is bound,
        # so external callers can also share one cache across harnesses.
        self.cache: Optional[EvidenceCache] = cache
        if self.cache is None and any((detector, segmenter, ocr)):
            self.cache = EvidenceCache()

        # Image bytes are cached per-episode (loaded on first need).
        self._image_bytes: Optional[bytes] = None
        self._image_hash: Optional[str] = None

    # ── actor binding (called by ActorAgent before each episode) ─────

    def bind_actor(
        self,
        *,
        scratchpad: Any = None,
        memory: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        """Wire the harness's side-effect channel to an actor.

        :class:`~decision_agents.actor_agent.ActorAgent` calls this in
        :meth:`reset` and again whenever the actor's scratchpad is
        rebuilt (e.g. on skill reselect).  Without a bind the harness
        falls back to its private scratchpad (useful for tests).
        """
        if scratchpad is not None:
            self.scratchpad = scratchpad
        if memory is not None:
            self._memory = memory
        if tracker is not None:
            self._tracker = tracker

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self._t = 0
        self._done = False
        self._last_obs = self.question or ""
        self._answer = None
        self._eid_counter = 0
        self._image_bytes = None
        self._image_hash = None
        if self.cache is not None:
            self.cache.clear()
        info: Dict[str, Any] = {
            "task": self.question,
            "image": self.image.to_dict() if self.image is not None else None,
            "harness": "VRHarness",
        }
        return self._last_obs, info

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply one VR op.

        Perception ops (``LOOK / CROP / READ_TEXT / COUNT / SEGMENT``)
        consult the bound backends when present and emit
        ``info["schema_delta"]`` and (for ``CROP``)
        ``info["images"]``.  Scratchpad effects are preserved so
        backend-less rollouts behave as in Phase 7.

        Unknown ops or malformed action strings are accepted but score
        ``0`` reward — the actor's parser already filters most of those
        upstream, but we stay permissive so a misbehaving LLM doesn't
        crash the rollout.
        """
        self._t += 1
        op, arg = parse_op_call(action)
        reward = 0.0
        done = False
        info: Dict[str, Any] = {
            "op": op,
            "arg": arg,
            "harness": "VRHarness",
        }
        schema_delta: List[Entity] = []
        new_images: List[VisualInput] = []

        if op == "LOOK":
            entity = self._do_look(arg)
            if entity is not None:
                schema_delta.append(entity)
            self._scratchpad_grounded(_clean_slot(arg) or "scene")
            self._maybe_clear_ground_flag()

        elif op == "CROP":
            entity, crop_img = self._do_crop(arg)
            if entity is not None:
                schema_delta.append(entity)
            if crop_img is not None:
                new_images.append(crop_img)
            self._scratchpad_grounded(_clean_slot(arg) or "scene")
            self._maybe_clear_ground_flag()

        elif op == "READ_TEXT":
            entity = self._do_read_text(arg)
            if entity is not None:
                schema_delta.append(entity)
            self._scratchpad_grounded(_clean_slot(arg) or "scene")
            self._maybe_clear_ground_flag()

        elif op == "COUNT":
            count, entity = self._do_count(arg)
            self._scratchpad_note(f"COUNT({arg}) = {count}")
            if entity is not None:
                schema_delta.append(entity)
            info["count"] = count

        elif op == "COMPARE":
            # Compare is purely a scratchpad note — no perception call.
            self._scratchpad_note(f"COMPARE({arg})")

        elif op == "SEGMENT":
            entity = self._do_segment(arg)
            if entity is not None:
                schema_delta.append(entity)

        elif op == "RETRIEVE":
            hits = self._do_retrieve(arg)
            if hits:
                self._scratchpad_memory(query=arg, hits=hits)
            info["memory_hits"] = len(hits)

        elif op == "NOTE":
            self._scratchpad_note(arg)

        elif op == "ANSWER":
            self._answer = arg
            done = True
            reward = self._score(arg)
            info["answer"] = arg
            info["correct"] = reward > 0

        else:
            # Unknown op → no-op, but still consume a step.
            info["unknown_op"] = True

        if schema_delta:
            info["schema_delta"] = schema_delta
        if new_images:
            info["images"] = new_images
        if self.cache is not None:
            info["cache_stats"] = self.cache.stats()

        if self._t >= self.max_steps and not done:
            done = True
            info["truncated"] = True

        self._done = done
        self._last_obs = self.question
        return self._last_obs, reward, done, info

    # ── action enumeration ───────────────────────────────────────────

    def valid_actions(self, state: HarnessState) -> List[str]:
        """Enumerate concrete VR ops + a few entity-keyed templates.

        The first action is always ``ANSWER("<text>")`` so the LLM has
        a clear "I'm done" path; the rest cycle through the standard
        VR ops with placeholder args (or schema-derived ones when the
        schema carries entities).
        """
        actions: List[str] = []
        seen: set[str] = set()

        # ANSWER first: it's the only action that terminates.
        ans = 'ANSWER("<text>")'
        actions.append(ans)
        seen.add(ans)

        # Schema-derived per-entity templates (LOOK / CROP / READ_TEXT /
        # SEGMENT) — one per entity to keep prompt economical.
        if state.schema is not None and state.schema.entities:
            for eid in state.schema.entity_order[:4]:
                for op in ("LOOK", "CROP", "READ_TEXT", "SEGMENT"):
                    rendered = f"{op}({eid})"
                    if rendered not in seen:
                        seen.add(rendered)
                        actions.append(rendered)

        # Caller-provided candidate args for COUNT/COMPARE.
        for op in ("COUNT", "COMPARE"):
            for arg in self.candidate_args.get(op, [f"<{op.lower()}>"])[:2]:
                rendered = f"{op}({arg})"
                if rendered not in seen:
                    seen.add(rendered)
                    actions.append(rendered)

        # Free-form ops.
        for fallback in (
            'LOOK("<query>")',
            'RETRIEVE("<keywords>")',
            'NOTE("<text>")',
        ):
            if fallback not in seen:
                seen.add(fallback)
                actions.append(fallback)

        return actions[:MAX_VALID_ACTIONS_IN_PROMPT]

    # ── optional cost lookup ─────────────────────────────────────────

    def action_kind(self, action: str) -> str:
        """Map an action string to the right ``RewardConfig`` cost field."""
        op, _ = parse_op_call(action)
        if op in ("LOOK", "CROP", "READ_TEXT", "COUNT", "COMPARE", "SEGMENT"):
            return ACTION_KIND_VR_LOOK
        if op == "RETRIEVE":
            return ACTION_KIND_VR_RETRIEVE
        if op == "NOTE":
            return ACTION_KIND_VR_NOTE
        if op == "ANSWER":
            return ACTION_KIND_VR_ANSWER
        return ACTION_KIND_PRIMITIVE

    # ── exposed accessors (read-only) ────────────────────────────────

    @property
    def t(self) -> int:
        return self._t

    @property
    def answer(self) -> Optional[str]:
        return self._answer

    # ──────────────────────────────────────────────────────────────────
    # Perception op implementations
    # ──────────────────────────────────────────────────────────────────

    def _do_look(self, query: str) -> Optional[Entity]:
        """Run the detector and mint an entity for the top hit.

        Returns ``None`` (no schema delta) when:

        * No detector is bound, or
        * The image cannot be loaded, or
        * The detector returns no hits above ``detect_threshold``.

        The entity's ``eid`` is auto-minted (``e_perc_<n>``) so it
        cannot collide with the upstream parsed schema's
        ``e1`` / ``e2`` ids.
        """
        if self.detector is None:
            return None
        image_bytes, image_hash = self._load_image_bytes()
        if image_bytes is None:
            return None

        q = (query or "scene").strip().strip('"').strip("'")
        hits = self._cached(
            image_hash, "detect",
            {"q": q, "thr": self.detect_threshold, "k": 1},
            lambda: self.detector.detect(  # type: ignore[union-attr]
                image_bytes, q, threshold=self.detect_threshold, top_k=1,
            ),
        )
        if not hits:
            return None
        top = hits[0]
        return self._mint_entity_from_detection(top, source_op="LOOK")

    def _do_crop(
        self, arg: str,
    ) -> Tuple[Optional[Entity], Optional[VisualInput]]:
        """Resolve a bbox (from existing entity or fresh detection),
        crop the image, and return both an entity and the crop image.

        Falls back to (None, None) when no detector is bound and the
        arg doesn't refer to an existing entity, or when the image
        can't be loaded.
        """
        image_bytes, image_hash = self._load_image_bytes()
        if image_bytes is None:
            return None, None

        bbox, label, entity = self._resolve_bbox(arg, image_hash, image_bytes)
        if bbox is None:
            return None, None

        crop_bytes = self._cached(
            image_hash, "crop", {"bbox": list(bbox)},
            lambda: crop_image_bytes(image_bytes, bbox),
        )
        # Pillow may return the input bytes verbatim when unavailable;
        # we still attach as a separate VisualInput so the LLM gets
        # *something* even on minimal CI environments.
        crop_b64 = base64.b64encode(crop_bytes).decode("ascii")
        crop_img = VisualInput(
            image_b64=crop_b64,
            mime_type="image/png",
            caption=f"crop({label or arg or 'region'})",
        )

        if entity is None:
            entity = Entity(
                eid=self._mint_eid("crop"),
                label=label or (arg or "crop"),
                pos=bbox,
                extra={"source_op": "CROP"},
            )
        return entity, crop_img

    def _do_read_text(self, arg: str) -> Optional[Entity]:
        """OCR over a region (resolved from arg → entity bbox or fresh detect)."""
        if self.ocr is None:
            return None
        image_bytes, image_hash = self._load_image_bytes()
        if image_bytes is None:
            return None

        bbox, label, entity = self._resolve_bbox(arg, image_hash, image_bytes)
        spans = self._cached(
            image_hash, "ocr",
            {"bbox": list(bbox) if bbox else None},
            lambda: self.ocr.read(image_bytes, bbox=bbox),  # type: ignore[union-attr]
        )
        if not spans:
            return None

        text = " ".join(s.text for s in spans if s.text).strip()
        if not text:
            return None

        if entity is None:
            entity = Entity(
                eid=self._mint_eid("ocr"),
                label=label or (arg or "text"),
                pos=bbox or spans[0].bbox,
                extra={"source_op": "READ_TEXT"},
            )
        # Preserve any pre-existing value as well.
        entity.value = text
        attrs = dict(entity.attributes or {})
        attrs["text"] = text
        attrs["ocr_score"] = f"{max(s.score for s in spans):.2f}"
        entity.attributes = attrs
        return entity

    def _do_count(
        self, arg: str,
    ) -> Tuple[int, Optional[Entity]]:
        """Detector-backed counting.  Returns ``(count, summary_entity)``."""
        if self.detector is None or not arg:
            return 0, None
        image_bytes, image_hash = self._load_image_bytes()
        if image_bytes is None:
            return 0, None

        q = arg.strip().strip('"').strip("'")
        hits = self._cached(
            image_hash, "count",
            {"q": q, "thr": COUNT_THRESHOLD, "k": COUNT_TOP_K},
            lambda: self.detector.detect(  # type: ignore[union-attr]
                image_bytes, q, threshold=COUNT_THRESHOLD, top_k=COUNT_TOP_K,
            ),
        )
        count = len(hits)
        if count == 0:
            return 0, None

        # Attach an aggregate "count summary" entity so the next
        # action prompt can reason over `entity.attributes["count"]`.
        summary = Entity(
            eid=self._mint_eid("count"),
            label=f"count({q})",
            pos=hits[0].bbox if hits else None,
            extra={"source_op": "COUNT"},
            attributes={"count": str(count), "query": q},
        )
        return count, summary

    def _do_segment(self, arg: str) -> Optional[Entity]:
        """SAM-style segmentation; refines bbox + records pixel area."""
        if self.segmenter is None:
            return None
        image_bytes, image_hash = self._load_image_bytes()
        if image_bytes is None:
            return None

        bbox, label, entity = self._resolve_bbox(arg, image_hash, image_bytes)
        if bbox is None:
            return None

        masks = self._cached(
            image_hash, "segment", {"bbox": list(bbox)},
            lambda: self.segmenter.segment(  # type: ignore[union-attr]
                image_bytes, prompt_bbox=bbox, label=label or arg,
            ),
        )
        if not masks:
            return None

        seg = masks[0]
        if entity is None:
            entity = Entity(
                eid=self._mint_eid("seg"),
                label=label or (arg or "segment"),
                pos=seg.bbox,
                extra={"source_op": "SEGMENT"},
            )
        attrs = dict(entity.attributes or {})
        attrs["area_px"] = str(int(seg.area_px))
        attrs["seg_score"] = f"{seg.score:.2f}"
        entity.attributes = attrs
        # Refine bbox (segmenter often shrinks to mask extent).
        entity.pos = seg.bbox
        return entity

    # ──────────────────────────────────────────────────────────────────
    # Perception helpers
    # ──────────────────────────────────────────────────────────────────

    def _resolve_bbox(
        self,
        arg: str,
        image_hash: str,
        image_bytes: bytes,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], str, Optional[Entity]]:
        """Resolve an op argument into (bbox, label, optional existing entity).

        Order of resolution:

        1. Strip surrounding quotes; if the arg matches the eid pattern
           ``e\\d+`` and we previously minted such an entity, return its
           ``pos`` and re-use it (so the schema_delta updates the
           existing entity rather than spawning a duplicate).
        2. Otherwise treat the arg as a free-form text query and run
           the detector for a top-1 hit (when a detector is bound).
        3. Else return ``(None, arg, None)`` and the caller decides
           whether to fall back to whole-image OCR / abort.

        Note: the harness doesn't currently keep its own entity table —
        it doesn't see the actor's parsed schema directly.  When the
        actor's schema_delta merge minted an entity from a previous
        ``LOOK``, the actor knows; the harness has to re-derive (cache
        hit makes this cheap).
        """
        q = (arg or "").strip().strip('"').strip("'")
        if not q:
            return None, "", None

        # Free-form text query → detector lookup.
        if self.detector is None:
            return None, q, None
        hits = self._cached(
            image_hash, "detect",
            {"q": q, "thr": self.detect_threshold, "k": 1},
            lambda: self.detector.detect(  # type: ignore[union-attr]
                image_bytes, q, threshold=self.detect_threshold, top_k=1,
            ),
        )
        if not hits:
            return None, q, None
        top = hits[0]
        return top.bbox, q, None

    def _mint_entity_from_detection(
        self, det: Detection, *, source_op: str,
    ) -> Entity:
        return Entity(
            eid=self._mint_eid(source_op.lower()),
            label=det.label,
            pos=det.bbox,
            extra={
                "source_op": source_op,
                "conf": f"{det.score:.2f}",
                **{k: v for k, v in det.extra.items() if isinstance(v, (str, int, float))},
            },
        )

    def _mint_eid(self, prefix: str) -> str:
        self._eid_counter += 1
        return f"e_{prefix}_{self._eid_counter}"

    def _load_image_bytes(self) -> Tuple[Optional[bytes], str]:
        """Return ``(image_bytes, image_hash)``; cached per-episode.

        Falls back to ``(None, "")`` when no image is bound or when
        decoding fails (e.g. a remote ``image_url`` we can't fetch
        offline).  Callers must check the bytes for None before
        invoking backends.
        """
        if self._image_bytes is not None and self._image_hash is not None:
            return self._image_bytes, self._image_hash
        if self.image is None:
            return None, ""

        # Prefer ``image_b64`` (in-memory; cheapest); else load from
        # ``image_path``; ``image_url`` is unsupported here on purpose
        # (keeps the harness offline-friendly).
        try:
            if self.image.image_b64:
                self._image_bytes = base64.b64decode(self.image.image_b64)
            elif self.image.image_path:
                with open(self.image.image_path, "rb") as f:
                    self._image_bytes = f.read()
            else:
                return None, ""
        except Exception as exc:  # pragma: no cover — defensive
            _LOGGER.warning("VRHarness failed to load image bytes: %s", exc)
            return None, ""

        self._image_hash = hash_image_bytes(self._image_bytes)
        return self._image_bytes, self._image_hash

    def _cached(
        self,
        image_hash: str,
        op: str,
        args: Any,
        compute: Any,
    ) -> Any:
        """Wrap a backend call in :class:`EvidenceCache` lookup.

        Falls back to a plain call when no cache is bound (shouldn't
        happen in practice — the constructor auto-creates one when any
        backend is present — but keeps the helper robust).
        """
        if self.cache is None:
            return compute()
        cached = self.cache.get(image_hash, op, args)
        if cached is not None:
            return cached
        result = compute()
        self.cache.put(image_hash, op, args, result)
        return result

    def _maybe_clear_ground_flag(self) -> None:
        """Tell the actor's tracker that a ground op fired, if bound."""
        if self._tracker is not None:
            try:
                self._tracker.clear_ground_flag(None)
            except Exception:  # pragma: no cover — defensive
                pass

    # ──────────────────────────────────────────────────────────────────
    # Scratchpad mutation helpers (Phase 7 semantics, preserved)
    # ──────────────────────────────────────────────────────────────────

    def _scratchpad_grounded(self, slot: str) -> None:
        """Mark *slot* as observed on the bound scratchpad."""
        sp = self.scratchpad
        try:
            sp.grounded_slots.setdefault(slot, "observed")
        except AttributeError:  # pragma: no cover — exotic stand-in
            _LOGGER.debug("VRHarness scratchpad missing grounded_slots; ignoring")

    def _scratchpad_note(self, text: str) -> None:
        if not text:
            return
        try:
            self.scratchpad.notes.append(text[:140])
            self.scratchpad.notes = self.scratchpad.notes[-5:]
        except AttributeError:  # pragma: no cover
            _LOGGER.debug("VRHarness scratchpad missing notes; ignoring")

    def _scratchpad_memory(
        self, *, query: str, hits: Sequence[Any]
    ) -> None:
        try:
            self.scratchpad.memory_hits.extend(
                {"query": (query or "")[:80], "hit": _stringify(h)} for h in hits
            )
            self.scratchpad.memory_hits = self.scratchpad.memory_hits[-5:]
        except AttributeError:  # pragma: no cover
            _LOGGER.debug("VRHarness scratchpad missing memory_hits; ignoring")

    def _do_retrieve(self, arg: str) -> List[Any]:
        """Run the RETRIEVE query against the bound memory store.

        Falls back to an empty hit list when no memory is bound, when
        the query is empty, or when the memory store raises — keeping
        the rollout going matters more than perfect recall.
        """
        if self._memory is None or not arg:
            return []
        try:
            return list(self._memory.query(arg, k=3))
        except Exception as exc:  # pragma: no cover — defensive
            _LOGGER.warning("VRHarness memory.query failed: %s", exc)
            return []

    def _score(self, answer: Optional[str]) -> float:
        """Return ``+1`` for an exact match against ``gold_answer``."""
        if self.gold_answer is None or answer is None:
            return 0.0
        a = str(answer).strip().strip('"').strip("'").lower()
        g = str(self.gold_answer).strip().lower()
        return 1.0 if a == g else 0.0


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _clean_slot(arg: str) -> str:
    """Strip surrounding whitespace + quotes from a slot/query string.

    Mirrors the cleaning the perception ops do on detector queries so
    scratchpad keys stay aligned with the schema-delta entity labels.
    """
    if not arg:
        return ""
    return arg.strip().strip('"').strip("'").strip()


def _stringify(hit: Any) -> str:
    """Render a memory hit compactly (mirrors ``actor_agent._stringify_memory_hit``)."""
    if isinstance(hit, dict):
        parts: List[str] = []
        for key in ("summary", "action", "outcome", "key"):
            v = hit.get(key)
            if v:
                parts.append(f"{key}={str(v)[:60]}")
        if not parts:
            parts = [f"{k}={str(v)[:60]}" for k, v in list(hit.items())[:3] if v]
        return " | ".join(parts) if parts else "(empty)"
    return str(hit)[:120]


__all__ = ["VRHarness", "VR_OPS"]
