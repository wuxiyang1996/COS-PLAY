#!/usr/bin/env python
"""Target-domain dispatch for the Phase-4/5/6 cross-domain transfer cycle.

Phase-4 (Day-5b) shipped game->game (within-gymv) transfer through
``_phase4_transfer_cycle.py`` with the target adapter, executor,
schema producer, and demos all hardcoded for ``target_domain='gymv'``.

Phase 5 of the cross-domain measurement plan extends this to the
remaining four target domains:

    visual_reasoning  -- VTB + TIR-Bench (image-VR)
    video             -- Video-Holmes + SIV-Bench (video-VR)
    osworld           -- desktop tasks
    browser           -- BrowserGym

Rather than rewriting ``_phase4_transfer_cycle._run_transfer`` per
domain, this module provides:

  * ``TargetBuild`` dataclass: the (adapter, harness, demos,
    success_fn_factory) bundle one transfer cell needs.
  * ``_TARGET_BUILDERS``: a per-target_domain dict of builder
    callables that each produce a ``TargetBuild`` given the parsed
    CLI args (the parent driver's ``argparse.Namespace``).
  * ``build_target(target_domain, args) -> TargetBuild``: the public
    entry point. Raises ``UnsupportedTargetDomain`` on unknown names
    and ``NotImplementedError`` for stages that haven't shipped yet.

Phase-5 sub-agents (Stages 1-4) add their per-domain builder here.
The gymv builder is the canonical reference; mirror its shape.

See ``implementation_notes/legacy/phase5-cross-domain-measurement.md`` for
the per-stage scope (which target_domain each sub-agent owns).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("phase4_target_dispatch")


# ---------------------------------------------------------------------------
# Public API: TargetBuild + dispatcher
# ---------------------------------------------------------------------------


@dataclass
class TargetBuild:
    """Per-cell bundle returned by a target-domain builder.

    Fields:
      target_domain:
        The canonical domain string registered with
        ``common.enums.TRANSFER_TARGET_DOMAINS`` (e.g. ``'gymv'``,
        ``'visual_reasoning'``, ``'video'``, ``'osworld'``,
        ``'browser'``).
      adapter:
        The ``SkillAdapter`` (or subclass) instance with its
        executor already bound (via ``adapter.set_executor(...)`` or
        a domain-specific ``bind_<domain>_executor`` helper).
      harness:
        ``SkillHarness`` built around an ``AdapterRegistry`` that has
        ``adapter`` registered. The driver loops calls
        ``few_shot.adapt(...)`` against this harness.
      demos:
        List of ``FewShotDemo``s pre-loaded from the target-domain's
        cold-start corpus. The driver passes these to
        ``FewShotAdapter.adapt(demos=...)`` per skill.
      success_fn_factory:
        Callable taking ``(pass_rate_threshold: float,
        require_episode_success: bool) -> SuccessFn``. The driver
        constructs a fresh per-skill ``SuccessFn`` by calling this
        factory inside the skill loop. For gymv this is
        ``harness.gymv_success.make_per_step_success_fn``; for VR /
        video / osworld / browser the per-domain ``*_success.py``
        modules export an analogous factory.
    """

    target_domain: str
    adapter: Any
    harness: Any  # SkillHarness; loose-typed to avoid import cycles
    demos: List[Any]  # List[FewShotDemo]
    success_fn_factory: Callable[..., Any]


class UnsupportedTargetDomain(SystemExit):
    """Raised by ``build_target`` for unknown ``--target-domain`` names."""


# Type of a per-domain builder. Each stage sub-agent registers their
# domain's builder by adding an entry to ``_TARGET_BUILDERS`` below.
TargetBuilder = Callable[[argparse.Namespace], TargetBuild]


def build_target(target_domain: str, args: argparse.Namespace) -> TargetBuild:
    """Dispatch to the per-domain builder for ``target_domain``.

    Raises:
      UnsupportedTargetDomain: if no builder is registered.
      NotImplementedError: if the builder is the placeholder stub for
        a stage that hasn't shipped yet (Stages 1-4 land progressively
        per ``implementation_notes/legacy/phase5-cross-domain-measurement.md``).
    """
    builder = _TARGET_BUILDERS.get(target_domain)
    if builder is None:
        raise UnsupportedTargetDomain(
            f"unknown --target-domain={target_domain!r}; "
            f"registered: {sorted(_TARGET_BUILDERS)}"
        )
    return builder(args)


# ---------------------------------------------------------------------------
# Builder: gymv (canonical reference)
# ---------------------------------------------------------------------------


def _build_gymv_target(args: argparse.Namespace) -> TargetBuild:
    """Build the gymv (within-game) transfer cell.

    This is the existing Day-5b path extracted verbatim out of
    ``_phase4_transfer_cycle._run_transfer``. Mirror this shape for
    all new domains:

      1. Build the target env (deterministic where possible).
      2. Build a schema producer that round-trips through
         ``parse_schema_canonical`` -> ``StateSchema.facts``.
      3. Build the executor that the adapter calls per hop.
      4. Construct the adapter; widen ``supported_types`` so
         REASONING / GROUNDING skills also dispatch.
      5. Wire ``adapter.set_executor(executor)`` (or
         ``bind_<domain>_executor(adapter, ...)`` for VR / video).
      6. Build a ``SkillHarness`` around the adapter via
         ``AdapterRegistry``.
      7. Build target-task demos (``List[FewShotDemo]``).
      8. Pick the per-domain success_fn factory
         (``make_per_step_success_fn`` for gymv;
         ``make_qa_success_fn`` for VR/video; etc.).
    """
    # Imports are deferred so importing this module (e.g. for the
    # dispatcher table introspection) doesn't trigger heavy harness
    # imports unless we actually build a target.
    from common.enums import SkillType
    from harness import (
        AdapterRegistry,
        HarnessConfig,
        SkillHarness,
        make_gaming_env_producer,
        make_gymv_executor,
        make_per_step_success_fn,
    )
    from harness.adapters.gymv_adapter import GymvAdapter
    from harness.few_shot_demos_gymv import build_demos_from_episodes

    # Reuse the Phase-2 driver's env-builder.
    from labeling_supplement._phase2_real_env_skill_smoke import build_env

    target_game = args.target  # for gymv, --target *is* the task name
    target_env, target_env_source = build_env(target_game)
    logger.info("target env: %s (source=%s)", target_game, target_env_source)
    schema_producer = make_gaming_env_producer(target_game)
    executor, _holder = make_gymv_executor(
        target_env,
        domain="gymv",
        task=target_game,
        on_unresolved="skip",
        schema_producer=schema_producer,
    )

    adapter = GymvAdapter()
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )
    adapter.set_executor(executor)

    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig(
        seed=0,
        default_budget_hops=12,
        default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_episodes(
        Path(args.actions_root),
        corpus="env_wrappers",
        game=target_game,
        max_episodes=args.max_episodes,
        max_demos_per_episode=args.max_demos_per_episode,
    )
    logger.info(
        "loaded %d target-task demo(s) from %s cold-start episodes",
        len(demos), target_game,
    )

    return TargetBuild(
        target_domain="gymv",
        adapter=adapter,
        harness=harness,
        demos=demos,
        success_fn_factory=make_per_step_success_fn,
    )


# ---------------------------------------------------------------------------
# Builders: cross-domain stubs (Stages 1-4 fill in)
# ---------------------------------------------------------------------------


def _stage_not_shipped(stage: str, target_domain: str, sub_section: str) -> Callable[
    [argparse.Namespace], TargetBuild
]:
    """Factory for a placeholder builder that fails clean with a
    pointer at the stage's design-memo section."""

    def _stub(args: argparse.Namespace) -> TargetBuild:  # noqa: ARG001
        raise NotImplementedError(
            f"{stage} ({target_domain!r}) not implemented yet -- see "
            f"implementation_notes/legacy/phase5-cross-domain-measurement.md "
            f"{sub_section}. Land the per-domain builder by replacing "
            f"_TARGET_BUILDERS[{target_domain!r}] with a real "
            f"`_build_{target_domain}_target` callable that mirrors "
            f"`_build_gymv_target`'s shape."
        )

    return _stub


def _build_visual_reasoning_target(args: argparse.Namespace) -> TargetBuild:
    """Stage-1 image-VR transfer cell -- VTB + TIR-Bench.

    Per-sample image binding shipped 2026-05-02 (Phase-5/6 §12.1 Tier 1
    follow-up). The dispatcher now:

    1. Discovers ``{task_id: image_path}`` from
       ``<cold_start_root>/<run>/<sub_corpus>/sample_*.json`` cross-referenced
       with ``frames/sample_NNN/frame_00.png`` siblings (see
       :func:`harness._vr_per_sample_executor.discover_task_to_image`).
    2. Wires a :class:`~harness._vr_per_sample_executor.TaskAwareVisualReasoningExecutor`
       wrapper into the adapter that lazily binds (and caches) one real
       :class:`~visual_reasoning_wrapper.skill_executor.VisualReasoningExecutor`
       per image. Falls back to the deterministic stub on tasks where no
       frame is on disk.

    When the discovery returns an empty map (cold-start tree has no
    timestamped run with frames), the adapter is left on the inherited
    stub executor -- the chain still runs but every hop identity-passes
    its predicates, mirroring the pre-2026-05-02 behaviour.
    """
    from common.enums import SkillType
    from harness import AdapterRegistry, HarnessConfig, SkillHarness
    from harness.adapters.visual_reasoning_adapter import VisualReasoningAdapter
    from harness.few_shot_demos_vr import build_demos_from_vr_samples
    from harness._vr_per_sample_executor import (
        TaskAwareVisualReasoningExecutor,
        discover_task_to_image,
    )
    import harness.qa_success  # noqa: F401  (registers success_fn factory)
    from harness.qa_success import make_qa_success_fn

    cold_start_root = Path(
        args.cold_start_root or "Cold-start-out-visual-reasoning"
    )
    if not cold_start_root.exists():
        raise SystemExit(
            f"cold_start_root missing: {cold_start_root} "
            f"(expected Cold-start-out-visual-reasoning/"
            f"{args.target}/sample_*.json)"
        )

    sub_corpus = args.target
    if sub_corpus not in ("visual_toolbench", "tir_bench"):
        raise SystemExit(
            f"--target {sub_corpus!r} not a known visual_reasoning "
            f"sub-corpus; expected one of: visual_toolbench, tir_bench"
        )

    adapter = VisualReasoningAdapter()
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )

    # Discover per-sample images and bind a real executor wrapper. If
    # nothing is found, we leave the adapter on its inherited stub so
    # the chain still runs (matches pre-Tier-1 behaviour).
    task_to_image = discover_task_to_image(
        cold_start_root, sub_corpus=sub_corpus,
    )
    if task_to_image:
        prefer_gdino = bool(getattr(args, "vr_prefer_gdino", False))
        adapter.set_executor(
            TaskAwareVisualReasoningExecutor(
                task_to_image,
                prefer_gdino=prefer_gdino,
            )
        )
        logger.info(
            "bound TaskAwareVisualReasoningExecutor with %d task->image "
            "mapping(s) for sub_corpus=%s (prefer_gdino=%s)",
            len(task_to_image), sub_corpus, prefer_gdino,
        )
    else:
        logger.warning(
            "no per-sample frames discovered under %s/<run>/%s/frames/; "
            "leaving adapter on inherited deterministic stub (chain will "
            "run but predicates identity-pass)",
            cold_start_root, sub_corpus,
        )

    registry = AdapterRegistry()
    registry.register(adapter)
    harness_obj = SkillHarness(registry, config=HarnessConfig(
        seed=0, default_budget_hops=8, default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_vr_samples(
        cold_start_root,
        sub_corpus=sub_corpus,
        max_demos=int(getattr(args, "max_episodes", 8))
        * int(getattr(args, "max_demos_per_episode", 1)),
    )
    logger.info("loaded %d VR demo(s) from %s/%s",
                len(demos), cold_start_root, sub_corpus)

    # Wrap the QA success_fn factory with the per-domain runtime
    # predicate translator (Phase-5/6 §12.3 / §11.5.0). For game->VR
    # transfers, the wrapper rewrites the source skill's
    # contract.effects_{add,del} from game vocabulary
    # (e.g. cumulative_reward_increased) to VR vocabulary
    # (answer_emitted, answer_matches_gold) before make_qa_success_fn
    # evaluates it. Diagonal (visual_reasoning->visual_reasoning) calls
    # are an identity passthrough.
    from harness.predicate_translator import with_predicate_translation
    return TargetBuild(
        target_domain="visual_reasoning",
        adapter=adapter,
        harness=harness_obj,
        demos=demos,
        success_fn_factory=with_predicate_translation(
            make_qa_success_fn, target_domain="visual_reasoning",
        ),
    )

def _build_video_target(args: argparse.Namespace) -> TargetBuild:
    """Video-VR transfer cell — Video-Holmes + SIV-Bench (Phase 5 §11.5).

    Mirrors :func:`_build_gymv_target`'s shape:
      1. Resolve the cold-start sub-corpus name from ``args.target``.
      2. Build the (deterministic) video executor and bind it into a
         ``VideoAdapter`` via :func:`bind_video_executor`.
      3. Construct a ``SkillHarness`` around an ``AdapterRegistry``
         that has the adapter registered.
      4. Harvest target-task demos from the cold-start corpus.
      5. Return the bundle plus the video QA success-fn factory the
         driver instantiates per skill.
    """
    from common.enums import SkillType
    from harness import AdapterRegistry, HarnessConfig, SkillHarness
    from harness._video_per_sample_executor import (
        TaskAwareVideoReasoningExecutor,
        discover_task_to_video_meta,
    )
    from harness.adapters.video_adapter import VideoAdapter, bind_video_executor
    from harness.few_shot_demos_video import build_demos_from_video_samples
    import harness.video_qa_success  # noqa: F401  registers success_fn factory
    from harness.video_qa_success import make_video_qa_success_fn

    cold_start_root = Path(
        args.cold_start_root or "Cold-start-out-visual-reasoning-video"
    )
    if not cold_start_root.exists():
        raise SystemExit(
            f"cold_start_root missing: {cold_start_root} "
            f"(expected Cold-start-out-visual-reasoning-video/"
            f"{args.target}/sample_*.json)"
        )

    sub_corpus = args.target
    if sub_corpus not in ("video_holmes", "siv_bench"):
        raise SystemExit(
            f"--target {sub_corpus!r} not a known video sub-corpus; "
            f"expected one of: video_holmes, siv_bench"
        )

    adapter = VideoAdapter()
    adapter.supported_types = (
        SkillType.ACTION,
        SkillType.MIXED,
        SkillType.GROUNDING,
        SkillType.REASONING,
    )
    # Mirror the Stage 1 (image-VR) per-sample binding pattern: discover
    # the cold-start task->video_meta map and bind a real
    # TaskAwareVideoReasoningExecutor when at least one mapping is
    # found. The wrapper routes InnerAction verbs to the real
    # VideoReasoningExecutor (decode + VLM tools) while legacy
    # video-domain verbs (SAMPLE_FRAME / EMIT_ANSWER / ...) stay on the
    # per-task deterministic stub so both verb sets co-exist. When the
    # cold-start tree is missing (CI without video data) we fall back
    # to the bare deterministic stub via bind_video_executor so the
    # chain still runs.
    task_to_video_meta = discover_task_to_video_meta(
        cold_start_root, sub_corpus=sub_corpus,
    )
    if task_to_video_meta:
        prefer_gdino = bool(getattr(args, "vr_prefer_gdino", False))
        num_frames = int(getattr(args, "video_num_frames", 8))
        adapter.set_executor(
            TaskAwareVideoReasoningExecutor(
                task_to_video_meta,
                num_frames=num_frames,
                prefer_gdino=prefer_gdino,
            )
        )
        logger.info(
            "bound TaskAwareVideoReasoningExecutor with %d task->video_meta "
            "mapping(s) for sub_corpus=%s (num_frames=%d, prefer_gdino=%s)",
            len(task_to_video_meta), sub_corpus, num_frames, prefer_gdino,
        )
    else:
        bind_video_executor(adapter, video_meta=None)
        logger.warning(
            "no cold-start video_meta found under %s/%s/sample_*.json; "
            "falling back to bare deterministic stub (chain will run "
            "but InnerAction hops identity-pass)",
            cold_start_root, sub_corpus,
        )

    registry = AdapterRegistry()
    registry.register(adapter)
    harness_obj = SkillHarness(
        registry,
        config=HarnessConfig(
            seed=0,
            default_budget_hops=8,
            default_budget_ms=30_000.0,
        ),
    )

    max_demos = int(getattr(args, "max_episodes", 8)) * int(
        getattr(args, "max_demos_per_episode", 1) or 1
    )
    demos = build_demos_from_video_samples(
        cold_start_root,
        sub_corpus=sub_corpus,
        max_demos=max(1, max_demos),
    )
    logger.info(
        "loaded %d video demo(s) from %s/%s",
        len(demos), cold_start_root, sub_corpus,
    )

    # See _build_visual_reasoning_target for the rationale; video uses
    # the same translator with target_domain="video" (which has the
    # full visual_reasoning vocab plus temporal_ordering_correct +
    # frame_referent_grounded -- so e.g. entity_disappeared maps to
    # temporal_ordering_correct here vs. dropped for image-VR).
    from harness.predicate_translator import with_predicate_translation
    return TargetBuild(
        target_domain="video",
        adapter=adapter,
        harness=harness_obj,
        demos=demos,
        success_fn_factory=with_predicate_translation(
            make_video_qa_success_fn, target_domain="video",
        ),
    )

def _build_osworld_target(args: argparse.Namespace) -> TargetBuild:
    """OSWorld desktop transfer cell.

    Mirror of :func:`_build_gymv_target` for the OSWorld target. The
    deterministic-stub executor (no real ``pyautogui``) keeps the
    dispatch chain firing so the per-hop success_fn evaluates
    effects against the producer-emitted facts. Real desktop binding
    is a later cut (rollout memo §6.1, §11.5.5).

    Builder shape (matches gymv's eight-step recipe):

      1. Resolve the cold-start root (``Cold-start-out-osworld/``).
      2. Build the OSWorld schema producer for ``args.target``.
      3. Build the deterministic-stub executor.
      4. Construct the adapter; widen ``supported_types`` so
         REASONING / GROUNDING skills also dispatch.
      5. Wire ``adapter.set_executor(executor)``.
      6. Build the harness around the adapter via ``AdapterRegistry``.
      7. Build target-task demos via
         :func:`build_demos_from_osworld_episodes`.
      8. Return the per-domain success_fn factory.
    """
    from common.enums import SkillType
    from harness import (
        AdapterRegistry,
        HarnessConfig,
        SkillHarness,
        make_osworld_executor,
        make_osworld_producer,
    )
    from harness.adapters.osworld_adapter import OsworldAdapter
    from harness.few_shot_demos_osworld import build_demos_from_osworld_episodes
    # Importing the module also registers the success_fn factory at
    # import time so ``success_fn_for_domain('osworld')`` resolves
    # downstream even if the caller never imports it directly.
    import harness.osworld_success  # noqa: F401
    from harness.osworld_success import make_osworld_per_step_success_fn

    cold_start_root = Path(
        getattr(args, "cold_start_root", None) or "Cold-start-out-osworld"
    )
    if not cold_start_root.exists():
        raise SystemExit(
            f"cold_start_root missing: {cold_start_root} "
            f"(expected Cold-start-out-osworld/<ts>/{args.target}/"
            f"<task-uuid>/episode_*.json)"
        )

    domain_name = args.target  # e.g. "vlc"
    schema_producer = make_osworld_producer(domain_name)
    executor, _holder = make_osworld_executor(
        domain="osworld",
        task=domain_name,
        on_unresolved="skip",
        schema_producer=schema_producer,
    )

    adapter = OsworldAdapter()
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )
    # Try to bind the real-env wrapper that talks to a live
    # ``happysixd/osworld-docker`` container fleet over HTTP. The
    # fleet is a pool of N containers (typically 13) discovered via
    # ``docker ps``; each task_id hash-pins to one container so a
    # hot loop of hops on the same task hits the same desktop state.
    # Falls back to the deterministic stub when (a) the cold-start
    # tree lacks task->meta entries, (b) the docker daemon is
    # unreachable, or (c) no containers are running.
    from harness._executor_helpers.osworld_client import OsworldContainerPool
    from harness._osworld_per_sample_executor import (
        TaskAwareOsworldExecutor, discover_task_to_osworld_meta,
    )
    task_to_osworld_meta = discover_task_to_osworld_meta(
        cold_start_root, domain_filter=domain_name,
    )
    osworld_pool = OsworldContainerPool.from_discovery()
    if task_to_osworld_meta and osworld_pool is not None:
        prefer_gdino = bool(getattr(args, "vr_prefer_gdino", False))
        adapter.set_executor(
            TaskAwareOsworldExecutor(
                task_to_osworld_meta,
                osworld_pool,
                prefer_gdino=prefer_gdino,
            )
        )
        logger.info(
            "bound TaskAwareOsworldExecutor with %d task->meta entries "
            "and %d-container pool (domain=%s, prefer_gdino=%s)",
            len(task_to_osworld_meta), osworld_pool.size,
            domain_name, prefer_gdino,
        )
    else:
        adapter.set_executor(executor)
        if not task_to_osworld_meta:
            logger.warning(
                "no cold-start task_meta discovered under %s/<run>/%s/; "
                "falling back to deterministic stub",
                cold_start_root, domain_name,
            )
        else:
            logger.warning(
                "no happysixd/osworld-docker containers running; "
                "falling back to deterministic stub (start the OSWorld "
                "container fleet to enable real-env binding)",
            )

    registry = AdapterRegistry()
    registry.register(adapter)
    harness_obj = SkillHarness(registry, config=HarnessConfig(
        seed=0,
        default_budget_hops=12,
        default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_osworld_episodes(
        cold_start_root,
        domain=domain_name,
        max_episodes=int(getattr(args, "max_episodes", 3)),
        max_demos_per_episode=int(getattr(args, "max_demos_per_episode", 2)),
    )
    logger.info(
        "loaded %d osworld demo(s) from %s/*/%s/",
        len(demos), cold_start_root, domain_name,
    )

    # Per-step OSWorld success_fn wrapped with the per-domain
    # predicate translator (Phase-5/6 §12.3). gymv->osworld is mostly
    # identity (osworld's vocab covers most game predicates by name)
    # plus cumulative_reward_increased -> task_status and dropping
    # entity_value_increased / decreased (no scalar-value entities on
    # the desktop).
    from harness.predicate_translator import with_predicate_translation
    return TargetBuild(
        target_domain="osworld",
        adapter=adapter,
        harness=harness_obj,
        demos=demos,
        success_fn_factory=with_predicate_translation(
            make_osworld_per_step_success_fn, target_domain="osworld",
        ),
    )


def _build_browser_target(args: argparse.Namespace) -> TargetBuild:
    """BrowserGym (browser) transfer cell — Stage 4.

    Mirrors `_build_gymv_target`'s shape: deferred imports,
    deterministic-stub executor (rollout memo §6.1), demos loaded
    from ``Cold-start-out-browsergym/<task_prefix>.*/episode_*.json``,
    and the per-domain ``"browser"`` success_fn factory registered
    via ``harness.browser_success`` at import time.

    The first cut does NOT drive a real Playwright browser — the
    deterministic stub executor exercises the dispatch + per-hop
    predicate evaluator end-to-end so Stage 4 acceptance can confirm
    the chain runs without raising. Real browser binding lands in a
    follow-up by replacing the closure
    `make_browsergym_executor` returns.
    """
    from common.enums import SkillType
    from harness import (
        AdapterRegistry,
        HarnessConfig,
        SkillHarness,
        make_browsergym_executor,
        make_browsergym_producer,
    )
    from harness.adapters.browser_adapter import BrowserAdapter
    import harness.browser_success  # noqa: F401  (registers success_fn factory)
    from harness.browser_success import make_browser_per_step_success_fn
    from harness.few_shot_demos_browsergym import (
        build_demos_from_browsergym_episodes,
    )

    cold_start_root = Path(
        args.cold_start_root or "Cold-start-out-browsergym"
    )
    if not cold_start_root.exists():
        raise SystemExit(
            f"cold_start_root missing: {cold_start_root} "
            f"(expected Cold-start-out-browsergym/{args.target}.*/episode_*.json)"
        )

    task_prefix = args.target  # e.g. "assistantbench"
    schema_producer = make_browsergym_producer(task_prefix)
    executor, _holder = make_browsergym_executor(
        domain="browser",
        task=task_prefix,
        on_unresolved="skip",
        schema_producer=schema_producer,
    )

    adapter = BrowserAdapter()
    # NB: BrowserAdapter is its own SkillAdapter subclass (NOT a
    # StubTransferTargetAdapter), so we widen `supported_types` on
    # the instance directly. The class default already includes all
    # four types but we re-assert defensively in case a future cut
    # tightens the class default.
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )
    # Try to bind the real-env wrapper that hosts a Playwright-driven
    # BrowserGym ``gym.Env`` in a subprocess running in the
    # ``browsergym`` conda env. The wrapper translates harness-shaped
    # verbs (``CLICK``/``FILL``/...) into BrowserGym high-level
    # actions (``click("47")`` etc.) and dispatches InnerAction
    # verbs (``GROUND``/``CHECK``/...) to a VisualReasoningExecutor
    # built from each step's screenshot. Falls back to the
    # deterministic stub when (a) cold-start data lacks the task,
    # (b) the helper subprocess fails to spawn (missing conda env,
    # missing playwright, etc.), or (c) ``gym.make`` raises.
    from harness._browser_per_sample_executor import (
        TaskAwareBrowserExecutor, discover_task_to_browser_meta,
    )
    task_to_browser_meta = discover_task_to_browser_meta(
        cold_start_root,
        task_prefix=(task_prefix + "." if task_prefix else None),
    )
    if task_to_browser_meta:
        prefer_gdino = bool(getattr(args, "vr_prefer_gdino", False))
        browser_conda_env = str(getattr(args, "browser_conda_env", "browsergym"))
        adapter.set_executor(
            TaskAwareBrowserExecutor(
                task_to_browser_meta,
                conda_env=browser_conda_env,
                prefer_gdino=prefer_gdino,
            )
        )
        logger.info(
            "bound TaskAwareBrowserExecutor with %d task->meta entries "
            "(conda_env=%s, prefer_gdino=%s); helper will spawn lazily "
            "on first hop",
            len({m['task_id'] for m in task_to_browser_meta.values()}),
            browser_conda_env, prefer_gdino,
        )
    else:
        adapter.set_executor(executor)
        logger.warning(
            "no cold-start browser_meta discovered under %s/%s.*/; "
            "falling back to deterministic stub",
            cold_start_root, task_prefix,
        )

    registry = AdapterRegistry()
    registry.register(adapter)
    harness_obj = SkillHarness(registry, config=HarnessConfig(
        seed=0,
        default_budget_hops=12,
        default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_browsergym_episodes(
        cold_start_root,
        task_prefix=task_prefix,
        max_episodes=int(getattr(args, "max_episodes", 3)),
        max_demos_per_episode=int(getattr(args, "max_demos_per_episode", 2)),
    )
    logger.info(
        "loaded %d browsergym demo(s) from %s/%s.*/",
        len(demos), cold_start_root, task_prefix,
    )

    # See _build_osworld_target for the rationale; browser uses the
    # same translator with target_domain="browser" (which lacks
    # entity_disappeared in its vocab -> remapped to attribute_changed,
    # the closest DOM-level analogue).
    from harness.predicate_translator import with_predicate_translation
    return TargetBuild(
        target_domain="browser",
        adapter=adapter,
        harness=harness_obj,
        demos=demos,
        success_fn_factory=with_predicate_translation(
            make_browser_per_step_success_fn, target_domain="browser",
        ),
    )


# ---------------------------------------------------------------------------
# Registry (the table sub-agents extend)
# ---------------------------------------------------------------------------


_TARGET_BUILDERS: Dict[str, TargetBuilder] = {
    "gymv": _build_gymv_target,
    "visual_reasoning": _build_visual_reasoning_target,
    "video": _build_video_target,
    "osworld": _build_osworld_target,
    "browser": _build_browser_target,
}


def registered_target_domains() -> List[str]:
    """Sorted view of currently-registered target domains."""
    return sorted(_TARGET_BUILDERS.keys())


__all__ = [
    "TargetBuild",
    "TargetBuilder",
    "UnsupportedTargetDomain",
    "build_target",
    "registered_target_domains",
]
