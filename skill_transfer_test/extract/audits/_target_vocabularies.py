"""Aspirational per-target vocabularies for Stage-0 upper-bound calculation.

These tables encode what each target domain's success_fn + schema producer
*will* register once Stages 1-6 land, per cross-domain-transfer-suite-rollout.md
Sections 11.5.4 / 11.5.5.

When a Stage's runtime success_fn / schema producer ships, the corresponding
entry below should be replaced by a runtime check via:

    from harness.gymv_success import registered_success_fn_domains, get_predicate_vocab

For now this table is the static oracle.

Provenance per target:
- gymv:              harness/gymv_success.EFFECT_PREDICATE_TYPES (8 types, shipped)
- osworld:           rollout memo Sections 11.5.4 / 11.5.5 (Phase 5 deliverables)
- browser:           rollout memo Sections 11.5.4 / 11.5.5 (Phase 2 deliverables)
- visual_reasoning:  rollout memo Sections 11.5.4 / 11.5.5 (Phase 3 - mostly already shipped via VisualReasoningExecutor)
                     plus mined predicates from single_shot_lift.mine_single_shot_effects
- video:             superset of visual_reasoning (build_video_visual_registry is a strict superset)
                     plus video-specific predicates from rollout memo Section 11.5.4
"""

from __future__ import annotations

TARGET_PREDICATE_VOCAB: dict[str, frozenset[str]] = {
    "gymv": frozenset({
        "entity_value_increased",
        "entity_value_decreased",
        "entity_count_changed",
        "entity_appeared",
        "entity_disappeared",
        "attribute_changed",
        "cumulative_reward_increased",
        "phase_transitioned",
    }),
    "osworld": frozenset({
        "entity_appeared",
        "entity_disappeared",
        "attribute_changed",
        "phase_transitioned",
        "entity_count_changed",
        "task_status",
        "last_action",
        "actor_used_action",
        "visited_entity",
    }),
    "browser": frozenset({
        "entity_appeared",
        "attribute_changed",
        "phase_transitioned",
        "entity_count_changed",
        "task_status",
        "last_action",
        "actor_used_action",
        "visited_entity",
    }),
    "visual_reasoning": frozenset({
        "answer_emitted",
        "answer_matches_gold",
        "answer_diverged_from_gold",
        "entity_grounded",
        "entity_appeared",
        "entity_value_increased",
        "entity_value_decreased",
        "phase_transitioned",
    }),
    "video": frozenset({
        "answer_emitted",
        "answer_matches_gold",
        "answer_diverged_from_gold",
        "entity_grounded",
        "entity_appeared",
        "entity_value_increased",
        "entity_value_decreased",
        "phase_transitioned",
        "temporal_ordering_correct",
        "frame_referent_grounded",
    }),
}


TARGET_SLOT_TYPE_VOCAB: dict[str, frozenset[str]] = {
    "gymv": frozenset({
        "tracked_entity", "selectable_entity", "navigable_region",
        "container_entity", "goal_indicator",
        "enum", "effect_predicate", "any",
    }),
    "osworld": frozenset({
        "tracked_entity", "selectable_entity", "container_entity",
        "navigable_region", "goal_indicator", "textual_anchor",
        "enum", "effect_predicate", "any",
    }),
    "browser": frozenset({
        "tracked_entity", "selectable_entity", "container_entity",
        "textual_anchor", "goal_indicator",
        "enum", "effect_predicate", "any",
    }),
    "visual_reasoning": frozenset({
        "tracked_entity", "container_entity", "textual_anchor", "goal_indicator",
        "enum", "effect_predicate", "any",
    }),
    "video": frozenset({
        "tracked_entity", "container_entity", "textual_anchor", "goal_indicator",
        "enum", "effect_predicate", "any",
    }),
}


TARGET_DOMAINS: tuple[str, ...] = (
    "gymv", "osworld", "browser", "visual_reasoning", "video",
)
