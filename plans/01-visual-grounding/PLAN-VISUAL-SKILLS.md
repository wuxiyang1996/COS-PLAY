# PLAN: Visual Skills for Perception and Grounding (Optional)

**Scope:** Define transferable visual grounding *strategies* as skills — multi-step perception programs that compose tools to resolve targets, disambiguate candidates, recover lost entities, and collect evidence across domains. These sit between raw perception tools and reasoning/action skills. Optional extension to the core pipeline.

**Upstream:** Structured `<state>` schemas from [Visual Grounding](PLAN-VISUAL-GROUNDING.md); perception tool registries (visual, video, cross-frame).
**Downstream:** Grounding-quality improvements consumed by [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) inner MDP; grounding patterns consumed by [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) for contract learning; cross-domain perception transfer consumed by [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md).

**Status:** Design proposal. Not required for the core pipeline — the system works without this layer (see [Visual Grounding §12](PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract) for the existing "skills for reasoning only, not grounding" rationale). This plan extends that design for scenarios where grounding itself requires reusable multi-step strategies.

---

## 0. Core principle

**Transfer happens at the level of state transitions and effect contracts, not at the level of raw actions or screenshots.**

A skill should be a state-transition program with evidence — not a text instruction like "click submit" or "move left." If visual grounding already produces a unified structured state, then each skill should live in that same space: when it applies, what it tries to achieve, what evidence it needs, what abstract operations it performs, what state changes it expects, how to know it succeeded or should abort.

For cross-domain transfer to work, the skill format must be grounded in the shared state schema rather than in domain-specific action names.

### 0.1 Implementation priority (skills stay general protocols)

Every visual / grounding skill in this plan is a **general protocol feasible across game, webagent, os-agent, video-understanding, and visual reasoning** (see [Skill Bank §0.1](../03-skill-bank/PLAN-SKILL-BANK.md#01-general-protocol-invariant-no-domain-specific-skill-families)). The skill format, the effect families (§4), and the cross-domain ontology (§5) apply to all five target domains, and no protocol in this plan is admitted unless it is feasible across all of them via adapter binding.

What narrows is **implementation order for adapters and replay slices**, not skill content:

- Short-video (Video-Holmes-style) is the first **evaluation arena** where general protocols like `collect_evidence_chain`, `disambiguate_target`, `locate_filter_select`, and `actor_action_binding` get their first `verified_domains` entry.
- Adapters for game / webagent / os-agent / visual reasoning exist from day one (each protocol carries the full five-domain adapter contract); they are populated and replay-verified in a staggered order, but the protocols themselves do not change.
- A protocol that turns out to only work on short video is not kept as a "short-video skill" — it is flagged as a failed transfer candidate and recorded in the originating skill's `known_failure_modes` / `do_not_transfer_if` (see [Skill Bank §4.3b](../03-skill-bank/PLAN-SKILL-BANK.md)).

### 0.2 Cross-domain `candidate_set` — one abstraction, five bindings

The shared slot `candidate_set` is the canonical example of how a single semantic skill carries across all target domains. The *slot* does not change; only its adapter-provided members do.

| Domain | Typical `candidate_set` members | "Select among candidates" looks like |
|--------|---------------------------------|--------------------------------------|
| Game | Legal moves / units / tiles at the current state | Choose the relevant move given goal + constraints |
| Webagent | UI controls discovered in the DOM / screenshot | Select the relevant control (button, link, input) |
| OS-agent | Windows / files / desktop objects in focus | Isolate the relevant window or object among desktop entities |
| Video understanding | Temporal moments / frames / clip segments | Pick the key moment for a claim (evidence frame) |
| Visual reasoning | Objects / regions / text spans in the image | Isolate the answer-bearing object/region |

A skill written over `candidate_set` plus a filter/role criterion works in all five without rewriting its protocol — the adapter supplies the set and the domain-specific effect realization.

---

## 1. Why extend skills to visual grounding?

### What the current design already does well

The existing [Visual Grounding §12](PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract) correctly separates perception (SFT/distillation) from reasoning (skills + GRPO). The inner MDP's GROUND action already lets the reasoning agent extend the schema when information is missing. Tool-loop traces are already mined for reasoning skill templates via the transferable skill extraction pipeline ([Skill Bank §9](../03-skill-bank/PLAN-SKILL-BANK.md#9-transferable-skill-extraction)).

### What it doesn't capture

Some grounding challenges are not single-tool calls but *multi-step perception strategies* — sequences of tool calls, checks, and decisions that recur across domains:

- **Disambiguating multiple candidates** — multiple similar-looking buttons, icons, game objects, or video subjects
- **Recovering a lost target** — entity was visible, disappeared due to occlusion/scrolling/scene change
- **Grounding by text anchor** — locating a visual element by reading nearby text first
- **Grounding by spatial relation** — finding target by its position relative to a known anchor
- **Localizing a key temporal moment** — narrowing down when an event happened in a video
- **Verifying interactability** — confirming a grounded target is actually actionable
- **Collecting multi-modal evidence** — combining text, position, color, and action clues into a single grounded target

These patterns share a critical property: they are the same strategy whether the domain is browser, desktop, game, image, or video. A "disambiguate multiple candidates" strategy looks structurally identical regardless of whether the candidates are buttons, icons, game items, or people in a video.

### The key distinction

Not everything in visual grounding should become a skill:

| Layer | Skill? | Training | Examples |
|-------|:------:|----------|----------|
| **Perception primitives** (tools) | No | Fixed / specialist models | `detect_objects`, `OCR`, `spatial_query`, `track_object`, `crop/zoom` |
| **Grounding strategies** (this plan) | Yes | Discoverable + transferable | `disambiguate_target`, `recover_lost_target`, `ground_by_text_anchor` |
| **Reasoning / action skills** (existing) | Yes | GRPO-trained | `constraint_satisfaction`, `blocker_resolution`, `evidence_based_conclusion` |

Perception primitives are atomic, domain-agnostic tools with clear I/O. Grounding strategies are *programs over those tools* — multi-step procedures with preconditions, branching, and success/failure conditions. This plan captures the middle layer.

---

## 2. Two kinds of skill effects

The existing skill bank defines skills by their effects on the world (eff_add, eff_del, eff_event). Visual grounding skills produce a different kind of effect — changes to the agent's *cognitive state* rather than the environment state.

| Effect type | Changes | Examples | Skill bank |
|-------------|---------|----------|------------|
| **World-effect skills** | External environment state | `selected(target)=true`, `opened(container)=true`, `distance(agent,target) decreases` | Existing [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) |
| **Belief/grounding-effect skills** | Internal binding / evidence state | `binding(target)=resolved`, `confidence(target)≥τ`, `candidate_count=1`, `evidence_collected=true` | This plan (new) |

A grounding skill's "effect" is not a world state change — it is a cognitive state change: a target gets uniquely bound, ambiguity is resolved, confidence rises above a threshold, evidence is collected, a temporal window is identified.

This distinction matters for contracts: grounding skill contracts use predicates like `binding()`, `confidence()`, `candidate_count()`, `evidence_chain_length()` rather than `selected()`, `opened()`, `moved()`.

**`evidence_role` mapping ([PLAN-SKILL-BANK.md §0.3 Clause B](../03-skill-bank/PLAN-SKILL-BANK.md#03-evidence-driven-invariant-no-opaque-skills)).** Visual / grounding skills fall into two evidence roles and only two:

| Skill kind | `evidence_role` | Required episode fields |
|-----|-----|-----|
| Grounding / localization / inspection / segmentation / temporal-window discovery | `GATHER` | `evidence_out ≠ ∅` — the produced [`GroundingRecord`](PLAN-VISUAL-GROUNDING.md) is the canonical `evidence_out` |
| Anchor / consistency / constraint / sufficiency checks over grounded evidence | `VERIFY` | `evidence_in ≠ ∅`; `verify_verdict ∈ {PASS, FAIL, INSUFFICIENT}` |

A visual skill that selects an answer with a cited evidence chain is not a `GATHER`/`VERIFY` skill in this plan — it is a `COMMIT` skill living in the main [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) that *consumes* `evidence_in` produced by the `GATHER`/`VERIFY` skills here. This separation keeps visual-skill semantics cleanly on the evidence-production side and prevents opaque "look-then-act" macros from entering this plan.

---

## 3. Unified skill format for cross-domain transfer

Each skill — whether grounding, reasoning, or action — follows the same structural template. Cross-domain transfer works because this format is defined over the shared `<state>` schema, not over domain-specific objects.

### 3a. Skill signature

Domain-agnostic identity:

```yaml
skill_id: acquire_target
name: Acquire Target
evidence_role: GATHER           # §2 mapping; required by Skill Bank §0.3
category: acquisition           # task-effect family (§4); must be consistent with evidence_role
intent_tag: bind_and_focus      # what the skill is trying to achieve
```

Names should be domain-agnostic: `acquire_target`, `open_container`, `navigate_to_region`, `disambiguate_candidate`, `verify_goal_state` — not `click_submit_button`, `double_click_icon`, `jump_over_pipe`.

### 3b. Typed slot variables

Instead of hardcoding domain objects, skills use typed slots that bind at runtime from grounded entities:

```yaml
slots:
  target:
    type: selectable_entity
  context_region:
    type: optional_region
  anchor_text:
    type: textual_anchor
  obstacle:
    type: blocking_entity
```

Slot types form a small cross-domain ontology (§5). The same skill transfers when different domain objects map to the same slot type.

### 3c. Preconditions

Predicates over the structured state — not domain-specific descriptions:

```yaml
preconditions:
  - exists(target)
  - visible(target) OR inferable(target)
  - not selected(target)
  - reachable(target)
  - absent(blocking_obstacle)
```

These reference shared schema fields: entity types, attributes, spatial relations, temporal relations, task constraints.

### 3d. Abstract plan

A small set of abstract operators, not raw environment actions:

```yaml
plan:
  - inspect(context_region OR around(target))
  - disambiguate_if_needed(target)
  - focus(target)
  - select(target)
```

Abstract operators include: `focus`, `approach`, `inspect`, `select`, `open`, `track`, `read`, `compare`, `wait_until`. Each domain maps these to concrete actions via an adapter (§6).

### 3e. Effects

The expected state transition — the most important part for transfer:

```yaml
effects:
  - selected(target) = true
  - focused(target) = true
```

Two skills are the same abstract skill if they produce the same effects, even when their low-level actions differ entirely (`click` vs `double_click` vs `move_and_interact` vs `find_and_track`).

### 3f. Success, abort, and evidence

```yaml
success:
  - selected(target) = true
abort:
  - not exists(target)
  - unreachable(target)
  - repeated_failure(select(target))
evidence_required:
  - localization(target)
  - action_affordance(target)
```

### 3g. Domain adapters

Per-domain execution realizations:

```yaml
domain_adapters:
  browser:
    - move_cursor(target)
    - click(target)
  os:
    - move_cursor(target)
    - double_click_if_needed(target)
  game:
    - move_to(target)
    - interact(target)
  video:
    - find_moment(target)
    - localize(target)
    - track(target)
```

The semantic contract is shared. Only the execution adapter varies by domain.

### 3h. Full example

```yaml
skill_id: acquire_target
summary: Bind to a target entity and bring it into actionable focus.
category: acquisition
intent_tag: bind_and_focus

slots:
  target:
    type: selectable_entity
  context_region:
    type: optional_region

preconditions:
  - exists(target)
  - visible(target) OR inferable(target)
  - not selected(target)

plan:
  - inspect(context_region OR around(target))
  - disambiguate_if_needed(target)
  - focus(target)
  - select(target)

effects:
  - selected(target) = true

success:
  - selected(target) = true

abort:
  - not exists(target)
  - unreachable(target)
  - repeated_failure(select(target))

evidence_required:
  - localization(target)
  - affordance(target)

domain_adapters:
  browser:
    - move_cursor(target)
    - click(target)
  os:
    - move_cursor(target)
    - double_click_if_needed(target)
  game:
    - move_to(target)
    - interact(target)
  video:
    - find_moment(target)
    - localize(target)
    - track(target)

transfer_hints:
  - browser/button
  - os/icon
  - game/item
  - video/person_or_object_track
```

---

## 4. Effect families

Skills organized by the kind of state change they create — not by domain. This taxonomy is the primary axis for cross-domain transfer.

| Family | State change | Grounding example | Action example |
|--------|-------------|-------------------|----------------|
| **Acquisition** | Bring target into focus / possession / control | `disambiguate_target`, `recover_lost_target` | `select_element`, `pick_up_item` |
| **Navigation** | Move attention or agent to region | `ground_by_spatial_relation`, `navigate_attention_to_region` | `scroll_to_section`, `move_to_area` |
| **Inspection** | Reveal hidden information | `ground_by_text_anchor`, `collect_supporting_evidence` | `open_details_panel`, `read_tooltip` |
| **Manipulation** | Change target state | — | `toggle_checkbox`, `push_object` |
| **Verification** | Test whether desired state holds | `verify_target_interactable`, `confirm_binding_correct` | `verify_goal_state`, `check_completion` |
| **Disambiguation** | Resolve multiple candidate bindings | `disambiguate_similar_candidates`, `filter_candidate_set` | `resolve_ambiguous_reference` |
| **Tracking** | Maintain entity identity over time | `track_entity_across_frames`, `reacquire_after_occlusion` | `follow_moving_target` |
| **Recovery** | Restore after failure / loss | `recover_target_after_loss`, `reground_stale_context` | `retry_failed_action`, `reposition_after_error` |

Grounding skills primarily populate: Acquisition, Navigation, Inspection, Verification, Disambiguation, Tracking, Recovery. Action skills primarily populate: Acquisition, Navigation, Manipulation, Verification.

---

## 5. Cross-domain entity ontology

For slot types to enable transfer, domain-specific grounded elements must map into a shared ontology:

| Ontology type | Browser | Desktop/OS | Game | Video |
|---------------|---------|------------|------|-------|
| `selectable_entity` | Button, link, menu item | Icon, file, window control | Tile, item, character | Person, object in frame |
| `interactive_entity` | Form field, toggle, slider | Text input, settings control | Lever, door, switch | — |
| `container_entity` | Dropdown, accordion, modal | Folder, window, panel | Chest, room, inventory | Scene, segment |
| `textual_anchor` | Label, heading, placeholder | Window title, menu text | Score display, dialogue | Subtitle, caption, overlay text |
| `navigable_region` | Page section, tab panel | Desktop area, window | Map area, level zone | Timeline segment |
| `tracked_entity` | Session element (across pages) | Persistent window/file | Moving character/object | Person across cuts |
| `goal_indicator` | Success message, checkmark | Task complete notification | Score threshold, goal flag | Answer evidence |
| `blocking_entity` | Modal, disabled state, error | Permission dialog, lock | Wall, enemy, obstacle | Occlusion, cut |

This ontology is the bridge. Without it, a browser button and a game lever are unrelated objects — with it, both are `interactive_entity` instances that the same abstract skill can operate on.

### Mapping rules

1. Each grounded entity from the `<state>` schema maps to one or more ontology types based on its attributes (`e*.state`, `e*.type`).
2. Mapping can be heuristic (attribute-based rules) or learned (LLM-based slot binding — see [Skill Bank §14 TODO](../03-skill-bank/PLAN-SKILL-BANK.md#14-todo)).
3. Ambiguous mappings produce multiple candidate bindings, handled by the `disambiguate_candidate` skill family.

---

## 6. Separating semantics from execution

Every skill has two layers:

### Semantic contract (shared across domains)

- Preconditions (predicates over shared schema)
- Effects (state transition predicates)
- Success / abort criteria
- Evidence requirements
- Typed slot variables

### Execution adapter (varies per domain)

- How abstract operators map to concrete actions
- Domain-specific tool calls
- Timing / sequencing constraints

| Abstract operator | Browser | Desktop/OS | Game | Video |
|-------------------|---------|------------|------|-------|
| `select(target)` | `click` | `click` / `double_click` | `move_to + interact` | `localize + track` |
| `inspect(region)` | `scroll_to + read` | `hover + read_tooltip` | `look_at + describe` | `sample_frames + describe_region` |
| `focus(target)` | `scroll_into_view + highlight` | `bring_to_front` | `center_camera` | `seek_to_frame` |
| `approach(target)` | `navigate_to_url` / `scroll` | `open_folder_path` | `pathfind + move` | `temporal_navigate` |
| `track(target)` | `observe_dom_changes` | `watch_window` | `follow_entity` | `track_object` |
| `read(anchor)` | `OCR` / `get_text_content` | `OCR` / `read_a11y` | `OCR` / `read_text_region` | `read_text_in_frame` |

Transfer means: reuse the semantic contract, write a new execution adapter.

---

## 7. Grounding skill bank

### Proposed grounding skills

These are the multi-step grounding strategies that recur across domains, defined using the format from §3.

| Skill | Trigger | Protocol (tool chain) | Effect |
|-------|---------|----------------------|--------|
| `disambiguate_target` | `candidate_count(target_type) > 1` | GROUND(candidate set) → RETRIEVE(anchor text) → CHECK(spatial relation with anchor) → COMMIT(best candidate) | `binding(target)=resolved`, `candidate_count=1` |
| `recover_target_after_loss` | `was_visible(target) AND not visible(target)` | GROUND(last_known_region) → CHECK(occlusion/scroll/scene_change) → GROUND(expanded_search) → COMMIT(reacquired OR lost) | `visible(target)=true` OR `abort: target_permanently_lost` |
| `ground_by_text_anchor` | `target_label=ambiguous AND nearby_text_exists` | GROUND(text_anchors) → CHECK(proximity to target) → COMMIT(target identity from text context) | `binding(target)=resolved`, `confidence(target)≥high` |
| `ground_by_spatial_relation` | `target_position=uncertain AND anchor_entity_known` | GROUND(anchor_entity) → CHECK(spatial relation) → COMMIT(target position from relation) | `localization(target)=precise` |
| `localize_key_moment` | `domain=video AND temporal_window=unknown` | GROUND(scene boundaries) → CHECK(candidate moments) → GROUND(detail at candidate) → COMMIT(event_timestamp) | `temporal_binding=resolved`, `evidence_timestamp=set` |
| `verify_target_interactable` | `target_bound AND action_pending` | CHECK(target.state != disabled) → CHECK(no blocking_entity) → CHECK(affordance matches action) → COMMIT(interactable=true/false) | `affordance(target)=confirmed` OR `abort: not_interactable` |
| `collect_supporting_evidence` | `evidence_chain_length < required` | GROUND(additional entities) → CHECK(relations to target) → RETRIEVE(prior observations) → COMMIT(evidence sufficiency) | `evidence_collected=true`, `confidence(answer)≥τ` |

### Grounding skill contracts

Grounding skills use the same contract structure as reasoning skills, but with belief/binding predicates:

```yaml
skill_id: disambiguate_target
summary: Resolve ambiguous target binding when multiple candidates match.
category: disambiguation
intent_tag: resolve_ambiguity

slots:
  target:
    type: selectable_entity
  candidate_set:
    type: list[selectable_entity]
  text_anchor:
    type: textual_anchor
    required: false

preconditions:
  - candidate_count(target_type) > 1
  - ambiguous_reference(target)

plan:
  - GROUND(candidate_set)
  - RETRIEVE(relevant_anchor_text)
  - CHECK(spatial_relation_with_anchor)
  - COMMIT(best_candidate)

effects:
  - binding(target) = resolved
  - confidence(target) >= high
  - candidate_count(target) = 1

success:
  - unique_binding(target)

abort:
  - no_candidate_after_filter
  - all_candidates_equally_ambiguous

evidence_required:
  - localization(all candidates)
  - distinguishing_feature OR spatial_context

domain_adapters:
  browser:
    - detect_objects(page_region)
    - read_text_region(nearby_labels)
    - spatial_query(candidate, label)
  game:
    - detect_objects(viewport)
    - describe_region(around each candidate)
    - compare(candidate_attributes)
  video:
    - grounded_detect(target_description)
    - track_object(each candidate, short_window)
    - compare_elements(candidates)
```

---

## 8. Three-layer skill bank hierarchy

The skill bank organizes into three layers to support both transfer and domain-specific robustness:

### Layer 1: Abstract transferable skills

Shared across all domains. Defined by semantic contracts only.

```
acquire_target
inspect_region
navigate_to_goal
verify_condition
disambiguate_candidate
track_entity
open_reveal_interact
recover_after_loss
collect_evidence
localize_temporal_event
```

### Layer 2: Domain adapters

Per-domain execution realizations of Layer 1 skills:

```
browser.acquire_target
os.acquire_target
game.acquire_target
video.acquire_target
```

### Layer 3: Environment-specific tactics

Concrete low-level wrappers:

```
browser.acquire_target.button_click
browser.acquire_target.link_follow
os.acquire_target.icon_doubleclick
os.acquire_target.drag_to_target
game.acquire_target.walk_then_interact
game.acquire_target.ranged_select
video.acquire_target.find_then_track
video.acquire_target.temporal_interpolate
```

### How the actor uses them

The decision process follows a three-step selection:

1. **What effect is needed?** → select abstract skill (Layer 1)
2. **Which entities fill the slots?** → bind from current grounded state
3. **Which adapter executes it here?** → select domain adapter (Layer 2), fall back to specific tactic (Layer 3)

This is more transferable than picking from a flat bank of environment-specific skills.

---

## 9. Automatic skill discovery from trajectories

Once the system produces unified grounded states, grounding skills should be discovered from state transitions — not from language descriptions.

### Step A: Segment trajectories

Break at points where:

- Key predicates change (binding resolved, target lost, candidate set changes)
- Intention changes (grounding → reasoning → action)
- Tool usage pattern changes (detection → spatial → OCR)
- Interaction target changes
- Success / failure boundaries

### Step B: Summarize each segment

For each segment, extract:

- Pre-state predicates (binding status, confidence, candidate count)
- Post-state predicates (same fields after segment)
- Bound slots (which entities were involved)
- Tool calls used
- Duration / success
- Evidence chain

### Step C: Cluster by effect pattern

Cluster segments primarily by:

- Similar effect predicates
- Similar slot types
- Similar success conditions

**Not** by raw action tokens. `click button`, `double click icon`, and `move and interact` all collapse into one abstract skill if they produce the same effect (`selected(target)=true`).

### Step D: Abstract skill + domain adapters

From each cluster, produce:

- One abstract skill (Layer 1) with the shared semantic contract
- Per-domain adapters (Layer 2) capturing domain-specific tool sequences
- Environment-specific tactics (Layer 3) for fine-grained variants within a domain

### Integration with existing extraction pipeline

This discovery process extends [Skill Bank §9 (Transferable Skill Extraction)](../03-skill-bank/PLAN-SKILL-BANK.md#9-transferable-skill-extraction):

- Stage A (Predicate normalization) → also normalizes grounding predicates (`binding()`, `confidence()`, `candidate_count()`)
- Stage B (Structural clustering) → clusters include grounding segments alongside reasoning segments
- Stage C (Template abstraction) → produces `TransferableSkill` with grounding-specific protocol hops
- Stage D (Transferability scoring) → domain coverage includes grounding-only domains (image QA, video QA)

---

## 10. Relationship to existing plans

### Reconciliation with Visual Grounding §12

[Visual Grounding §12](PLAN-VISUAL-GROUNDING.md#12-schema-completeness-guarantee-grounding--reasoning-contract) states: "Skills live in the reasoning/action layers, not in grounding. Grounding is perception (SFT/distillation); reasoning is strategy (skills + GRPO)."

This plan does **not** contradict that design decision. It extends it with a nuance:

| What | Skill? | Rationale |
|------|:------:|-----------|
| Perception *tools* (detect, OCR, track, crop) | No | Atomic operations with clear I/O — stay as tools |
| Grounding *strategies* (disambiguate, recover, verify, ground-by-relation) | Yes (this plan) | Multi-step programs with preconditions, branching, and contracts — reusable across domains |
| Reasoning *skills* (constraint_satisfaction, blocker_resolution) | Yes (existing) | Strategic decision-making in the inner MDP |

The existing inner MDP GROUND action already invokes grounding tools. Grounding skills from this plan provide *reusable templates for how to GROUND effectively* — they are to GROUND what reasoning skills are to the full hop chain.

### Relationship to Skill Bank

Grounding skills enter the [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) through the same pipeline:

- **Discovery:** Stages 1–2 (boundary proposal, segmentation) extended to grounding segments
- **Contracts:** Stage 3 learns grounding contracts (binding/confidence predicates)
- **Maintenance:** Stage 4 proposes/filters/executes on grounding skills
- **Query:** `SkillQueryEngine` serves grounding skills alongside reasoning skills

The bank stores both kinds of skills with a `skill_type` tag: `"reasoning"` or `"grounding"`.

### Relationship to Skill Crafter

The [Skill Crafter](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) applies its three creation modes to grounding skills:

- **Composer:** Chain grounding skills (`disambiguate → verify → select`)
- **Generalizer:** Transfer grounding strategies across domains (browser disambiguation → game disambiguation)
- **Hypothesizer:** Propose new grounding strategies from visual failures
- **Failure Reflector:** Diagnose grounding failures (`grounding_error` in §6.2 taxonomy) and propose grounding-specific recovery

### Relationship to Action Agent

The [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) inner MDP can invoke grounding skills when GROUND hops are needed:

```
Schema arrives with <uncertainty> e5.label=high
    ↓
Inner MDP: grounding skill "disambiguate_target" triggered
    ↓
hop1: GROUND(candidate set) via detect_objects
hop2: CHECK(spatial relation) via spatial_query
hop3: COMMIT(best candidate)
    ↓
Schema updated: binding(e5)=resolved, confidence=high
    ↓
Reasoning skill continues from updated schema
```

---

## 11. How the synthesis-reflection agent helps with transfer

The frozen 32B/72B synthesis-reflection agent ([Skill Crafter §2](../04-skill-crafter/PLAN-SKILL-CRAFTER.md#2-architecture)) is critical for grounding skill transfer. Its offline tasks include:

### Identifying "new skill vs. new adapter"

The most important question for transfer. Given:

- Browser: click a button to select it
- Game: walk to an object and interact with it
- Video: find a person in a frame and track across cuts

The synthesis agent should determine: are these three different skills, or one abstract `acquire_target` with three domain adapters?

**Decision criteria:**

- If preconditions are structurally similar → same abstract skill
- If effects are structurally similar → same abstract skill
- If success criteria are structurally similar → same abstract skill
- If only the tool sequences differ → different adapters, same skill

### Merging skills with the same effect pattern

When two grounding skills from different domains produce the same belief-state change, they should be merged into one abstract skill. The synthesis agent proposes merges; the acceptance gate ([Skill Bank §7](../03-skill-bank/PLAN-SKILL-BANK.md#7-grpo-co-evolution)) verifies.

### Diagnosing transfer failures

When a grounding skill transfers from browser to game and fails, the failure is rarely "wrong skill." It is more often:

- Slot type mapping is wrong (text anchor doesn't exist in games)
- Preconditions are too narrow/wide for the target domain
- Adapter is incomplete (missing a tool call)
- Effect predicates use domain-specific thresholds

The synthesis agent should prioritize fixing the abstract contract or adapter, not inventing a new skill.

---

## 12. What usually goes wrong

| Failure mode | Consequence | Prevention |
|-------------|-------------|------------|
| Skills stored too close to raw actions | Nothing transfers | Define skills by effects, not by tool sequences |
| Slots too domain-specific (`submit_button`, `folder_icon`) | Cross-domain matching fails | Use ontology types (§5) |
| Effects underspecified ("completed sub-task") | Cannot tell whether two skills are equivalent | Explicit predicates: `binding()`, `confidence()`, `candidate_count()` |
| Semantics and execution mixed in one object | Brittle, unmergeable skills | Strict two-layer separation (§6) |
| All visual tools promoted to skills | Skill bank inflates into a tool menu | Only multi-step strategies become skills; atomic tools stay as tools |
| Grounding and reasoning skills mixed without distinction | Reward and verification confused | Tag with `skill_type`, use appropriate contract predicates |

---

## 13. Rollout order

This plan is optional and depends on the core pipeline being functional.

**Phase 0 — Catalog grounding patterns (after Visual Grounding Phase 2)**
1. Review tool-loop traces from all domains.
2. Identify recurring multi-step grounding patterns manually.
3. Write 5–7 grounding skill templates by hand.
4. Validate: do these patterns actually recur across domains?

**Phase 1 — Grounding skill format and contracts**
1. Define the grounding contract schema (binding/confidence predicates).
2. Extend `TransferableSkill` to support grounding-type effects.
3. Extend `SkillBankMVP` storage to accept `skill_type="grounding"`.

**Phase 2 — Automatic discovery**
1. Extend trajectory segmentation to identify grounding segments (tool-call-heavy segments with belief-state changes).
2. Cluster grounding segments by effect pattern.
3. Abstract into Layer 1 / Layer 2 / Layer 3 hierarchy.

**Phase 3 — Cross-domain transfer**
1. Transfer grounding skills between browser and game.
2. Transfer between static (image QA) and temporal (video QA) domains.
3. Evaluate: does a grounding skill learned from browser disambiguation help game target acquisition?

**Phase 4 — Integration with inner MDP**
1. Allow `hop_select` to invoke grounding skills (not just individual GROUND tool calls).
2. Measure: does grounding-skill invocation improve schema completeness compared to ad-hoc GROUND hops?

---

## 14. TODO

| Task | Priority | Status |
|------|----------|--------|
| Catalog recurring grounding patterns from tool-loop traces | P1 | Not started |
| Define grounding contract schema (binding/confidence predicates) | P1 | Not started |
| Write 5–7 grounding skill templates manually | P1 | Not started |
| Extend `TransferableSkill` for grounding-type effects | P2 | Not started |
| Cross-domain entity ontology mapping rules | P2 | Not started |
| Extend trajectory segmentation for grounding segments | P2 | Not started |
| Grounding effect clustering pipeline | P2 | Not started |
| Three-layer skill bank hierarchy implementation | P2 | Not started |
| Domain adapter registry (abstract operator → concrete tool call) | P2 | Not started |
| Integration with `hop_select` for grounding skill invocation | P3 | Not started |
| Cross-domain grounding transfer evaluation | P3 | Not started |
| Synthesis-reflection agent: skill-vs-adapter decision procedure | P3 | Not started |

---

## 15. Reference

- Existing skill bank plan: [PLAN-SKILL-BANK.md](../03-skill-bank/PLAN-SKILL-BANK.md) — transferable skill extraction (§10), data model (§3)
- Visual grounding plan: [PLAN-VISUAL-GROUNDING.md](PLAN-VISUAL-GROUNDING.md) — schema (§3), tool registries (§9), schema completeness (§12)
- Action agent plan: [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) — inner MDP (§5), GROUND action, uncertainty-driven triggering (§10)
- Skill crafter plan: [PLAN-SKILL-CRAFTER.md](../04-skill-crafter/PLAN-SKILL-CRAFTER.md) — composition (§3), generalization (§4), failure reflection (§6)
- System plan index: [`plans/README.md`](../README.md)
