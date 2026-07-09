# PLAN: Uncertainty as a Calibrated System-Level Signal

**Status:** Cross-cutting contract for uncertainty across grounding, actor, harness, orchestrator, and evaluation.
**Owner:** Pipeline Orchestrator (defines the contract); Visual Grounding + Action Agent + Harness (produce / consume).
**Companions:** [PLAN-VISUAL-GROUNDING.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md) (per-field `<uncertainty>`), [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) (Path A / B / C routing), [PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md) (uncertainty-driven GROUND), [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) (gating), [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (replay + promotion), [PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md) (evaluation slices).

---

## 1. Why uncertainty needs its own contract

Today, "uncertainty" appears in several plans but means different things:

- Visual Grounding emits a per-field `<uncertainty>` block that the staged pipeline already uses to choose Path A (accept), Path B (tool repair), or Path C (offline escalation) ([PLAN-VISUAL-GROUNDING-MILESTONES.md §Path A/B/C](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)).
- The Action Agent uses an `uncertainty` flag to decide whether to fire an extra `GROUND` / `CHECK` hop inside the inner MDP.
- The first-target evaluation contract carries a single self-reported `uncertainty: float` field on the `support_package` ([PLAN-EVAL-FIRST-TARGET.md §2.3](../00-system/PLAN-EVAL-FIRST-TARGET.md)).
- The Harness uses ad hoc thresholds inside individual gates.

These are five different signals, on different scopes (field / entity / state / evidence / answer), produced by different components, and consumed without a shared meaning. There is no shared definition of what "0.7 confidence" means, no calibration check, no per-domain tuning, and no rule that ties confidence to compute spend.

**The contract this plan introduces:**

1. Confidence is reported at well-defined **scopes** (field, entity, state, evidence, answer).
2. Confidence is split by **source** (parser, validator, tool, cross-view, missing-field).
3. Confidence drives **routing decisions** through documented thresholds, not magic numbers.
4. Confidence is **calibrated** with reliability curves and ECE per slice.
5. Downstream consumers (actor, harness, replay, orchestrator, eval) read this contract instead of inventing local thresholds.

The driving design principle: **the system should only "pay more" — extra tools, more hops, slower verifier, large-model escalation — when uncertainty is high.** Without a calibrated signal, the system either under-spends and accepts wrong answers, or over-spends uniformly and burns budget.

This plan keeps the first version deliberately small: coarse labels, a reliability curve, and a single routing table. It is sized to ship inside the existing staged grounding pipeline and `support_package` contract, not to replace them.

---

## 2. Uncertainty levels and granularity

Five named scopes. Each one is produced by a specific layer and consumed by a specific layer. They are **related but not identical** — a parse can have low `field_confidence` on one attribute yet high `answer_confidence` because the answer does not depend on that field.

| Scope | Produced by | Granularity | What it measures |
|---|---|---|---|
| `field_confidence` | Schema parser (`schema_gen`) | One per `<entity>.<attribute>` (e.g. `e3.label`, `e7.pos`) | How sure the parser is about that single slot. Mirrors the existing per-field `<uncertainty>` block. |
| `entity_confidence` | Schema parser + validator | One per entity in `<entities>` | Aggregate over the entity's fields, plus existence ("does this entity actually exist in the frame"). |
| `state_confidence` | Validator + cross-view check | One per `<state>` snapshot | Whole-snapshot reliability: schema-completeness, internal consistency, cross-view stability. |
| `evidence_confidence` | Inner-MDP loop (GROUND / CHECK / RETRIEVE results) | One per `EvidenceRef` (frame id, region, tool-call id) | How well the cited evidence supports the claim it is attached to — strength of the warrant, not strength of the parse. |
| `answer_confidence` | Actor at COMMIT time | One per `final_answer` | End-to-end self-reported confidence in the committed answer, conditioned on accumulated evidence and reasoning steps. |

**Why all five and not one number:**

- A confident parse of an irrelevant entity contributes nothing to answer confidence. Collapsing them hides bugs (high `state_confidence`, low `answer_confidence` → reasoning failure, not parsing failure).
- Routing decisions live at different scopes: Path A/B/C is a **state-level** decision; "fire one more GROUND hop on entity e5" is an **entity-level** decision; "accept this evidence ref into the warrant" is an **evidence-level** decision; "commit vs keep reasoning" is an **answer-level** decision.
- Calibration only makes sense when you tie a probability to a measurable outcome. Each scope has its own outcome (field correct vs not, entity present vs not, snapshot consistent vs not, evidence supports claim vs not, answer correct vs not), so each must be calibrated against its own outcome.

**Hierarchical aggregation rule (default, overridable):**

```
entity_confidence(e)  = min over required fields of field_confidence(e.f)   # weakest link
state_confidence(s)   = weighted_mean(entity_confidence) * schema_complete(s)
evidence_confidence(r)= claim_support_score(r) * state_confidence(r.state)
answer_confidence(a)  = f(evidence_confidence over warrant, hop_trace, parser_priors)
```

The aggregator is replaceable; `f` for `answer_confidence` starts as a small learned head over the warrant + trace, not a hand-rule.

---

## 3. Sources of uncertainty

Two answers with the same `state_confidence = 0.6` can need very different responses depending on **why** they are uncertain. The contract therefore tags each confidence value with a **source breakdown** (small dict, summed externally — not a single tag).

Five named sources:

1. **Parser intrinsic uncertainty** — the VLM parser's own self-reported confidence on a field/entity. Read directly from the head logits or from the existing `<uncertainty>` block. Weak under distribution shift; strong on in-distribution slices.
2. **Validator disagreement** — schema validator disagrees with the parser (missing required field, type violation, geometry impossible, label not in vocabulary). Binary or graded depending on the rule.
3. **Tool disagreement** — when a Path B tool (e.g. `detect_objects`, `describe_region`, OmniParser, GroundingDINO) is invoked, the disagreement between parser and tool on the same slot. Strong signal because it is independent of the parser's self-assessment.
4. **Cross-view / temporal inconsistency** — same entity / claim disagreed across frames, across crops, or across hop revisits. For video this is the dominant source; for single images it is dormant.
5. **Missing-required-field uncertainty** — schema declares a field required for the current task / hop, parser produced nothing or `null`. This is structurally different from a confident-but-wrong field and must be tracked separately.

Each `*_confidence` value carries a `sources: {parser: w1, validator: w2, tool: w3, crossview: w4, missing: w5}` weighting. Routing rules in §4 read these weights, not just the scalar.

**Why split:**

- High confidence with `sources={parser: 1.0}` and zero validator/tool input is much weaker evidence than high confidence with `sources={parser: 0.3, validator: 0.3, tool: 0.4}`. Routing must distinguish them.
- Different sources have different costs to reduce. Parser uncertainty needs more training data; validator disagreement needs schema fixes; tool disagreement needs better tool selection or arbitration; cross-view needs a temporal aggregator; missing-required-field is fixed by either a tool call or a schema-spec correction. Without source tags, we cannot pick the right repair.
- Calibration is per-source as well as per-domain (§5). Parser scores often miscalibrate one way (overconfident); tool agreement scores often miscalibrate the other way (under-confident on repaired entities).

---

## 4. Routing thresholds

Routing is **threshold on a calibrated confidence band**, not on a single number. Three bands, named consistently across the pipeline:

| Band | Default cutoff (state_confidence after calibration) | Meaning |
|---|---|---|
| `high` | ≥ 0.80 | Accept direct parse. No extra spend. |
| `medium` | 0.50 – 0.80 | Validate more. Run selective tool checks on the lowest-confidence fields/entities only. |
| `low` | < 0.50 | Escalate to heavier path. |

Mapping into the existing staged grounding pipeline:

| Band | Visual Grounding routing ([Milestones §Path A/B/C](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)) | Inner MDP behavior | Harness behavior |
|---|---|---|---|
| `high` | **Path A** — accept direct parse | Skip extra GROUND/CHECK hops; allow COMMIT | Accept without extra checks |
| `medium` | **Path B (selective)** — tool-repair only the low-confidence fields/entities | Fire one targeted GROUND or CHECK on the weakest entity | Run lightweight gates only |
| `low` | **Path C** — full tool loop or offline / slow escalation | Force GROUND + RETRIEVE; block COMMIT until uncertainty drops or budget exhausted | Run full gate stack; may veto |

Two important rules:

- **Per-scope routing.** Path A/B/C uses `state_confidence`. Inner-MDP per-hop routing uses `entity_confidence` of the targeted entity. Answer-commit gating uses `answer_confidence`. Evidence acceptance uses `evidence_confidence`. Do not collapse to a single number.
- **Per-domain thresholds.** Each domain (gymv, browser, os, image_qa, video_qa) has its own calibrated cutoff table. The default cutoffs above are the bootstrap values; calibration replaces them per domain in Phase 3.

**Source-aware override:** if `sources.missing > 0` for any required field, the snapshot is forced to `low` regardless of scalar value. Missing-required-field is structurally not "medium confidence."

---

## 5. Calibration metrics

A confidence value has a contract: "of all claims I marked at confidence p, p of them should be correct." This must be measured, not assumed.

**Required metrics (computed per scope, per domain, per source where applicable):**

- **Reliability curve.** Bin predictions by reported confidence (e.g. 10 bins). For each bin, plot reported confidence vs empirical accuracy. A perfectly calibrated model lies on the diagonal. Reported per scope (`field`, `entity`, `state`, `evidence`, `answer`).
- **Expected Calibration Error (ECE).** Weighted average gap between reported and empirical accuracy across bins. Single scalar per (scope, domain) cell. Target: ECE ≤ 0.05 on the answer scope on the first eval slice; looser early.
- **Per-domain calibration.** Separate reliability curves and ECE for each domain in {gymv, browser, os, image_qa, video_qa}. A model can be well calibrated overall and badly calibrated per slice.
- **Per-field calibration.** Inside the parser scope, separate reliability for the high-traffic fields: `entity.label`, `entity.pos`, `entity.state`, `relations.*`, `state_flags.*`. Field-level miscalibration is what corrupts entity- and state-level scores via the aggregator.
- **Correlation with downstream success.** For each scope, measure the rank correlation (Spearman) between confidence and downstream success — answer correctness for `answer_confidence`, gate-pass for `state_confidence`, claim-supported for `evidence_confidence`. A scope whose confidence does not correlate with its outcome is broken regardless of ECE.

**Reporting.** A single weekly **calibration dashboard** with: one reliability curve panel per (scope, domain) cell, an ECE matrix, a per-field heatmap, and the correlation table. Numbers are stored alongside `EpisodeTrace` so calibration is recomputable from logs.

**What we do not measure (yet):**

- No proper scoring rules (Brier, log loss) in the first cut — they are implied by ECE-style binning and we do not yet need them for routing decisions.
- No coverage / selective-prediction curves until Phase 3, where they become necessary for budget tuning.

---

## 6. Downstream effects

Once uncertainty is calibrated and split by scope/source, every downstream consumer reads the same contract.

**6.1 Actor behavior (inner MDP)**

- `entity_confidence(e) = low` on a referenced entity → force a `GROUND(e)` hop before COMMIT (within hop budget).
- `evidence_confidence(r) = low` on a candidate warrant → force a `CHECK(r)` or `RETRIEVE` hop before COMMIT.
- `answer_confidence < commit_floor[domain]` → block COMMIT, fall back to "ABSTAIN with best-effort + evidence" rather than emit a low-confidence answer (when the slice allows abstention).
- Source-aware: if `sources.tool > 0` on a high-confidence entity, the actor may skip the redundant GROUND it would otherwise fire on parser-only confidence.

**6.2 Harness gating**

- Gates already configurable per skill ([PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md)); add a per-gate **confidence floor**.
- `state_confidence` band selects which gates run: `high` → static + replay only; `medium` → + shadow on the suspect slice; `low` → full stack including non-regression on adjacent slices.
- Vetoes record `confidence_at_veto` so we can later ask "did we veto too aggressively in well-calibrated bands?"

**6.3 Replay prioritization**

- Replay queue is sorted by **expected information gain** ≈ `1 - max_band_score`. Episodes whose `state_confidence` lies near a band boundary (e.g. 0.78 — just under `high`) and whose downstream success is uncertain are replayed first; high-confidence-and-correct episodes are deprioritized.
- Episodes flagged with `sources.crossview > 0` (cross-view inconsistency) are also priority-replayed because they signal temporal aggregation bugs.

**6.4 Promotion / rollback decisions**

- Skill promotion ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) reads not just gate pass-rate but **confidence-conditioned pass-rate**. A skill that passes at `high` confidence but tanks at `low` confidence is promoted with a routing constraint ("usable only on `high` snapshots") rather than rejected outright.
- Rollback triggers on calibration drift: if ECE on a domain doubles week-over-week without other regressions, freeze promotions for that domain until investigated.

**6.5 Evaluation slices**

- The first-target eval ([PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md)) reports Joint Success Rate **bucketed by reported confidence band**. We need to see "JSR @ high / medium / low" separately, not only the pooled number.
- Add a **selective JSR**: accuracy on the subset where the system reported `high`. This is the metric most tied to user trust.
- Keep the existing single self-reported `uncertainty` field on `support_package`, but back it with `answer_confidence` from this contract; add `state_confidence` and `evidence_confidence` summaries as optional fields for richer slicing.

---

## 7. Minimal implementation order

Three phases. Each phase ends in a measurable artifact, not a refactor.

**Phase 1 — Coarse bands (1–2 weeks).**

- Emit only the three labels `low | medium | high` at every scope, derived from the existing per-field `<uncertainty>` and validator outputs via simple rules (mostly already implemented in the staged pipeline).
- Wire bands into Path A/B/C routing in Visual Grounding (already partially in place — formalize the cutoffs).
- Wire bands into the inner MDP GROUND/CHECK trigger and COMMIT block.
- Eval reports JSR bucketed by band.
- Acceptance: routing decisions made by band-derived rules match the current ad hoc rules within 5 %, and JSR-by-band table is in the weekly eval.

**Phase 2 — Numeric confidence (2–3 weeks).**

- Replace label-only output with numeric confidence in `[0, 1]` at each scope, plus the `sources` weight dict.
- Implement the aggregation rule in §2 (overridable per scope).
- Persist `field/entity/state/evidence/answer` confidences and `sources` in `EpisodeTrace`.
- Compute reliability curves and ECE per (scope, domain) on a fixed eval slice; do **not** retune cutoffs yet.
- Acceptance: dashboard renders reliability curves + ECE for all five scopes on at least two domains; correlation with downstream success > 0.3 on `answer_confidence`.

**Phase 3 — Calibration dashboard and per-domain tuned thresholds (3–4 weeks).**

- Add an isotonic-regression (or temperature-scaled) calibrator per (scope, domain), retrained weekly from `EpisodeTrace`.
- Replace the default cutoffs in §4 with per-domain tuned cutoffs chosen on the calibration set to maximize selective JSR at fixed `high`-band coverage (e.g. ≥ 60 % of episodes routed `high`).
- Wire confidence-conditioned gate selection (§6.2) and confidence-conditioned promotion (§6.4).
- Add replay prioritization by band-distance (§6.3).
- Acceptance: ECE ≤ 0.05 on `answer_confidence` per active domain; selective JSR @ `high` is at least + 8 points over pooled JSR; budget per correct answer drops vs Phase 2 baseline.

This order keeps the system shippable at every step. Phase 1 is a contract change with no new metric. Phase 2 adds metrics without changing routing. Phase 3 changes routing once metrics show it is safe to.

---

## 8. Risks / anti-goals

Things this plan deliberately **does not** do, and why:

- **Do not build a huge probabilistic framework first.** No Bayesian belief layer, no full graphical model, no learned uncertainty propagation network as the v1. The system needs a usable signal in 1–2 weeks, not a research project. Phases 2–3 leave room for a richer model later, but only after the simple version proves it routes better than the ad hoc rules.
- **Do not require perfect calibration before use.** Routing reads bands first, raw scores second. A miscalibrated score still routes correctly if the band cutoffs are tuned (Phase 3) — we use calibration to *tune the cutoffs*, not as a precondition for using the signal at all. ECE targets are aspirational, not gating, until Phase 3.
- **Do not overload a single `evidence_confidence` field with everything.** It is tempting to roll parser, validator, tool, cross-view, missing-field, and answer-correctness probability into one number. This collapses sources we need separately for repair (§3), and collapses scopes we need separately for routing (§4) and for calibration (§5). Keep the five scopes and the source dict distinct even when most consumers only read one of them.
- **No uncertainty-driven *training* in v1.** Confidence-weighted loss, selective annotation, active learning — all out of scope until Phase 3 lands. The first job is to make the signal trustworthy enough to gate compute; using it to gate gradients is later.
- **No abstention-as-default.** Abstention is allowed only on slices that explicitly support it ([PLAN-EVAL-FIRST-TARGET.md](../00-system/PLAN-EVAL-FIRST-TARGET.md)). The contract is "spend more when uncertain," not "answer less when uncertain."
- **No single global threshold.** Cutoffs are per (scope, domain), and the source-aware override (§4) can force a band downward. Anyone proposing "let's just set τ = 0.7 everywhere" is asking for a regression.

---

## 9. Open questions

Tracked here so they do not get lost between phases:

- How should `answer_confidence` be calibrated on **open-ended** slices, where "correct" is judge-defined rather than exact-match? Likely needs a separate per-judge calibrator.
- For video, should `state_confidence` aggregate across the temporal window, or be computed per-frame and then aggregated by the inner MDP? Phase 2 default is per-frame + max over warrant frames; revisit after data.
- Where do **frozen-teacher (32B/72B)** confidence outputs slot in — as a sixth source in §3, or as their own scope? Provisionally: a sixth source `teacher_disagreement`, only populated when the teacher was actually invoked.
