# PLAN: System North-Star — Canonical Scoreboard and Go/No-Go Policy

**Status:** Control document. This plan does not introduce new modules; it pins the **single scoreboard** and the **decision policy** every other plan must report against.
**Owner:** Pipeline Orchestrator (table assembly + CI emission). Each row's columns are owned by the originating module (§7).
**Companions:** [PLAN-EVAL-FIRST-TARGET.md](PLAN-EVAL-FIRST-TARGET.md) (joint task contract), [PLAN-PIPELINE-ORCHESTRATOR.md](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (eval matrix, gates), [PLAN-HARNESS.md](../05-harness/PLAN-HARNESS.md) (`SkillEpisode`), [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) (Path A/B/C), [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) (promotion / rollback).

---

## 1. Why a north-star scoreboard is necessary

The project has six modules — **Visual Grounding, Action Agent (Actor), Skill Harness, Skill Bank, Skill Crafter, Pipeline Orchestrator** — each with its own metrics, gates, ablation suites, and per-phase milestones. Each module is incentivized to report the numbers it most directly controls: grounding reports schema completeness and Path A rate; the Harness reports binding and veto pass rates; the Crafter reports proposal acceptance rates; the bank reports retrieval hit-at-k; the Actor reports decision quality.

This is necessary but dangerous. Without a single shared scoreboard:

- **Module-level gains masquerade as system-level gains.** A grounding rev that lifts schema completeness from 0.91 to 0.94 looks like progress; if Joint Success Rate ([PLAN-EVAL-FIRST-TARGET.md §5](PLAN-EVAL-FIRST-TARGET.md#5-joint-success-definition-headline)) is flat, no end-user benefit was created.
- **Reward hacking goes undetected.** A Harness change that aggressively vetoes risky skill calls can lift veto precision while quietly tanking task success because too few skills get through.
- **Cost regressions hide behind quality wins.** A new policy that adds two extra grounding calls per instance can buy 1 point of accuracy at 30% more compute and look like a win.
- **Trade-offs become invisible.** Actor quality and Harness quality can move in opposite directions; without a control document, the project optimizes whichever number is in front of the loudest engineer.

The orchestrator's evaluation matrix already requires separate analysis of **actor quality, harness filtering / veto quality, overall system performance, skill-use efficiency, reasoning-step usefulness, and transfer robustness** ([PLAN-PIPELINE-ORCHESTRATOR.md §0a.5](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a5-evaluation-implication)). This document is the **single place** where those axes are projected into one canonical, reproducible table, and where the go/no-go rules over that table are written down — so module work cannot replace end-to-end work.

This plan is binding. Any release report that does not print the §4 table is non-conforming, and any promotion that violates a §5 rule is invalid even if all module-level gates pass.

---

## 2. Metric layers

Every reported number lives in exactly one of three layers. The layer determines how the number is allowed to move the headline.

### 2.1 Layer 1 — End-task metrics (what the system delivers)

These are the only numbers that may justify a release on their own.

| Metric | Definition | Source |
|--------|------------|--------|
| **Answer Accuracy** | `mean(AnswerCorrect)` over evaluation instances ([PLAN-EVAL-FIRST-TARGET.md §7.1](PLAN-EVAL-FIRST-TARGET.md#71-automatic-answer-evaluation)) | eval driver |
| **Evidence Support Rate** | `mean(EvidenceValid)` over evaluation instances ([PLAN-EVAL-FIRST-TARGET.md §7.2](PLAN-EVAL-FIRST-TARGET.md#72-llm-as-judge-for-evidence-support)) | LLM-judge + optional human audit |
| **Joint Success Rate** | `mean(AnswerCorrect AND EvidenceValid)` ([PLAN-EVAL-FIRST-TARGET.md §5](PLAN-EVAL-FIRST-TARGET.md#5-joint-success-definition-headline)) | eval driver (combines the two above) |

### 2.2 Layer 2 — Mechanism metrics (why the system delivers)

These explain *why* Layer 1 moved. They are not substitutes for Layer 1 (§3, §5).

| Metric | Definition | Source |
|--------|------------|--------|
| **Path A acceptance rate** | fraction of grounding steps accepted on direct parse without tool repair / escalation ([PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)) | grounding module |
| **Schema completeness** | fraction of `<state>` records satisfying the canonical schema constraints ([README — Canonical `<state>`](../README.md), [PLAN-VISUAL-GROUNDING.md §12](../01-visual-grounding/PLAN-VISUAL-GROUNDING.md)) | grounding validator |
| **Binding success rate** | `SkillHarness.bind_skill` pass / attempts ([PLAN-HARNESS.md §15](../05-harness/PLAN-HARNESS.md#15-metrics)) | Harness |
| **Evidence pass rate** | Gate G0 pass / `finalize_episode` calls ([PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates), [§5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode)) | Harness |
| **Transfer pass rate** | shadow → active promotion fraction ([PLAN-HARNESS.md §15](../05-harness/PLAN-HARNESS.md#15-metrics)) | Harness / Orchestrator |
| **Promotion precision** | of skills promoted ACTIVE in the window, fraction not subsequently rolled back / deprecated ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) | Orchestrator |
| **Rollback rate** | rollbacks / promotions in the reporting window | Orchestrator |

Mechanism metrics are required on the canonical table (§4) for diagnosis but cannot be cited as system improvement on their own (§5).

### 2.3 Layer 3 — Cost metrics (what the delivery costs)

These keep Layer 1 honest. A Layer 1 win at the price of a Layer 3 regression is *not free* and must be reported jointly.

| Metric | Definition | Source |
|--------|------------|--------|
| **Avg hops** | mean inner-MDP hops per instance ([PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md)) | Actor / orchestrator telemetry |
| **Avg grounding calls** | mean grounding-tool invocations per instance | grounding module |
| **Avg tool calls** | mean tool invocations per instance (incl. video / DOM / desktop tool calls, excluding grounding) | orchestrator telemetry |
| **Latency** | wall-clock seconds per instance, p50 + p95 | orchestrator telemetry |
| **Compute / token / API cost** | summed model tokens × tier price + external API spend per instance | orchestrator cost ledger |

Cost metrics are required columns on the canonical table; they are headline-blocking under the §5.3 rule.

---

## 3. Primary vs secondary metrics

There is exactly one primary metric. There are exactly two secondary metrics. Everything else is mechanism (Layer 2) or cost (Layer 3) and is **explanatory, not headline**.

| Tier | Metric | Status |
|------|--------|--------|
| **Primary** | **Joint Success Rate** | The single headline number. Bolded in every report. Cited first in every release. |
| **Secondary** | Answer Accuracy | Reported alongside Joint Success; never reported without it ([PLAN-EVAL-FIRST-TARGET.md §5](PLAN-EVAL-FIRST-TARGET.md#5-joint-success-definition-headline)). |
| **Secondary** | Evidence Support Rate | Reported alongside Joint Success; never reported without it. |
| **Mechanism** | All Layer 2 metrics | Reported on the canonical table for diagnosis. **Improvement here without Layer 1 movement is not a system improvement** (§5.1). |
| **Cost** | All Layer 3 metrics | Reported on the canonical table. Headline-blocking under §5.3. |

**Why Joint Success and not Answer Accuracy:** Answer Accuracy alone can be inflated by a model that guesses right while citing fabricated or decorative evidence (failure classes `F3`, `F4` in [PLAN-EVAL-FIRST-TARGET.md §6](PLAN-EVAL-FIRST-TARGET.md#6-failure-taxonomy)). Joint Success requires both correctness and verifiable evidence and cannot be gamed by either path in isolation. The whole project — grounding, actor, harness, bank, crafter, orchestrator — exists to push Joint Success up while keeping Layer 3 cost flat or down.

---

## 4. Canonical reporting table

Every release prints **one** table at the top of its report, in this exact column order. The Pipeline Orchestrator is responsible for assembling it from the per-module sources (§7).

### 4.1 The table

| Setting | Answer Acc | Evidence Support | **Joint Success** | Path A | Binding Success | Transfer Pass | Rollback Rate | Avg Tool Calls | Cost ($/inst) | Latency (s/inst, p50/p95) |
|---------|-----------|------------------|-------------------|--------|-----------------|---------------|---------------|----------------|---------------|----------------------------|
| `overall` | | | | | | | | | | |
| `easy` | | | | | | | | | | |
| `medium` | | | | | | | | | | |
| `hard` | | | | | | | | | | |
| `single_hop` | | | | | | | | | | |
| `multi_hop` | | | | | | | | | | |
| `direct_visual` | | | | | | | | | | |
| `temporal` | | | | | | | | | | |
| `social_reasoning` | | | | | | | | | | |
| `cross_domain_transfer` | | | | | | | | | | |

### 4.2 Required companion tables

Directly below the canonical table, in this order, the report prints:

1. **Failure taxonomy distribution** — per [PLAN-EVAL-FIRST-TARGET.md §6](PLAN-EVAL-FIRST-TARGET.md#6-failure-taxonomy), `F1`–`F7` counts and percentages, on the `overall` row at minimum.
2. **Module quality strip** — three numbers, one row each: Actor decision quality (top-1 on Harness-eligible set), Harness filter precision and veto precision/recall ([PLAN-HARNESS.md §20.4](../05-harness/PLAN-HARNESS.md#204-metrics)), schema completeness ([Layer 2 §2.2](#22-layer-2--mechanism-metrics-why-the-system-delivers)). These are the orchestrator's required separated axes ([PLAN-PIPELINE-ORCHESTRATOR.md §0a.5](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#0a5-evaluation-implication)) projected into the scoreboard.
3. **Promotion / rollback ledger** — per the reporting window: promotions count, rollbacks count, promotion precision, top three rollback reasons.
4. **Few-shot transfer table** — per [PLAN-PIPELINE-ORCHESTRATOR.md §6.2a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#62a-few-shot-transfer-target-domain-only). One row per `target_domain ∈ TRANSFER_TARGET_DOMAINS` ([common/enums.py](../03-skill-bank/PLAN-SKILL-BANK.md#04-source-domain--transfer-target-asymmetry)), columns: K-shot pass rate at `K = few_shot.k_shot_default`, transfer skill coverage, multi-target generalization, adaptation cost, target-domain regression rate, source-vs-target gap. This table is the **operational scoreboard for the project's central thesis** (game → other-domain transfer); a release that improves Joint Success on `direct_visual` while regressing every row of this table is **not** a transfer release.

### 4.3 Reporting rules

- The `overall` row is the **headline**. Joint Success is bolded.
- Rows are emitted in the order shown; do not reorder per release.
- Every cell in the table comes from the same `eval_suite_id` and `bank_snapshot_id` ([PLAN-PIPELINE-ORCHESTRATOR.md §3a.2](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a2-batch-evaluation-schedule)) so the table is reproducible.
- Empty cells are reported as `n/a` with a one-line reason in the appendix; they are never silently zero.
- Anything not on the canonical table or in the three companion tables is **appendix-only** and must not appear in the release summary or commit message.
- Two consecutive releases must use the same `eval_suite_id` unless an explicit eval-suite version bump is recorded in the report header — otherwise the deltas are not comparable.

---

## 5. Stop/go decision rules

The rules in this section are **binding on promotions, releases, and "we shipped X" claims**. They take precedence over module-level gate pass rates. Each rule is stated with a concrete trigger and the required action.

### 5.1 Mechanism without end-task — **NO-GO as system improvement**

> If Layer 2 metrics improve but Joint Success Rate does not, do not claim a true system improvement.

Trigger: any Layer 2 metric improves by ≥ its noise band on the `overall` slice while Joint Success Rate is flat (within noise) or down. Action: the change may still merge if it is a refactor, plumbing, or pre-requisite for later work, but the release notes **must not** claim a system improvement, and the change is not eligible for the "improves headline" channel in the changelog. Module owners may still cite the Layer 2 win in their internal docs.

### 5.2 Joint Success up but rollbacks worsen — **NO-GO for promotion**

> If Joint Success improves but rollback / regression rate worsens too much, do not promote.

Trigger: Joint Success Rate up, **and** rollback rate in the reporting window rises above its policy threshold (default: **rollback rate ≥ 2× the trailing 3-release median, or promotion precision < 0.7**). Action: the change does not flow into ACTIVE; it is held in `provisional` ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) until the next eval window. The orchestrator's promotion transaction ([PLAN-PIPELINE-ORCHESTRATOR.md §3a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#3a-promotion-transaction-and-rollback-protocol)) must respect this rule, not just per-skill gate verdicts.

### 5.3 Cost down, evidence collapses — **NO-GO as optimization**

> If cost drops but evidence quality collapses, do not accept as optimization.

Trigger: any Layer 3 metric improves on the `overall` slice while Evidence Support Rate drops by more than its noise band, **or** failure classes `F3` / `F4` ("right answer, broken evidence") rise above their trailing window. Action: the change is rejected as an optimization. It may be re-proposed only with a fix that restores Evidence Support Rate to within the noise band of the prior release. This rule exists because evidence quality is the project's primary defense against silent reward-hacking.

### 5.4 Actor down, Harness up — **MUST report explicitly**

> If actor quality degrades while harness quality rises, report it explicitly.

Trigger: Actor decision quality (top-1 on Harness-eligible set) drops while Harness filter precision or veto precision rises, on the same eval window. Action: **the release report must call this out by name in the summary**, not bury it in the module strip. The architectural risk is that the trainable Actor is being absorbed by the frozen 72B Harness ([PLAN-HARNESS.md §1a.5](../05-harness/PLAN-HARNESS.md#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor)); if this pattern persists across two releases, it triggers a review of the Actor/Harness boundary before any further Harness expansion is merged.

### 5.5 Slice-level regressions — **MUST not be hidden by overall gains**

> If `overall` Joint Success rises but any slice's Joint Success drops by more than its noise band, the slice regression is reported in the summary, not only in the table.

Trigger: `overall` up, any of the §4.1 slice rows down beyond noise. Action: the slice and the magnitude are named in the release summary. This rule is what stops the project from quietly trading `hard` or `cross_domain_transfer` quality for `easy` quality.

### 5.6 Noise bands

Every rule above references a noise band. The noise band per metric per slice is **the trailing 3-release standard deviation on that slice**, recomputed each release. A change is "within noise" if its absolute delta is ≤ 1× that band. Rules 5.2 and 5.3 use 2× thresholds where noted. Noise bands are recorded in the release report header so rule applications are auditable.

---

## 6. Phase-wise metric emphasis

The canonical table (§4) is reported **every release in every phase** — its shape does not change. What changes per phase is **which columns must show measurable progress** to claim phase exit.

| Phase | Primary emphasis (must move) | Secondary emphasis (must not regress) | Phase exit signal |
|-------|-------------------------------|----------------------------------------|-------------------|
| **Phase 0–1 — Grounding + Harness MVP** | Schema completeness; Path A acceptance rate | Layer 3 cost; binding success | Path A and schema completeness reach the targets in [PLAN-VISUAL-GROUNDING-MILESTONES.md](../01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md); every skill invocation flows through the Harness ([PLAN-HARNESS.md §14 Phase 0](../05-harness/PLAN-HARNESS.md#14-phased-implementation-plan)). Joint Success is *measured* but not yet the gating signal. |
| **Phase 2 — Evidence-driven loop** | Evidence Support Rate; Evidence pass rate (Gate G0) | Answer Accuracy; Layer 3 cost | Evidence Support Rate is non-trivial on the `overall` row and `F3` / `F4` are below their initial values. The eval triple ([PLAN-EVAL-FIRST-TARGET.md §5.1](PLAN-EVAL-FIRST-TARGET.md#51-other-required-headline-numbers)) becomes the headline. |
| **Phase 3 — System integration** | **Joint Success Rate** (primary headline) | All Layer 2; all Layer 3 | Joint Success Rate is the gating number. §5.1 starts being enforced strictly: mechanism wins without Joint Success movement do not justify releases. |
| **Phase 4 — Transfer hardening** | Few-shot transfer table (§4.2 #4): per-target K-shot pass rate, transfer skill coverage, multi-target generalization; promotion precision; rollback rate | Joint Success Rate on the source-domain (`gymv`) slice; Layer 3 cost; source-vs-target gap shrinking | The `cross_domain_transfer` row of the canonical table is healthy **and** every row of the few-shot transfer companion table is non-trivial (no target domain stuck at zero coverage); §5.2 is enforced strictly; rollback rate (full-system, not partial-deprecation) trends down release over release. |

Phase exit requires the primary-emphasis columns to move *and* the secondary-emphasis columns to stay within their noise bands. A phase is **not** exited by improving the primary at the cost of the secondary.

---

## 7. Ownership and automation

Each column on the canonical table has exactly one **producing module** and one **consuming surface**. The Pipeline Orchestrator owns assembly; it does not own the underlying numbers.

### 7.1 Column ownership

| Column | Producer | Source artifact |
|--------|----------|-----------------|
| Answer Accuracy | eval driver | `eval/run_<id>/scores.jsonl` ([PLAN-EVAL-FIRST-TARGET.md §3](PLAN-EVAL-FIRST-TARGET.md#3-required-system-outputs)) |
| Evidence Support Rate | LLM judge (+ optional human audit) | `eval/run_<id>/evidence_verdicts.jsonl` ([PLAN-EVAL-FIRST-TARGET.md §7.2](PLAN-EVAL-FIRST-TARGET.md#72-llm-as-judge-for-evidence-support)) |
| **Joint Success Rate** | eval driver | derived from the two above |
| Path A acceptance rate | grounding module | grounding telemetry (`schema_gen` head) |
| Schema completeness | grounding validator | grounding telemetry |
| Binding success rate | Harness | `SkillEpisode.slot_bindings` + `SkillHarness.bind_skill` outcomes ([PLAN-HARNESS.md §5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode)) |
| Evidence pass rate | Harness | Gate G0 outcomes in `SkillHarness.finalize_episode` |
| Transfer pass rate | Harness + Orchestrator | `TransferManager` shadow→active fraction ([PLAN-HARNESS.md §5.4](../05-harness/PLAN-HARNESS.md#54-transfermanager)) |
| Promotion precision | Orchestrator | `PromotionOrchestrator` ledger ([PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)) |
| Rollback rate | Orchestrator | rollback log over reporting window |
| Avg hops | Actor / orchestrator | inner-MDP records ([PLAN-ACTION-AGENT.md](../02-action-agent/PLAN-ACTION-AGENT.md)) |
| Avg grounding calls | grounding module | grounding telemetry |
| Avg tool calls | orchestrator telemetry | per-instance tool ledger |
| Latency (p50 / p95) | orchestrator telemetry | per-instance wall-clock |
| Cost ($/inst) | orchestrator cost ledger | tokens × tier price + API spend |

### 7.2 Companion-table ownership

| Companion table | Producer |
|-----------------|----------|
| Failure taxonomy distribution | eval driver ([PLAN-EVAL-FIRST-TARGET.md §6](PLAN-EVAL-FIRST-TARGET.md#6-failure-taxonomy)) |
| Module quality strip — Actor row | Actor offline eval ([PLAN-HARNESS.md §20.4](../05-harness/PLAN-HARNESS.md#204-metrics)) |
| Module quality strip — Harness row | Harness eval (filter precision, veto precision/recall) |
| Module quality strip — Grounding row | grounding validator (schema completeness) |
| Promotion / rollback ledger | Orchestrator |
| Few-shot transfer table | Orchestrator (assembles from `GateService._run_transfer` Stage 3a verdicts + per-target `verified_domains` log; see [PLAN-PIPELINE-ORCHESTRATOR.md §6.2a](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md#62a-few-shot-transfer-target-domain-only)) |

### 7.3 Automation contract

- **Assembly.** The orchestrator runs the eval driver on the pinned `eval_suite_id`, collects the per-column artifacts above by `bank_snapshot_id`, and writes the canonical table + three companion tables to a single Markdown file `releases/<release_id>/scoreboard.md`.
- **CI emission.** The release CI job fails if any of the following holds:
  - the canonical table is missing a column,
  - any cell is empty (use `n/a` with a reason instead),
  - `eval_suite_id` or `bank_snapshot_id` differs across cells,
  - any §5 rule fires and the release notes do not contain the required disclosure,
  - a §5.2 / §5.3 rule fires and a promotion is still proposed.
- **History.** Every `scoreboard.md` is checked into the repo. The trailing 3-release noise band (§5.6) is computed from these files; missing prior scoreboards mean the noise band is `unknown` and §5 thresholds use the conservative defaults stated in those rules.
- **No silent metric drift.** Adding, removing, or redefining any column on the canonical table requires an explicit version bump of this plan and is announced in the release notes; the same bump must update the eval driver and the orchestrator's assembler.

---

## 8. Anti-goals

- **Do not invent a new headline metric.** Joint Success Rate is the single primary metric (§3). New numbers may live in the appendix; they may not displace, dilute, or be averaged with the headline.
- **Do not let mechanism metrics replace end-task metrics.** A release whose only progress is in Layer 2 is a refactor, not a system improvement (§5.1). Calling it otherwise is non-conforming.
- **Do not collapse the table into a single score.** No "system score = w1·X + w2·Y + …" composite is allowed on the headline. The whole point of separating Layer 1 / Layer 2 / Layer 3 is that they are **not** interchangeable; collapsing them re-creates the ambiguity this plan exists to remove.
- **Do not ship slice regressions silently.** Overall gains do not erase slice losses (§5.5). Any slice regression beyond noise is named in the summary.
- **Do not let the frozen 72B Harness silently absorb the Actor.** §5.4 is enforced; if the Actor degrades while the Harness improves on two consecutive releases, the Actor/Harness boundary is reviewed before further Harness expansion ([PLAN-HARNESS.md §1a.5](../05-harness/PLAN-HARNESS.md#1a5-why-the-frozen-72b-harness-should-not-replace-the-actor)).
- **Do not chase leaderboards.** This scoreboard tracks the project's own delivery against itself across releases. Adding external benchmarks is governed by [PLAN-EVAL-FIRST-TARGET.md §11](PLAN-EVAL-FIRST-TARGET.md#11-non-goals); they do not replace the canonical table.
- **Do not let Layer 3 cost regressions hide behind Layer 1 wins.** §5.3 is enforced; cost columns are required and headline-blocking when evidence collapses.
- **Do not redefine the gate semantics, lifecycle, or `SkillEpisode` schema in this plan.** Those live in [PLAN-UNIFIED-SKILL-GATE.md](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md), [PLAN-HARNESS.md §10](../05-harness/PLAN-HARNESS.md#10-promotion-gates), and [PLAN-HARNESS.md §5.1](../05-harness/PLAN-HARNESS.md#51-skillepisode). This plan *consumes* them; if it needs to change them, change the upstream plan first.
