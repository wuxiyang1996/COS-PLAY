# plans/legacy — finished plan corpus

This folder holds plan material that is **already delivered or superseded**:

- Applied refactor **edit plans** whose content was folded into the live
  `PLAN-*.md` files under `plans/00-system/` … `plans/09-implementation/`.
- **Archived discussions** kept only for provenance.

Active engineering reads the numbered stage / component folders and
[`plans/README.md`](../README.md); this directory is an audit trail.

## Contents

| Path | Status | Notes |
|------|--------|------|
| [`10-edits/README.md`](10-edits/README.md) | ✅ **DONE** | Harness control-plane reconciliation, transferable IR / HopTrace edit dispatch, lightweight visual grounding — all applied upstream. |
| [`10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md`](10-edits/PLAN-EDITS-HARNESS-CONTROL-PLANE.md) | ✅ **DONE** | Same — revision notes folded into Harness / Orchestrator / Bank plans. |
| [`10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md`](10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) | ✅ **DONE** | Same — HopTrace / Skill IR phases folded into Skill Bank / Action Agent / Crafter / Gate plans (historical two-level MDP wording may remain in places; live posture is single-MDP + lane (a) — see [`implementation_notes/legacy/skill-lane-decision.md`](../../implementation_notes/legacy/skill-lane-decision.md)). |
| [`10-edits/PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md`](10-edits/PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md) | ✅ **DONE** | Same — grounding-as-perception-layer folded into Visual Grounding milestone docs. |
| [`99-archive/README.md`](99-archive/README.md) | ✅ **DONE (superseded)** | Index for archived discussion. |
| [`99-archive/DISCUSSION-MCP-VS-HARNESS.md`](99-archive/DISCUSSION-MCP-VS-HARNESS.md) | ✅ **DONE (superseded)** | Historical MCP vs harness terminology note. |

## Reading rule

> **Where `legacy/` disagrees with the live `plans/0X-…/PLAN-*.md` files or
> with [`implementation_notes/pre-training-readiness-audit.md`](../../implementation_notes/pre-training-readiness-audit.md),
> the live plan + audit ledger wins.**
