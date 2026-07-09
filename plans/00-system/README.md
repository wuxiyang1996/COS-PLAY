# 00-system — System-level control documents

Top-of-stack documents that govern how the rest of the corpus is read and how releases are judged. They do **not** implement modules — they pin invariants, scoreboards, and ownership.

## Status (repo snapshot — 2026-05-02)

**Shipped:** E0-style eval driver + canonical scoreboard assembler + eval suite loader (`evaluation/{driver,scoreboard,answer_evaluator}.py`, `orchestrator/eval_suite.py`) — see [`IMPLEMENTATION-STATUS.md`](../../IMPLEMENTATION-STATUS.md).  
**Open:** NORTHSTAR §5 stop/go rules firing automatically off **live** GRPO scoreboards (operational once fast-loop runs); slice/regression policies remain governance-heavy — track [`implementation_notes/pre-training-readiness-audit.md`](../../implementation_notes/pre-training-readiness-audit.md) §2–§8.

| Document | Purpose |
|----------|---------|
| [`PLAN-SYSTEM-NORTHSTAR.md`](PLAN-SYSTEM-NORTHSTAR.md) | The single canonical scoreboard (Layer 1 end-task / Layer 2 mechanism / Layer 3 cost), the headline metric (Joint Success Rate), and the binding go/no-go decision rules every other plan must report against. |
| [`PLAN-EVAL-FIRST-TARGET.md`](PLAN-EVAL-FIRST-TARGET.md) | First end-to-end evaluation contract for the project: short-video evidence-grounded reasoning. Defines task contract, six evaluation axes, failure taxonomy `F1`–`F7`, judge protocol, slice plan, phase rollout (E0 → E2). |
| [`DISCUSSION-COMPONENT-RESPONSIBILITIES.md`](DISCUSSION-COMPONENT-RESPONSIBILITIES.md) | Role walkthrough across Skill Bank Agent / Harness / Orchestrator on four short-video scenarios. Onboarding aid; live plan files win on conflict. |

Read these first if you are new to the repo, then jump to the stage / component plan you need.

Back to [plans/README.md](../README.md).
