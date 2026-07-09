# implementation_notes — design memos & sprint ledger

This directory holds **implementation-facing narrative**: decisions that shipped,
cross-domain rollout plans, and the rolling **pre-training readiness audit**.

## Active *(keep here — not legacy)*

| Doc | Status | Purpose |
|-----|--------|---------|
| [`pre-training-readiness-audit.md`](pre-training-readiness-audit.md) | 🟢 **ACTIVE** | Sprint ledger (S0–S4); open vs closed items inline (strikethrough + shipped annotations). Source of truth for “what blocks training **today**”. |
| [`cross-domain-transfer-suite-rollout.md`](cross-domain-transfer-suite-rollout.md) | 🟡 **PARTIAL** | Phase 1.5 + Phase-5/6 measurement **infra shipped**; **real-env binding** for browser / OS / video executors + Phase-1.5b TODOs still open (see doc banner). |

## Legacy *(✅ DONE — moved under [`legacy/`](legacy/README.md))*

Finished decision memos and measurement plans live under **`implementation_notes/legacy/`**:

- Single-MDP trade-off, lane-(a) skill decision, vLLM topology (T2.8), protocol lift shipped, Crafter/Harness/Orchestrator roles memo, intra-gymv harness usability memo, Phase-5/6 cross-domain **measurement** memo.

See **[`legacy/README.md`](legacy/README.md)** for the full table and file list.

## Cross-links

| Doc | Role |
|-----|------|
| [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) | Repo-wide shipped vs open inventory. |
| [`plans/README.md`](../plans/README.md) | Canonical plan corpus index (`legacy/` mirrors finished plan edits). |
