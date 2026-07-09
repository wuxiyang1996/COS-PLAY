# MCP vs harness — plans directory check

**Question:** Does this `plans/` directory explicitly mention **MCP** or a **harness**, and are those ideas already present under different names?

This repo is closer than the Video_Skills one. It still does not explicitly say “MCP” or “harness,” but it has a much clearer harness-shaped architecture.

## What is already there

- The plans define a shared end-to-end pipeline: **Visual Grounding → Action Agent → Skill Bank → Skill Crafter**, with feedback loops between them. That is basically a **system runtime layout**, which is a core part of a harness.

- The Action Agent plan explicitly includes a **decision loop** and a **two-level MDP** with inner actions like **GROUND | CHECK | RETRIEVE | COMMIT | EXECUTE**. That is very harness-like because it specifies how an episode is stepped and controlled.

- The Skill Bank plan includes a **query/select API** with cross-domain retrieval, plus **shared primitives**, **adapters**, and a **unified structured state interface**. That is the closest thing here to an MCP-like interface layer, though it is still repo-specific rather than a standardized protocol.

- The repo also defines a **canonical `<state>` schema** shared across components, along with shared slot names like `target`, `blocker`, `constraint`, and `history_anchor`. That is exactly the kind of common representation a harness usually depends on.

- Most importantly, it has a **three-agent role split** with distinct roles, model assignments, and update timescales:
  - **Actor / Decision Agent**
  - **Skill-Use / Operational Agent**
  - **Synthesis-Reflection Agent**

  This is stronger than just a loose idea; it is already a runtime decomposition.

## Verdict

| | |
|--|--|
| **MCP** | Still **no explicit MCP layer** |
| **Harness** | **Yes** — implicitly and fairly strongly |

## How “harness” maps in this repo

In this repo, “harness” would map to:

- the shared pipeline
- the decision loop
- the two-level MDP runner
- the tool/skill retrieval interfaces
- the canonical state schema
- the agent-role orchestration and update schedule

## One-line summary

This repo already has a **research harness design**, but not yet a clean **software harness abstraction** and not an **MCP protocol layer**.

---

*Optional follow-up:* turn this into a Cursor-ready refactor plan with explicit folders like `harness/`, `protocols/`, `tools/`, and `agents/`.
