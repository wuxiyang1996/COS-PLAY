# §6 — Limitations

> **Reviewer ask**: "no honest limitations section."

The current submission's limitations cluster into five buckets, in
descending order of how much they bound the claims.

## L1. Scale of the Bank Agent vs. RAG memory baselines

We compare against retrieval-augmented LLM agents that use
unstructured episode memory.  We do **not** scale the LLM-as-judge or
the `harness.validator` LoRA to the same training compute as the
actor; for a fixed-budget comparison §A in the appendix re-ranks the
methods at matched FLOPs.

## L2. Held-out generalisation is *closed* world

Cross-domain transfer is reported within the curated eval suite
(BrowserGym held-out splits, OSWorld test-small, the four
visual-reasoning benchmarks, and a held-out GymV subset).  We do not
exercise zero-shot transfer to fundamentally novel modalities (e.g.
audio, embodied physical control); §5.5's claim is best read as
"transfer across visually-grounded text-and-action domains".

## L3. Promotion-gate dependence on a frozen 35B judge

The lifecycle gating in §3.3 currently hard-codes a 35B reasoning
model as the promotion judge.  Ablation §5.3.3 (`--promotion-bypass-mode
permissive`) shows what happens with the gate stripped, but does not
quantify how much of the gain is judge-quality-dependent.  Replacing
the judge with a smaller open-weights reasoner is left to future work.

## L4. Skill-bank churn under long horizons

Lifetime-distribution analysis (§5.2.4) is right-censored at
~`{steps_at_eval}` outer steps; we cannot yet say how the long-tail
behaves on multi-week training.  We empirically observe a
super-linear growth in `n_active` between phases 4–6, but the bank
deduplication policy is heuristic.

## L5. Single backbone

All numbers use a single Qwen3.5-9B vision-language backbone.  We do
not yet show that the harness/crafter/promotion architecture is
backbone-agnostic.  Reproducing on Llama-3-Vision and Pixtral is
listed in the appendix's "future work" panel.
