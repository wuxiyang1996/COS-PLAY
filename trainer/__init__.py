"""
Training infrastructure for co-evolving Decision Agent (GRPO) and SkillBank Agent (Hard-EM).

Subpackages:
    common   — shared configs, metrics, evaluation harness, logging, seeds
    decision — VLM decision agent GRPO trainer (env wrapper, rollout collector, replay buffer)
    skillbank — SkillBank Hard-EM trainer (ingest, decode, contract, update, gating)
"""
