"""
Bank Maintenance — Skill Bank Update: Split / Merge / Refine + fast re-decode.

Maintains a high-quality Skill Bank by applying three update operations:

  **SPLIT**   one skill contains multiple modes → split into child skills
  **MERGE**   two skills are near-duplicates → merge into one canonical skill
  **REFINE**  contracts are too strong / weak → drop fragile or add discriminative literals

Also updates:
  - Duration model p(ℓ|k)
  - Indices (effect inverted index, MinHash/LSH, optional ANN)

Entry point: :func:`run_bank_maintenance.run_bank_maintenance`.

All public symbols are importable from the sub-modules directly::

    from skill_agents_grpo.bank_maintenance.config import BankMaintenanceConfig
    from skill_agents_grpo.bank_maintenance.run_bank_maintenance import run_bank_maintenance
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with skill_bank ↔ stage3_mvp."""
    _lazy_map = {
        "BankMaintenanceConfig": "skill_agents_grpo.bank_maintenance.config",
        "run_bank_maintenance": "skill_agents_grpo.bank_maintenance.run_bank_maintenance",
        "BankMaintenanceResult": "skill_agents_grpo.bank_maintenance.run_bank_maintenance",
        "BankDiffReport": "skill_agents_grpo.bank_maintenance.schemas",
        "RedecodeRequest": "skill_agents_grpo.bank_maintenance.schemas",
        "SkillProfile": "skill_agents_grpo.bank_maintenance.schemas",
    }
    if name in _lazy_map:
        import importlib
        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BankMaintenanceConfig",
    "run_bank_maintenance",
    "BankMaintenanceResult",
    "BankDiffReport",
    "RedecodeRequest",
    "SkillProfile",
]
