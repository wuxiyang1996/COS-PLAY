"""Cross-corpus skill bank extraction for the transfer-test measurement layer.

Local extractor that ingests GPT-5.4 cold-start rollouts from the four
target corpora (browsergym + osworld + 4 visual benchmarks) and emits
canonical `{report, skill}`-shape SkillRecords matching the
`labeling/skill_bank_out/` schema.

Two lift architectures live here:

- :mod:`sequence_lift` — for browser/osworld; multi-step rollouts with
  an ``experiences[]`` spine. Drives
  :class:`skill_agents.pipeline.SkillBankAgent` end-to-end (segment +
  effects-contract + cluster + materialize), then runs
  :func:`labeling._protocol_lift.lift_protocol_to_typed_hops` on each
  materialised skill.

- :mod:`single_shot_lift` — for VTB / TIR-Bench / Video-Holmes /
  SIV-Bench; one rollout per sample, with `schema + answer_reasoning +
  answer + gold_answer + correct + judge` as the entire trace. Parses
  the reasoning chain for ``e\\d+`` schema-entity references and lifts
  to a typed `(GROUND -> CHECK/RETRIEVE -> VERIFY -> COMMIT)` protocol.

See :doc:`/implementation_notes/cross-domain-transfer-suite-rollout`
§5.5 for the full design.
"""

from __future__ import annotations
