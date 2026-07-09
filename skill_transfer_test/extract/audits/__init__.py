"""Stage-0 pre-flight static audits for cross-domain transfer.

See ``implementation_notes/legacy/phase5-cross-domain-measurement.md`` Section 3
for the design.

Each audit is invokable as a module:

    python -m skill_transfer_test.extract.audits.vocab_jaccard
    python -m skill_transfer_test.extract.audits.predicate_firing_static
    python -m skill_transfer_test.extract.audits.slot_binding_feasibility

Or all three at once via the combined runner:

    python -m skill_transfer_test.extract.audits._runner

Outputs land under ``cross_domain_results/_phase0/<run_id>/`` (gitignored).
"""
