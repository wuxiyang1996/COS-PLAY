"""Phase-0 grounding-label collection (PLAN-VISUAL-GROUNDING-MILESTONES §5).

Generates ``(frame, heuristic_schema, gpt4o_schema)`` triples for the
two interactive domains where we have a reliable text-only heuristic
parser:

* ``collect_gymv.py``    — Gym-V environments
* ``collect_browser.py`` — BrowserGym environments
* ``cross_validate.py``  — score the heuristic vs the vision-LLM teacher
  (``gpt-5.5`` by default; ``VLM_LABEL_MODEL`` to override)
                            on the collected triples

Each script writes JSONL to a stable layout::

    labeling/output/grounding/<domain>/<env_id>/triples.jsonl
    labeling/output/grounding/<domain>/<env_id>/frames/step_NNN.png

Triples files feed the Phase-1 ``schema_gen`` SFT pipeline directly
(see ``trainer/SFT/schema_gen/data_loader.py``).
"""
