# `vlm_wrapper.tests`

Pytest package for the **cross-domain** parts of the stack: schema
parsers, validators, and the `cascaded_ground` end-to-end smoke that
exercises multi-hop tool calling. Per-domain GPT-4o adapter tests now
live next to each wrapper:

| Wrapper                                | Test file                                                             |
|----------------------------------------|------------------------------------------------------------------------|
| `gymv_wrapper`                         | `gymv_wrapper/tests/test_gpt4o_parsers.py`                            |
| `browsergym_wrapper`                   | `browsergym_wrapper/tests/test_gpt4o_parsers.py`                      |
| `osworld_wrapper`                      | `osworld_wrapper/tests/test_gpt4o_parsers.py`                         |
| `visual_reasoning_wrapper` (TIR-Bench, Video-Holmes, synthesizers) | `visual_reasoning_wrapper/tests/test_gpt4o_parsers.py` |

## Running tests

From the repo root, with the `vlm_benchmarks` conda env (or equivalent) activated:

```bash
# Cross-domain core (this folder)
pytest vlm_wrapper/tests -q

# All domain-specific contract tests + the cross-domain core
pytest gymv_wrapper/tests browsergym_wrapper/tests osworld_wrapper/tests \
       visual_reasoning_wrapper/tests vlm_wrapper/tests -q
```

Live API tests (require keys and may bill):

```bash
# Cross-domain (cascaded_ground tool loop)
pytest vlm_wrapper/tests/test_gpt4o_parsers.py -m live -q

# All live tests across every wrapper
pytest gymv_wrapper/tests browsergym_wrapper/tests osworld_wrapper/tests \
       visual_reasoning_wrapper/tests vlm_wrapper/tests -m live -q
```

Each per-wrapper `tests/` folder ships its own `conftest.py` that
loads the project `.env` file so live tests pick up
`OPENAI_API_KEY` / `VLM_TEST_API_KEY` even when `pytest` is launched
from a plain shell. Parent documentation: [`../README.md`](../README.md).
