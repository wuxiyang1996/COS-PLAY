"""Few-shot example library for visual-schema generation.

Provides one curated GPT-5.4 gold ``<state>...</state>`` example per supported
domain.  Used by ``build_adaptive_system_prompt(few_shot_examples=...)`` to
inject worked examples that anchor naming conventions, section ordering, and
phrasing for any base model — closes the gap between base-Qwen3.5-35B and the
schema_gen-LoRA-tuned variant without per-domain SFT.

Examples were selected from the ``Cold-start-out-{,-gymv,-browsergym,-osworld,
-visual-reasoning,-visual-reasoning-video}`` dumps and ``labeling/output/
grounding/`` triples by:

  1. Filtering to schemas that close ``</state>``.
  2. Requiring all 6 canonical sections (entities, relations, state_flags,
     targets, uncertainty, actions).
  3. Targeting ~2 000-character schemas (long enough to be representative,
     short enough not to dominate the prompt).
  4. Skipping any sample whose ``<state>`` header has the wrong ``domain=``
     tag (the cold-start labeler mistagged ~1.7-29 % of OSWorld and
     image_qa/video_qa frames as ``domain=browser``).

The on-disk source files live under ``vlm_wrapper/few_shot_examples/<domain>.txt``
so curators can edit them without touching this module.

Domain key conventions (mirror the schema_gen data_loader):

  - ``"gymv"``         — Temporal_*-v0 (Gym-Retro)
  - ``"env_wrappers"`` — env_wrappers buckets (super_mario, candy_crush,
                          tetris, twenty_forty_eight); shares the gymv
                          schema spec
  - ``"browser"``      — BrowserGym (assistantbench, miniwob.*)
  - ``"desktop"``      — OSWorld (chrome, gimp, libreoffice_*, vlc, vs_code,
                          thunderbird, multi_apps, os)
  - ``"image_qa"``     — visual_toolbench, tir_bench
  - ``"video_qa"``     — siv_bench, video_holmes

Usage::

    from vlm_wrapper.few_shot_library import get_few_shot_examples
    from vlm_wrapper.schema import build_adaptive_system_prompt

    examples = get_few_shot_examples("browser", n=1)
    prompt = build_adaptive_system_prompt(
        domain="browser", few_shot_examples=examples,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR / "few_shot_examples"

# Cached examples — read once, re-served on every call.
_CACHE: dict[str, List[str]] = {}


def _load_key(key: str) -> List[str]:
    """Load all example files for a key (cached).

    File naming convention: ``<key>.txt`` for the primary example,
    ``<key>.2.txt``, ``<key>.3.txt`` for additional examples.  ``<key>``
    can be either a domain (e.g. ``"browser"``) or a domain.task pair
    (e.g. ``"env_wrappers.candy_crush"``).  Missing files are silently
    ignored.
    """
    if key in _CACHE:
        return _CACHE[key]
    out: List[str] = []
    primary = _EXAMPLES_DIR / f"{key}.txt"
    if primary.is_file():
        out.append(primary.read_text().strip())
    n = 2
    while True:
        extra = _EXAMPLES_DIR / f"{key}.{n}.txt"
        if not extra.is_file():
            break
        out.append(extra.read_text().strip())
        n += 1
    _CACHE[key] = out
    return out


def _normalize_task_slug(task_id: Optional[str]) -> Optional[str]:
    """Reduce a free-form task_id (e.g. ``make_gaming_env/candy_crush``,
    ``browsergym/miniwob.book-flight``) to a single short slug we can look
    up on disk: the last path segment, stripped of dotted suffixes.

    Examples::

        make_gaming_env/candy_crush      -> candy_crush
        env_wrappers/super_mario         -> super_mario
        browsergym/miniwob.book-flight   -> miniwob
        59f21cfb-...-uuid                -> 59f21cfb (rarely useful)
        Temporal/ThunderForceIII-v0      -> ThunderForceIII-v0

    Returns ``None`` if *task_id* is empty or yields an unsuitable slug.
    """
    if not task_id:
        return None
    slug = task_id.strip().split("/")[-1]
    # For env_wrappers / gymv we only have plain task names; for browser
    # we keep the task family up to the first dot.
    slug = slug.split(".", 1)[0]
    slug = slug.strip()
    return slug or None


def get_few_shot_examples(
    domain: str,
    n: int = 1,
    *,
    task_id: Optional[str] = None,
    fallback_domain: Optional[str] = None,
) -> List[str]:
    """Return up to *n* curated ``<state>...</state>`` examples.

    Resolution order (first non-empty wins):

      1. ``{domain}.{task_slug}`` — task-specific (most precise; avoids
         vocabulary leakage from sibling tasks).
      2. ``{domain}`` — domain-default (used when no task-specific
         example is on disk).
      3. ``{fallback_domain}`` — last-resort fallback (e.g. ``gymv``
         when only an ``env_wrappers`` lookup was attempted).

    Parameters
    ----------
    domain : str
        One of ``"gymv"``, ``"env_wrappers"``, ``"browser"``,
        ``"desktop"``, ``"image_qa"``, ``"video_qa"``.
    n : int
        Maximum number of examples to return.  ``n=0`` returns ``[]``
        (zero-shot).
    task_id : str, optional
        The specific task_id (e.g. ``"make_gaming_env/candy_crush"``).
        When provided, the loader first looks for
        ``{domain}.{task_slug}.txt`` before falling back to the domain
        default.  Strongly recommended for env_wrappers/gymv to avoid
        cross-task vocabulary bleed (e.g. 2048's ``tile_X_N`` labels
        leaking into a candy_crush prompt).
    fallback_domain : str, optional
        Last-resort fallback domain.

    Returns
    -------
    list[str]
        Up to *n* example schemas.
    """
    if n <= 0:
        return []
    slug = _normalize_task_slug(task_id)
    examples: List[str] = []
    if slug:
        examples = _load_key(f"{domain}.{slug}")
    if not examples:
        examples = _load_key(domain)
    if not examples and fallback_domain:
        examples = _load_key(fallback_domain)
    return examples[:n]


def list_supported_domains() -> List[str]:
    """Return all domains for which we have at least one curated example."""
    if not _EXAMPLES_DIR.is_dir():
        return []
    seen: set = set()
    for p in sorted(_EXAMPLES_DIR.glob("*.txt")):
        # split off the optional ".N" suffix for multi-shot files
        name = p.stem.split(".")[0]
        seen.add(name)
    return sorted(seen)


def render_examples_block(examples: Iterable[str], domain: str) -> str:
    """Format one or more examples into a system-prompt-ready block.

    The exact wording matters: the model needs to be told (a) these are the
    canonical format, (b) it must mimic the *naming/ordering*, not the
    *content* (entities/coordinates differ per frame).
    """
    examples = [e.strip() for e in examples if e and e.strip()]
    if not examples:
        return ""
    parts = [
        f"## Worked examples for {domain}",
        "",
        "Below are canonical reference outputs produced by the GPT-5.4 labeler",
        "for representative frames in this domain.  Mimic the **naming**,",
        "**section ordering**, **task=** convention, **goal=** phrasing, and",
        "**ontology vocabulary** shown here.  Do NOT copy the entities,",
        "coordinates, or values verbatim — those differ per frame.  Use these",
        "to anchor the *style*; ground the *content* in the screenshot you",
        "are about to be shown.",
        "",
    ]
    for i, ex in enumerate(examples, start=1):
        parts.append(f"### Example {i}")
        parts.append("")
        parts.append(ex)
        parts.append("")
    return "\n".join(parts)
