"""Question-class router for benchmark prompts.

The visual-reasoning benchmarks ship a single free-text question per
sample.  Different questions need different reasoning *primitives* —
counting questions need ``count_value``, ratio / proportion questions
need ``compute_ratio``, comparative questions need ``compare_values``,
narrative video questions need ``track_object`` /
``describe_frame``.  When the goal text only mentions detection-style
tools, the model defaults to ``detect_objects`` even on a "what
proportion" question, then narrates the ratio without computing it.

This module classifies a question into one or more *classes* and
returns a small block of text the parser injects into the
``goal=`` argument of ``GroundingRequest``.  The injection lists the
tools the question class wants to see called and the schema sections
they should populate.

Designed to be lightweight (regex / keyword based) so it runs in
microseconds — we lean on the VLM to do the actual reasoning, this
router just nudges the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Question-class enumeration — kept small on purpose.  The router
# returns a *set* of these classes (questions can be multi-class).
QUESTION_CLASSES: tuple[str, ...] = (
    "count",        # "how many", "count of …", "number of …"
    "ratio",        # "what proportion", "what fraction", "percent of …"
    "compare",      # "which is bigger", "is X more than Y", "closer to"
    "spatial",      # "left of", "above", "between", "next to"
    "ocr",          # "what does the sign say", "the text reads"
    "identity",     # "who is …", "name the person", "what character"
    "temporal",     # "when does", "before / after", "first / last"
    "social",       # "what relationship", "why are they", "intent / motive"
    "verify",       # "is the claim true", "does X hold"
    "answer",       # always-on terminal class — every question commits an answer
)

# Per-class signals (ordered by specificity).  Patterns are anchored
# loosely so a question phrased as "How many people are in the
# scene?" matches "count" without false-firing on "How many of the
# objects are red?" (which is also "count" + "ratio").
_CLASS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "count": [
        re.compile(r"\bhow many\b", re.IGNORECASE),
        re.compile(r"\bnumber of\b", re.IGNORECASE),
        re.compile(r"\bcount\b", re.IGNORECASE),
        re.compile(r"\btotal of\b", re.IGNORECASE),
    ],
    "ratio": [
        re.compile(r"\bwhat (?:proportion|fraction|percent(?:age)?)\b", re.IGNORECASE),
        re.compile(r"\bproportion of\b", re.IGNORECASE),
        re.compile(r"\bratio of\b", re.IGNORECASE),
        re.compile(r"\bpercent(?:age)? of\b", re.IGNORECASE),
        re.compile(r"\bfraction of\b", re.IGNORECASE),
    ],
    "compare": [
        re.compile(r"\b(?:bigger|larger|smaller|greater|less|fewer|more)\b",
                   re.IGNORECASE),
        re.compile(r"\bwhich (?:is|one|of)\b", re.IGNORECASE),
        re.compile(r"\bcloser to\b", re.IGNORECASE),
        re.compile(r"\bfarther (?:from|than)\b", re.IGNORECASE),
        re.compile(r"\b(?:tallest|shortest|widest|narrowest|highest|lowest)\b",
                   re.IGNORECASE),
    ],
    "spatial": [
        re.compile(r"\b(?:left|right|above|below|under|over|inside|outside|between|"
                   r"next to|adjacent|behind|in front)\b", re.IGNORECASE),
        re.compile(r"\bdistance (?:from|between|to)\b", re.IGNORECASE),
        re.compile(r"\bwhere (?:is|are)\b", re.IGNORECASE),
    ],
    "ocr": [
        re.compile(r"\bsign (?:say|read)s?\b", re.IGNORECASE),
        re.compile(r"\b(?:label|caption|text|writing) (?:say|read|show)s?\b",
                   re.IGNORECASE),
        re.compile(r"\bwhat does .*? (?:say|read)\b", re.IGNORECASE),
        re.compile(r"\bwhat is written\b", re.IGNORECASE),
    ],
    "identity": [
        re.compile(r"\b(?:who is|who are|name (?:the|this|that))\b",
                   re.IGNORECASE),
        re.compile(r"\bwhich character\b", re.IGNORECASE),
    ],
    "temporal": [
        re.compile(r"\b(?:before|after) (?:the|this)\b", re.IGNORECASE),
        re.compile(r"\bwhen (?:does|did|will)\b", re.IGNORECASE),
        re.compile(r"\b(?:first|last) (?:scene|frame|moment|to)\b",
                   re.IGNORECASE),
        re.compile(r"\bsequence\b", re.IGNORECASE),
    ],
    "social": [
        re.compile(r"\b(?:relationship|intent|motive|why are they|why is "
                   r"he|why is she|emotion|feel(?:ing)?|attitude)\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:family|friends|colleagues|strangers)\b",
                   re.IGNORECASE),
    ],
    "verify": [
        re.compile(r"\bis (?:it|the) (?:true|correct|right)\b", re.IGNORECASE),
        re.compile(r"\b(?:true or false|t/f)\b", re.IGNORECASE),
        re.compile(r"\bdoes .*? hold\b", re.IGNORECASE),
    ],
}


@dataclass
class RoutingDecision:
    """Output of :func:`classify_question`."""

    classes: list[str]
    required_tools: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    derivation_kinds: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render the decision as a bullet-list prompt fragment.

        The output is meant to be appended inside the ``goal=`` of a
        ``GroundingRequest`` so the VLM sees it before tool invocation.
        Returns an empty string when no classes fired (common for
        plain "what is in the image" questions).
        """
        if not self.classes:
            return ""
        lines = [
            (
                "Question-class router (auto-detected): "
                f"{', '.join(self.classes)}."
            ),
        ]
        if self.required_tools:
            lines.append(
                "Required tools — call AT LEAST ONE before <answer>: "
                + ", ".join(self.required_tools) + "."
            )
        if self.suggested_tools:
            lines.append(
                "Suggested observation tools to ground inputs: "
                + ", ".join(self.suggested_tools) + "."
            )
        if self.derivation_kinds:
            lines.append(
                "Each <derivations> row must use one of "
                f"kind∈{{{', '.join(self.derivation_kinds)}}}."
            )
        for inst in self.instructions:
            lines.append(inst)
        return "\n".join(lines)


def classify_question(
    question: str,
    *,
    modality: str = "image",
) -> RoutingDecision:
    """Return the question's classes + the tools the prompt should require.

    Parameters
    ----------
    question : str
        The raw user prompt for one benchmark sample.
    modality : str
        ``"image"`` or ``"video"`` — used to pick the right
        observation-tool set (e.g. ``detect_objects_at_frame`` vs
        ``detect_objects``).

    The decision is non-exclusive: a question can be tagged "count +
    ratio" (e.g. "what fraction of the people are wearing hats"),
    which forces both ``count_value`` and ``compute_ratio``.
    """
    text = (question or "").strip()
    if not text:
        return RoutingDecision(classes=[])

    classes: list[str] = []
    for cls in QUESTION_CLASSES:
        patterns = _CLASS_PATTERNS.get(cls)
        if not patterns:
            continue
        if any(p.search(text) for p in patterns):
            classes.append(cls)

    # Every benchmark sample produces an answer, so verify_claim is
    # always promoted as the final reasoning step.
    classes.append("answer")

    return _build_decision(classes=classes, modality=modality)


def _build_decision(
    *, classes: list[str], modality: str,
) -> RoutingDecision:
    is_video = modality == "video"
    detect_tool = "detect_objects_at_frame" if is_video else "detect_objects"
    grounded_tool = "detect_objects_at_frame" if is_video else "grounded_detect"
    describe_tool = "describe_frame" if is_video else "describe_region"

    required: list[str] = []
    suggested: list[str] = []
    derivations: list[str] = []
    instructions: list[str] = []

    if "count" in classes:
        required.append("count_value")
        suggested += [grounded_tool, detect_tool]
        derivations.append("COUNT")
        instructions.append(
            "For the count: call a detection tool first, then "
            "`count_value(value=N, label='…', refs='e1,e2,…')`."
        )
    if "ratio" in classes:
        required.append("compute_ratio")
        suggested += [grounded_tool, detect_tool]
        derivations.append("RATIO")
        instructions.append(
            "For the proportion / percentage: ground numerator and "
            "denominator with detection tools first, then call "
            "`compute_ratio(numerator=…, denominator=…, label='…', "
            "refs='d1,d2')`."
        )
    if "compare" in classes:
        required.append("compare_values")
        suggested += ["measure_distance", "spatial_query", grounded_tool]
        derivations.append("COMPARE")
        instructions.append(
            "For the comparison: measure both quantities (sizes / "
            "distances / counts) with the appropriate tool, then call "
            "`compare_values(a=…, b=…, op='>', label_a='…', "
            "label_b='…')`."
        )
    if "spatial" in classes:
        suggested += ["spatial_query", "measure_distance", detect_tool]
    if "ocr" in classes:
        suggested.append("read_text_in_frame" if is_video else "read_text_region")
    if "identity" in classes:
        suggested.append(describe_tool)
    if "temporal" in classes:
        suggested += ["find_moment", "detect_scene_changes", "sample_frames"]
    if "social" in classes:
        suggested += ["describe_frame", "track_object"]

    # answer / verify_claim closes every chain.
    required.append("verify_claim")
    derivations.append("VERIFY")
    instructions.append(
        "Final reasoning step before <answer>: call "
        "`verify_claim(claim='<your answer string>', "
        "evidence_refs='e?,d?,hop?')` so the answer is bound to its "
        "evidence ids."
    )

    if "verify" in classes and "verify_claim" not in required:
        required.append("verify_claim")

    seen_classes: list[str] = []
    for c in classes:
        if c not in seen_classes:
            seen_classes.append(c)

    def _dedup(seq: list[str]) -> list[str]:
        out: list[str] = []
        for s in seq:
            if s not in out:
                out.append(s)
        return out

    return RoutingDecision(
        classes=seen_classes,
        required_tools=_dedup(required),
        suggested_tools=_dedup(suggested),
        derivation_kinds=_dedup(derivations),
        instructions=instructions,
    )


__all__ = [
    "QUESTION_CLASSES",
    "RoutingDecision",
    "classify_question",
]
