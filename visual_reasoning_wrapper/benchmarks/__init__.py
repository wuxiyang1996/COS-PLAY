"""Benchmark loaders + VLM parsers for the image/video QA domains.

Each module exposes:

  1. A streaming ``iter_*_samples`` helper.
  2. A ``parse_*_sample`` entry point that runs ``cascaded_ground`` and
     returns a uniform dict (``schema``, ``answer``, ``tool_trace``, …).

Image benchmarks (HF ``datasets`` + cache):

* ``visual_toolbench`` — VisualToolBench (tool-enabled perception /
  transformation / reasoning; arXiv:2510.12712).
* ``tir_bench``       — TIR-Bench (agentic thinking-with-images;
  arXiv:2511.01833).

Video benchmarks (local data under ``data/``):

* ``video_holmes`` — Video-Holmes multi-hop video QA.
* ``siv_bench``    — SIV-Bench social-interaction video QA.

Image loaders need ``Pillow`` and ``datasets`` (plus HF cache or network).
Video loaders need ``decord`` or ``opencv-python`` (SIV-Bench reuses
Video-Holmes' frame sampler).
"""

from .siv_bench import (
    SIVBenchSample,
    default_siv_bench_root,
    iter_siv_bench_samples,
    load_siv_bench_questions,
    parse_siv_bench_sample,
)
from .tir_bench import (
    TIRBenchSample,
    default_tir_bench_root,
    iter_tir_bench_samples,
    load_tir_bench_image,
    parse_tir_bench_batch,
    parse_tir_bench_sample,
)
from .video_holmes import (
    VideoHolmesSample,
    default_video_holmes_root,
    iter_video_holmes_samples,
    load_video_holmes_questions,
    parse_video_holmes_sample,
    sample_video_frames,
)
from .visual_toolbench import (
    VisualToolBenchSample,
    default_visual_toolbench_root,
    iter_visual_toolbench_samples,
    load_visual_toolbench_image,
    parse_visual_toolbench_batch,
    parse_visual_toolbench_sample,
)

__all__ = [
    # VisualToolBench
    "VisualToolBenchSample",
    "default_visual_toolbench_root",
    "iter_visual_toolbench_samples",
    "load_visual_toolbench_image",
    "parse_visual_toolbench_sample",
    "parse_visual_toolbench_batch",
    # TIR-Bench
    "TIRBenchSample",
    "default_tir_bench_root",
    "iter_tir_bench_samples",
    "load_tir_bench_image",
    "parse_tir_bench_sample",
    "parse_tir_bench_batch",
    # Video-Holmes
    "VideoHolmesSample",
    "default_video_holmes_root",
    "iter_video_holmes_samples",
    "load_video_holmes_questions",
    "parse_video_holmes_sample",
    "sample_video_frames",
    # SIV-Bench
    "SIVBenchSample",
    "default_siv_bench_root",
    "iter_siv_bench_samples",
    "load_siv_bench_questions",
    "parse_siv_bench_sample",
]
