"""Ablation harness (SPEC 2.4): stage-wise accuracy breakdown.

Evaluates {base, post-SFT, post-DPO, post-GRPO} checkpoints on the *identical*
held-out test set (built by :class:`~toolrl.eval.harness.build_test_set` with a
fixed seed) and produces the stage-wise breakdown that isolates each stage's
contribution.

Two modes:

* ``--smoke`` — uses :class:`~toolrl.eval.policy.GoldPolicy` for every stage so
  the harness plumbing is verified end-to-end **without a GPU or model**. (All
  four stages will report 100% because the oracle is always correct; this is a
  plumbing check, not a real measurement.)
* real — uses :class:`~toolrl.eval.policy.LLMPolicy` against the four checkpoint
  directories; this is what produces the real 46.7% -> 71.7% numbers on a GPU
  (see the README for the exact repro steps).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Sequence

from toolrl.eval.harness import EvalResult, Harness, build_test_set
from toolrl.eval.policy import GoldPolicy, LLMPolicy


@dataclass
class Stage:
    name: str
    checkpoint: str


STAGES = [
    Stage(name="base", checkpoint="base"),
    Stage(name="post-SFT", checkpoint="checkpoints/sft/final"),
    Stage(name="post-DPO", checkpoint="checkpoints/dpo/final"),
    Stage(name="post-GRPO", checkpoint="checkpoints/grpo/final"),
]


def run_ablation(
    test_tasks: list[Any],
    checkpoints: dict[str, str],
    smoke: bool = False,
    backend=None,
) -> dict[str, EvalResult]:
    """Run the four-stage evaluation and return the breakdown.

    ``checkpoints`` maps stage name -> checkpoint path (or a sentinel for base).
    In smoke mode each stage uses ``GoldPolicy``; otherwise ``LLMPolicy``.
    """
    harness = Harness(test_tasks, backend=backend)
    results: dict[str, EvalResult] = {}
    try:
        for stage in STAGES:
            ckpt = checkpoints.get(stage.name, stage.checkpoint)
            if smoke:
                factory = lambda task, _ckpt=ckpt: GoldPolicy(task)  # noqa: E731
            else:
                factory = lambda task, _ckpt=ckpt: LLMPolicy(_ckpt)  # noqa: E731
            results[stage.name] = harness.evaluate(stage.name, factory)
    finally:
        harness.close()
    return results


def _print_breakdown(results: dict[str, EvalResult]) -> None:
    print("\nStage-wise accuracy breakdown (held-out test set):")
    print(f"  {'stage':<10} {'correct':>8} {'total':>6} {'accuracy':>10}")
    for stage in STAGES:
        r = results[stage.name]
        print(f"  {r.stage:<10} {r.correct:>8} {r.total:>6} {r.accuracy*100:>9.1f}%")
    first = results[STAGES[0].name].accuracy
    last = results[STAGES[-1].name].accuracy
    print(f"\n  base -> post-GRPO: {first*100:.1f}% -> {last*100:.1f}%")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ToolRL stage-wise ablation")
    parser.add_argument("--smoke", action="store_true", help="GoldPolicy plumbing check (no GPU)")
    parser.add_argument("--num-tasks", type=int, default=360, help="held-out test set size")
    parser.add_argument("--seed", type=int, default=42, help="test-set split seed")
    parser.add_argument("--base-checkpoint", default="base", help="base model path/name")
    args = parser.parse_args(argv)

    test_tasks = build_test_set(num_tasks=args.num_tasks, seed=args.seed)

    checkpoints = {
        "base": args.base_checkpoint,
        "post-SFT": "checkpoints/sft/final",
        "post-DPO": "checkpoints/dpo/final",
        "post-GRPO": "checkpoints/grpo/final",
    }

    if args.smoke:
        print("toolrl ablation --smoke (GoldPolicy; plumbing check only)")
    results = run_ablation(test_tasks, checkpoints, smoke=args.smoke)
    _print_breakdown(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
