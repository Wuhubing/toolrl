"""Task-success reward wiring for veRL (``custom_reward_function``).

This module is the reward side of the ToolRL <-> veRL integration. It exports a
``compute_score`` function matching veRL's documented custom-reward signature
(``data_source, solution_str, ground_truth, extra_info``) so it can be wired in
via ``custom_reward_function.path`` / ``.name`` in ``configs/verl_grpo.yaml``.

Reward definition (see ``toolrl/env/task_env.py`` and the README)
---------------------------------------------------------------
The reward is **binary task success only**: ``1.0`` if the final submitted
answer matches ground truth, ``0.0`` otherwise. There is **no partial credit**
for intermediate correct sub-steps. Two independent channels produce the same
signal:

1. During rollout, the :class:`~toolrl.train.verl_tool.ToolRLTool` returns the
   per-step reward from ``TaskEnv.step()`` (``info["reward"]``), which is 0 for
   every ``query`` call and 1.0/0.0 for the ``finish`` call. veRL accumulates
   those into ``extra_info["tool_rewards"]`` (a list).
2. Here, ``compute_score`` reduces that list to the task-success signal — the
   maximum per-step reward, which equals 1.0 iff the rollout ended in a correct
   ``finish`` — and falls back to grading the final answer text directly when no
   tool rewards are present (e.g. single-turn evaluation).

Because the reward is binary and only the ``finish`` step can carry a 1.0,
``max(tool_rewards)`` is the task-success indicator; no credit is given for a
rollout that never correctly finished.
"""

from __future__ import annotations

from typing import Any

from toolrl.env.task_env import answer_matches


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the task-success reward (binary; no partial credit).

    Returns a dict with a ``score`` key (veRL accepts either a float or a dict
    with ``score``). The score is 1.0 iff the rollout produced the correct final
    answer, 0.0 otherwise.
    """
    extra_info = extra_info or {}

    tool_rewards = extra_info.get("tool_rewards")
    if tool_rewards:
        score = float(max(float(r) for r in tool_rewards))
        return {"score": score, "task_success": score}

    score = float(answer_matches(solution_str, ground_truth))
    return {"score": score, "task_success": score}
