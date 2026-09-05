"""Staged post-training pipeline (SPEC 2.4): SFT -> DPO -> GRPO.

Each stage is a runnable script (``sft.py`` / ``dpo.py`` / ``grpo.py``) that:

* imports cleanly **without** a GPU or the heavy frameworks installed (veRL,
  torch, transformers, TRL are imported lazily and only on the real-run path),
* validates its config standalone (``--smoke``), and
* on a GPU machine, launches the real trainer:

  * SFT -> TRL :class:`~trl.SFTTrainer`
  * DPO -> TRL :class:`~trl.DPOTrainer`
  * GRPO -> veRL ``verl.trainer.main_ppo`` with ``algorithm.adv_estimator=grpo``

The GRPO stage is a *thin* wrapper: it wires ToolRL's :class:`~toolrl.env.task_env.TaskEnv`
and its task-success reward into veRL's rollout/tool interface and lets veRL do
the rollout batching, group-relative advantage computation, and the policy
update. It never reimplements the GRPO update rule.
"""

from toolrl.train.config import DPOConfig, GRPOConfig, SFTConfig

__all__ = ["SFTConfig", "DPOConfig", "GRPOConfig"]
