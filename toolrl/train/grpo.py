"""GRPO stage (SPEC 2.4): a **thin** wrapper around veRL's GRPO trainer.

This stage intentionally contains **no** advantage computation, no
group-relative normalization, and no policy-gradient update. All of that is
veRL's. What this script does:

1. Loads and validates ``configs/verl_grpo.yaml`` (veRL's own config
   vocabulary: ``algorithm.adv_estimator=grpo``, ``actor_rollout_ref.rollout.n``
   group size, ``use_kl_loss``, ``multi_turn.enable`` + ``tool_config_path``,
   ``custom_reward_function``).
2. (Optionally) builds the rollout parquet dataset — one row per task embedding
   the serialized schema + task so :class:`~toolrl.train.verl_tool.ToolRLTool`
   can reconstruct the :class:`~toolrl.env.task_env.TaskEnv` per rollout.
3. Launches ``python -m verl.trainer.main_ppo`` with the config flattened to
   Hydra CLI overrides. veRL does rollout batching, drives the tool agent loop,
   computes group-relative advantages, and updates the policy.

The ToolRL <-> veRL integration split:
* **ToolRL owns** ``TaskEnv`` (multi-turn step/reset + task-success reward) and
  its wiring into veRL's tool interface (``toolrl/train/verl_tool.py``) and
  reward interface (``toolrl/train/reward.py``).
* **veRL owns** the agent loop, rollout batching, group-relative advantage, KL
  regularization in the loss, and the GRPO update rule.

``--smoke`` validates the config and prints the resolved veRL command without
importing veRL or touching a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from toolrl.train.config import GRPOConfig

__all__ = ["GRPOConfig", "main"]


def run_smoke(cfg: GRPOConfig) -> int:
    cfg.validate()
    print("toolrl GRPO --smoke OK")
    print(f"  adv_estimator: {cfg.adv_estimator}")
    print(f"  group_size (rollout.n): {cfg.group_size}")
    print(f"  train_batch_size: {cfg.train_batch_size}")
    print(f"  use_kl_loss: {cfg.use_kl_loss}")
    print(f"  model_path: {cfg.model_path}")
    print("  resolved veRL command:")
    print("    " + " ".join(cfg.build_command()))
    print("  (veRL NOT imported; no GPU required for validation)")
    return 0


def run_train(cfg: GRPOConfig) -> int:
    """Launch veRL's GRPO trainer as a subprocess (veRL imported only by veRL)."""
    import subprocess

    # veRL must be on PATH; do not import it here so this module stays
    # importable without veRL installed.
    command = cfg.build_command()
    print("toolrl GRPO: launching veRL trainer:")
    print("  " + " ".join(command))
    return subprocess.call(command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ToolRL GRPO stage (thin veRL wrapper)")
    parser.add_argument("--config", default="configs/verl_grpo.yaml", help="veRL GRPO config YAML")
    parser.add_argument("--smoke", action="store_true", help="validate config + print command (no GPU)")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="generate the GRPO rollout parquet and exit",
    )
    args = parser.parse_args(argv)

    cfg = GRPOConfig.from_yaml(args.config)
    cfg.validate()

    if args.smoke:
        return run_smoke(cfg)

    if args.prepare_data:
        from toolrl.train.config import _get
        from toolrl.train.data import build_grpo_rows, write_grpo_parquet

        num_tasks = int(_get(cfg.raw, "toolrl", "num_tasks", default=2048))
        seed = int(_get(cfg.raw, "toolrl", "seed", default=42))
        train_files = cfg.raw.get("data", {}).get("train_files", "data/grpo/train.parquet")
        rows = build_grpo_rows(num_tasks=num_tasks, seed=seed)
        write_grpo_parquet(rows, train_files)
        print(f"toolrl GRPO data written to {train_files}")
        return 0

    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
