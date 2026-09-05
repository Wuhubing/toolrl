"""SFT stage (SPEC 2.4): TRL ``SFTTrainer`` on gold tool-call trajectories.

Rationale for TRL over veRL here: veRL's strength is RL (its multi-turn agent
loop + GRPO/PPO update), and standing up its FSDP + Ray stack just to run plain
supervised fine-tuning is disproportionate. TRL's ``SFTTrainer`` operates
directly on HuggingFace checkpoints — the exact format the DPO stage consumes
and the GRPO stage resumes from — so SFT and DPO both use TRL on the same
checkpoints, and only the GRPO stage (where the RL actually happens) depends on
veRL. See the README for the full rationale.

The script imports cleanly and validates its config with ``--smoke`` on a
CPU-only machine; ``transformers``/``trl`` are imported lazily, only on the real
run path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from toolrl.train.config import SFTConfig


def run_smoke(cfg: SFTConfig) -> int:
    cfg.validate()
    print("toolrl SFT --smoke OK")
    print(f"  model_path:   {cfg.model_path}")
    print(f"  train_data:   {cfg.train_data}")
    print(f"  output_dir:   {cfg.output_dir}")
    print(f"  lr:           {cfg.learning_rate}")
    print(f"  epochs:       {cfg.num_train_epochs}")
    print(f"  num_tasks:    {cfg.num_tasks}")
    print("  (no GPU / model loaded — config validated standalone)")
    return 0


def build_dataset(cfg: SFTConfig) -> Any:
    """Build the SFT dataset from gold trajectories (pandas/datasets lazy)."""
    from toolrl.train.data import build_sft_examples, write_grpo_parquet

    examples = build_sft_examples(num_tasks=cfg.num_tasks, seed=cfg.seed)
    path = Path(cfg.train_data)
    if path.suffix == ".parquet":
        write_grpo_parquet(examples, path)
    else:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
    return examples


def run_train(cfg: SFTConfig) -> int:
    import torch  # noqa: F401  (lazy; asserts a working torch install)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    examples = build_dataset(cfg)
    dataset = Dataset.from_list([{"text": e["prompt"] + "\n" + e["completion"]} for e in examples])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    trl_cfg = TRLSFTConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_seq_length=cfg.max_seq_length,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=trl_cfg,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model(Path(cfg.output_dir) / "final")
    tokenizer.save_pretrained(Path(cfg.output_dir) / "final")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ToolRL SFT stage (TRL SFTTrainer)")
    parser.add_argument("--config", default="configs/sft.yaml", help="path to SFT config YAML")
    parser.add_argument("--smoke", action="store_true", help="validate config only (no GPU)")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="generate the SFT dataset from gold trajectories and exit",
    )
    args = parser.parse_args(argv)

    cfg = SFTConfig.from_yaml(args.config)
    cfg.validate()

    if args.smoke:
        return run_smoke(cfg)

    if args.prepare_data:
        build_dataset(cfg)
        print(f"toolrl SFT data written to {cfg.train_data}")
        return 0

    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
