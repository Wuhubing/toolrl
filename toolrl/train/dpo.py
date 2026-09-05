"""DPO stage (SPEC 2.4): TRL ``DPOTrainer`` on gold-vs-failed preference pairs.

Runs on top of the SFT checkpoint. Preference pairs are ``(chosen, rejected)``
where the chosen response is the gold tool-call trajectory and the rejected
response is a deterministic *failed* trajectory (the same gold queries followed
by a wrong final answer) — i.e. higher-reward vs lower-reward behavior for the
same task. TRL is used (not veRL) for the same reason as SFT: DPO is a plain
supervised preference loss that TRL implements directly on HF checkpoints, and
veRL's RL stack is disproportionate for it. See the README.

``--smoke`` validates config without importing transformers/trl.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from toolrl.train.config import DPOConfig


def run_smoke(cfg: DPOConfig) -> int:
    cfg.validate()
    print("toolrl DPO --smoke OK")
    print(f"  model_path:   {cfg.model_path}")
    print(f"  train_data:   {cfg.train_data}")
    print(f"  output_dir:   {cfg.output_dir}")
    print(f"  beta:         {cfg.beta}")
    print(f"  lr:           {cfg.learning_rate}")
    print(f"  num_tasks:    {cfg.num_tasks}")
    print("  (no GPU / model loaded — config validated standalone)")
    return 0


def build_dataset(cfg: DPOConfig) -> Any:
    from toolrl.train.data import build_dpo_examples

    examples = build_dpo_examples(num_tasks=cfg.num_tasks, seed=cfg.seed)
    path = Path(cfg.train_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return examples


def run_train(cfg: DPOConfig) -> int:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    examples = build_dataset(cfg)
    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    trl_cfg = TRLDPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trl_cfg,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(Path(cfg.output_dir) / "final")
    tokenizer.save_pretrained(Path(cfg.output_dir) / "final")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ToolRL DPO stage (TRL DPOTrainer)")
    parser.add_argument("--config", default="configs/dpo.yaml", help="path to DPO config YAML")
    parser.add_argument("--smoke", action="store_true", help="validate config only (no GPU)")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="generate DPO preference pairs and exit",
    )
    args = parser.parse_args(argv)

    cfg = DPOConfig.from_yaml(args.config)
    cfg.validate()

    if args.smoke:
        return run_smoke(cfg)

    if args.prepare_data:
        build_dataset(cfg)
        print(f"toolrl DPO data written to {cfg.train_data}")
        return 0

    return run_train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
