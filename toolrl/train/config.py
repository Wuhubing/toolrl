"""Stage configuration: standalone loading + validation (no heavy deps).

This module is intentionally dependency-light (PyYAML + stdlib) so that every
training/eval script can ``import`` and ``--smoke``-validate its config on a
CPU-only machine with no GPU and none of the heavy frameworks (veRL / torch /
transformers / TRL) installed. The heavy imports live *inside* the ``run_*``
functions of the individual stage scripts, never at module import time.

The three config classes here are plain dataclasses over the loaded YAML
mapping. Validation is explicit (a list of required keys / values), so a bad
config fails loudly with an actionable message instead of blowing up deep
inside a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (safe loader)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path!r} must be a YAML mapping, got {type(data).__name__}")
    return data


def _get(d: dict[str, Any], *path: str, default: Any = ...) -> Any:
    """Dotted-path access into a nested dict, e.g. ``_get(cfg, "a", "b", "c")``."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            if default is not ...:
                return default
            raise ValueError(f"missing config key {'/'.join(path)!r}")
        cur = cur[key]
    return cur


def _require_str(d: dict[str, Any], *path: str) -> str:
    v = _get(d, *path)
    if not isinstance(v, str) or not v:
        raise ValueError(f"config key {'/'.join(path)!r} must be a non-empty string, got {v!r}")
    return v


# --------------------------------------------------------------------------- #
# SFT config
# --------------------------------------------------------------------------- #


@dataclass
class SFTConfig:
    """Configuration for the SFT stage (TRL ``SFTTrainer``).

    Fields mirror a minimal, sensible subset of the TRL/HF training arguments;
    the stage script maps them onto :class:`~transformers.TrainingArguments`
    and :class:`~trl.SFTConfig` at run time.
    """

    model_path: str
    output_dir: str
    train_data: str
    learning_rate: float = 1e-5
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    num_tasks: int = 2048
    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 10

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SFTConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown SFT config keys: {sorted(unknown)}")
        return cls(**{k: d[k] for k in known if k in d})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        return cls.from_dict(load_yaml(path))

    def __post_init__(self) -> None:
        # YAML may parse "1e-5" as a string; coerce numeric fields defensively.
        self.learning_rate = float(self.learning_rate)
        self.num_train_epochs = float(self.num_train_epochs)
        self.per_device_train_batch_size = int(self.per_device_train_batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_seq_length = int(self.max_seq_length)
        self.num_tasks = int(self.num_tasks)
        self.seed = int(self.seed)
        self.save_steps = int(self.save_steps)
        self.logging_steps = int(self.logging_steps)

    def validate(self) -> None:
        _require_str(vars(self), "model_path")
        _require_str(vars(self), "output_dir")
        _require_str(vars(self), "train_data")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")


# --------------------------------------------------------------------------- #
# DPO config
# --------------------------------------------------------------------------- #


@dataclass
class DPOConfig:
    """Configuration for the DPO stage (TRL ``DPOTrainer``)."""

    model_path: str
    output_dir: str
    train_data: str
    learning_rate: float = 5e-6
    beta: float = 0.1
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    max_prompt_length: int = 2048
    num_tasks: int = 2048
    seed: int = 42
    save_steps: int = 500
    logging_steps: int = 10

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DPOConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown DPO config keys: {sorted(unknown)}")
        return cls(**{k: d[k] for k in known if k in d})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOConfig":
        return cls.from_dict(load_yaml(path))

    def __post_init__(self) -> None:
        self.learning_rate = float(self.learning_rate)
        self.beta = float(self.beta)
        self.num_train_epochs = float(self.num_train_epochs)
        self.per_device_train_batch_size = int(self.per_device_train_batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_length = int(self.max_length)
        self.max_prompt_length = int(self.max_prompt_length)
        self.num_tasks = int(self.num_tasks)
        self.seed = int(self.seed)
        self.save_steps = int(self.save_steps)
        self.logging_steps = int(self.logging_steps)

    def validate(self) -> None:
        _require_str(vars(self), "model_path")
        _require_str(vars(self), "output_dir")
        _require_str(vars(self), "train_data")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")


# --------------------------------------------------------------------------- #
# GRPO config (veRL)
# --------------------------------------------------------------------------- #


def _iter_leaves(d: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_key, leaf_value)`` for a nested mapping (depth-first)."""
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _iter_leaves(v, full)
        else:
            yield full, v


def _render_leaf(v: Any) -> str | None:
    """Render a single scalar/list value as a Hydra-friendly override string."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if v == "" or any(c in v for c in " \t\"'[]{}"):
            return f"'{v}'"
        return v
    if isinstance(v, (list, tuple)):
        parts = []
        for item in v:
            if isinstance(item, str):
                parts.append(f"'{item}'")
            else:
                parts.append(str(item))
        return f"[{', '.join(parts)}]"
    return str(v)


@dataclass
class GRPOConfig:
    """Configuration for the GRPO stage: a thin wrapper over a veRL config.

    The YAML this wraps (``configs/verl_grpo.yaml``) is expressed in veRL's own
    config vocabulary (``algorithm`` / ``data`` / ``actor_rollout_ref`` /
    ``reward_model`` / ``custom_reward_function`` / ``trainer``). This class
    only *validates* that vocabulary and turns it into the CLI overrides that
    ``python -m verl.trainer.main_ppo`` consumes. It contains **no** advantage /
    update / rollout logic — that is all veRL's.
    """

    config_path: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], config_path: str = "") -> "GRPOConfig":
        return cls(config_path=config_path, raw=dict(d))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GRPOConfig":
        return cls(config_path=str(path), raw=load_yaml(path))

    # -- accessors --------------------------------------------------------- #

    @property
    def adv_estimator(self) -> str:
        return str(_get(self.raw, "algorithm", "adv_estimator", default="gae"))

    @property
    def group_size(self) -> int:
        return int(_get(self.raw, "actor_rollout_ref", "rollout", "n", default=1))

    @property
    def model_path(self) -> str:
        return str(_get(self.raw, "actor_rollout_ref", "model", "path", default=""))

    @property
    def train_batch_size(self) -> int:
        return int(_get(self.raw, "data", "train_batch_size", default=0))

    @property
    def use_kl_loss(self) -> bool:
        return bool(_get(self.raw, "actor_rollout_ref", "actor", "use_kl_loss", default=False))

    # -- validation -------------------------------------------------------- #

    def validate(self) -> None:
        algo = _get(self.raw, "algorithm", default={})
        if not isinstance(algo, dict):
            raise ValueError("'algorithm' must be a mapping")
        if self.adv_estimator != "grpo":
            raise ValueError(
                f"algorithm.adv_estimator must be 'grpo' for the GRPO stage, "
                f"got {self.adv_estimator!r} (this stage must not reimplement "
                f"the advantage rule)"
            )
        if self.group_size < 2:
            raise ValueError(
                "actor_rollout_ref.rollout.n (GRPO group size) must be > 1, "
                f"got {self.group_size}"
            )
        if self.use_kl_loss is not True:
            raise ValueError(
                "actor_rollout_ref.actor.use_kl_loss must be True for GRPO "
                "(KL is added to the actor loss, not the reward)"
            )
        if self.train_batch_size < 1:
            raise ValueError("data.train_batch_size must be >= 1")

        rollout = _get(self.raw, "actor_rollout_ref", "rollout", default={})
        mode = rollout.get("mode")
        if mode not in (None, "async", "sync"):
            raise ValueError(f"actor_rollout_ref.rollout.mode must be 'async'/'sync', got {mode!r}")

        multi_turn = rollout.get("multi_turn", {})
        if isinstance(multi_turn, dict):
            enabled = multi_turn.get("enable", False)
        else:
            enabled = bool(multi_turn)
        if not enabled:
            raise ValueError(
                "actor_rollout_ref.rollout.multi_turn.enable must be True to drive "
                "the multi-turn TaskEnv via veRL's tool agent loop"
            )

        agent = rollout.get("agent", {})
        loop = agent.get("default_agent_loop")
        if loop not in (None, "tool_agent"):
            raise ValueError(
                f"actor_rollout_ref.rollout.agent.default_agent_loop must be "
                f"'tool_agent', got {loop!r}"
            )

        reward = _get(self.raw, "custom_reward_function", default={})
        if not reward.get("path"):
            raise ValueError(
                "custom_reward_function.path must point at the task-success "
                "reward module (toolrl/train/reward.py)"
            )

    # -- CLI override rendering ------------------------------------------- #

    # Top-level keys that are ToolRL metadata, not veRL config, and therefore
    # must NOT be forwarded to ``verl.trainer.main_ppo``.
    _NON_VERL_TOP_LEVEL = {"toolrl"}

    def to_verl_args(self) -> list[str]:
        """Flatten the config into ``key=value`` overrides for veRL's Hydra CLI."""
        args: list[str] = []
        for key, val in _iter_leaves(self.raw):
            top = key.split(".", 1)[0]
            if top in self._NON_VERL_TOP_LEVEL:
                continue
            rendered = _render_leaf(val)
            if rendered is None:
                continue
            args.append(f"{key}={rendered}")
        return args

    def build_command(self) -> list[str]:
        """The exact command that launches veRL's GRPO trainer."""
        return ["python3", "-m", "verl.trainer.main_ppo", *self.to_verl_args()]
