"""Tests for stage config loading + validation (standalone, no GPU)."""

from pathlib import Path

import pytest
import yaml

from toolrl.train.config import DPOConfig, GRPOConfig, SFTConfig, load_yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_verl_grpo_yaml_is_valid_yaml():
    data = yaml.safe_load(open(CONFIGS / "verl_grpo.yaml"))
    assert isinstance(data, dict)
    assert data["algorithm"]["adv_estimator"] == "grpo"


def test_grpo_config_validates():
    cfg = GRPOConfig.from_yaml(CONFIGS / "verl_grpo.yaml")
    cfg.validate()  # must not raise
    assert cfg.adv_estimator == "grpo"
    assert cfg.group_size > 1
    assert cfg.use_kl_loss is True


def test_grpo_config_rejects_non_grpo():
    cfg = GRPOConfig.from_dict({"algorithm": {"adv_estimator": "gae"}, "actor_rollout_ref": {}})
    with pytest.raises(ValueError, match="adv_estimator"):
        cfg.validate()


def test_grpo_config_rejects_small_group():
    cfg = GRPOConfig.from_dict(
        {
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": True},
                "rollout": {"n": 1, "multi_turn": {"enable": True}, "agent": {"default_agent_loop": "tool_agent"}},
            },
            "data": {"train_batch_size": 8},
            "custom_reward_function": {"path": "toolrl/train/reward.py"},
        }
    )
    with pytest.raises(ValueError, match="group size"):
        cfg.validate()


def test_grpo_to_verl_args_excludes_toolrl_metadata():
    cfg = GRPOConfig.from_yaml(CONFIGS / "verl_grpo.yaml")
    args = cfg.to_verl_args()
    assert any(a.startswith("algorithm.adv_estimator=") for a in args)
    assert any(a.startswith("actor_rollout_ref.rollout.n=") for a in args)
    assert not any(a.startswith("toolrl.") for a in args)
    assert "toolrl.num_tasks=2048" not in args


def test_grpo_build_command():
    cfg = GRPOConfig.from_yaml(CONFIGS / "verl_grpo.yaml")
    cmd = cfg.build_command()
    assert cmd[0] == "python3"
    assert "-m" in cmd and "verl.trainer.main_ppo" in cmd


def test_sft_config_validates():
    cfg = SFTConfig.from_yaml(CONFIGS / "sft.yaml")
    cfg.validate()
    assert cfg.model_path == "base"


def test_dpo_config_validates():
    cfg = DPOConfig.from_yaml(CONFIGS / "dpo.yaml")
    cfg.validate()
    assert cfg.beta > 0


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="unknown"):
        SFTConfig.from_dict({"model_path": "x", "output_dir": "y", "train_data": "z", "bogus": 1})


def test_schema_domains_yaml_unchanged_and_valid():
    # configs/schema_domains.yaml (TR-001) must remain valid YAML.
    data = load_yaml(CONFIGS / "schema_domains.yaml")
    assert set(data["domains"]) == {"ecommerce", "logistics", "hr"}
