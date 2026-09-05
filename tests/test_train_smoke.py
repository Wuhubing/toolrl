"""Tests for the training stage --smoke paths and veRL wiring (no GPU)."""

from pathlib import Path

import pytest

from toolrl.train import dpo, grpo, sft
from toolrl.train.config import DPOConfig, GRPOConfig, SFTConfig
from toolrl.train.reward import compute_score
from toolrl.train.verl_tool import ToolRLTool, openai_tool_schema

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sft_smoke():
    assert sft.main(["--config", str(CONFIGS / "sft.yaml"), "--smoke"]) == 0


def test_dpo_smoke():
    assert dpo.main(["--config", str(CONFIGS / "dpo.yaml"), "--smoke"]) == 0


def test_grpo_smoke():
    assert grpo.main(["--config", str(CONFIGS / "verl_grpo.yaml"), "--smoke"]) == 0


def test_grpo_config_importable_from_stage():
    from toolrl.train.grpo import GRPOConfig as G

    assert G is GRPOConfig


def test_verl_tool_imports_without_verl():
    # Importing the module must not require veRL. The class exists either way.
    assert ToolRLTool.__name__ == "ToolRLTool"
    assert openai_tool_schema()["function"]["name"] == "toolrl"


def test_verl_tool_requires_verl_to_instantiate():
    from toolrl.train import verl_tool

    if verl_tool._HAS_VERL:
        pytest.skip("veRL is installed in this environment")
    with pytest.raises(ImportError, match="veRL"):
        ToolRLTool({}, None)


def test_compute_score_tool_rewards_channel():
    # A correct finish contributes a 1.0 in tool_rewards -> score 1.0.
    assert compute_score("toolrl", "", "42", {"tool_rewards": [0.0, 0.0, 1.0]})["score"] == 1.0
    # No correct finish -> score 0.0 (no partial credit for correct queries).
    assert compute_score("toolrl", "", "42", {"tool_rewards": [0.0, 0.0]})["score"] == 0.0


def test_compute_score_fallback_matches_answer():
    assert compute_score("toolrl", "42", "42")["score"] == 1.0
    assert compute_score("toolrl", "41", "42")["score"] == 0.0
    assert compute_score("toolrl", "West", "west")["score"] == 1.0


def test_sft_dpo_configs_present():
    SFTConfig.from_yaml(CONFIGS / "sft.yaml").validate()
    DPOConfig.from_yaml(CONFIGS / "dpo.yaml").validate()
