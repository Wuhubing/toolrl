"""Tests for the eval harness and stage-wise ablation (no GPU)."""

from toolrl.data.schema_generator import assert_disjoint
from toolrl.eval.ablation import STAGES, run_ablation
from toolrl.eval.harness import Harness, build_test_set
from toolrl.eval.policy import GoldPolicy, WrongAnswerPolicy
from toolrl.train.data import generate_split

# Small pool so tests stay fast; the real path uses the canonical pool size.
POOL = 20


def test_split_is_disjoint_at_schema_level():
    train, test = generate_split(seed=42, num_schemas_per_domain=POOL)
    train_schemas = [s for s, _ in train]
    test_schemas = [s for s, _ in test]
    assert train_schemas and test_schemas
    assert_disjoint(train_schemas, test_schemas)  # must not raise


def test_build_test_set_uses_test_side_of_canonical_split():
    train, test = generate_split(seed=42, num_schemas_per_domain=POOL)
    test_pairs = build_test_set(num_tasks=8, seed=42, num_schemas_per_domain=POOL)
    test_names = {s.name for s, _ in test_pairs}
    train_names = {s.name for s, _ in train}
    assert test_names and test_names.isdisjoint(train_names)
    assert test_names <= {s.name for s, _ in test}


def test_gold_policy_scores_perfectly():
    test_pairs = build_test_set(num_tasks=8, seed=42, num_schemas_per_domain=POOL)
    assert test_pairs
    harness = Harness(test_pairs)
    try:
        result = harness.evaluate("post-GRPO", lambda t: GoldPolicy(t))
    finally:
        harness.close()
    assert result.total == len(test_pairs)
    assert result.correct == result.total
    assert result.accuracy == 1.0


def test_wrong_answer_policy_scores_zero():
    test_pairs = build_test_set(num_tasks=8, seed=42, num_schemas_per_domain=POOL)
    harness = Harness(test_pairs)
    try:
        result = harness.evaluate("post-GRPO", lambda t: WrongAnswerPolicy(t))
    finally:
        harness.close()
    assert result.correct == 0
    assert result.accuracy == 0.0


def test_ablation_smoke_produces_four_stage_breakdown():
    test_pairs = build_test_set(num_tasks=8, seed=42, num_schemas_per_domain=POOL)
    checkpoints = {s.name: s.checkpoint for s in STAGES}
    results = run_ablation(test_pairs, checkpoints, smoke=True)
    assert set(results) == {"base", "post-SFT", "post-DPO", "post-GRPO"}
    for name in ("base", "post-SFT", "post-DPO", "post-GRPO"):
        assert results[name].stage == name
        assert results[name].accuracy == 1.0  # GoldPolicy is always correct


def test_eval_result_dict():
    test_pairs = build_test_set(num_tasks=4, seed=42, num_schemas_per_domain=POOL)
    harness = Harness(test_pairs)
    try:
        result = harness.evaluate("base", lambda t: GoldPolicy(t))
        d = result.to_dict()
        assert set(d) >= {"stage", "total", "correct", "accuracy", "failures"}
        assert d["accuracy"] == 1.0
    finally:
        harness.close()
