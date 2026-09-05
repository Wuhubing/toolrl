"""Tests for the multi-turn task environment (SPEC 2.3)."""

import pytest

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.data.task_generator import TaskGenerator
from toolrl.env.task_env import TaskEnv, answer_matches, normalize_answer


@pytest.fixture(scope="module")
def _fixtures():
    generator = SchemaGenerator()
    schema = generator.generate("hr", seed=41)
    task = TaskGenerator().generate_tasks(schema, n=1)[0]
    return schema, task


def test_step_returns_result_done_info(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=5)
    env.reset()
    result, done, info = env.step({"name": "query", "arguments": {"sql": "SELECT 1"}})
    assert isinstance(result, dict)
    assert done is False
    assert set(info) == {"reward", "success", "turns_used", "max_turns", "failure_reason"}
    env.teardown()


def test_correct_answer_yields_task_success(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=5)
    env.reset()
    env.step({"name": "query", "arguments": {"sql": task.gold_sql[0]}})
    env.step({"name": "query", "arguments": {"sql": task.gold_sql[1]}})
    result, done, info = env.step({"name": "finish", "arguments": {"answer": task.ground_truth}})
    assert done is True
    assert result["correct"] is True
    assert info["reward"] == 1.0
    assert info["success"] is True
    env.teardown()


def test_wrong_answer_yields_zero_reward(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=5)
    env.reset()
    result, done, info = env.step({"name": "finish", "arguments": {"answer": "not-the-answer"}})
    assert done is True
    assert info["reward"] == 0.0
    assert info["success"] is False
    env.teardown()


def test_turn_limit_exhaustion_is_failure(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=2)
    env.reset()
    result, done, info = env.step({"name": "query", "arguments": {"sql": "SELECT 1"}})
    assert done is False
    # second (final) turn used on a query -> forced failure, not silent truncation
    result, done, info = env.step({"name": "query", "arguments": {"sql": "SELECT 2"}})
    assert done is True
    assert info["success"] is False
    assert info["reward"] == 0.0
    assert info["failure_reason"] == "turn_limit_exhausted"
    env.teardown()


def test_step_past_limit_terminates(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=1)
    env.reset()
    env.step({"name": "query", "arguments": {"sql": "SELECT 1"}})  # exhausts limit
    result, done, info = env.step({"name": "query", "arguments": {"sql": "SELECT 2"}})
    assert done is True
    assert info["failure_reason"] == "turn_limit_exhausted"
    env.teardown()


def test_step_requires_reset(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=3)
    with pytest.raises(RuntimeError):
        env.step({"name": "query", "arguments": {"sql": "SELECT 1"}})


def test_observation_includes_schema_and_tools(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=3)
    obs = env.reset()
    assert obs["instruction"] == task.instruction
    assert {t["name"] for t in obs["tables"]} == {t.name for t in schema.tables}
    assert {t["name"] for t in obs["tools"]} == {"query", "finish"}
    env.teardown()


def test_answer_normalization():
    assert answer_matches(42, "42")
    assert answer_matches("  West ", "west")
    assert answer_matches(3.0, 3)
    assert not answer_matches(42, 43)
    assert normalize_answer("  Delivered ") == "delivered"


def test_teardown_on_exception_guaranteed(_fixtures):
    schema, task = _fixtures
    env = TaskEnv(task, schema, max_turns=3)
    env.reset()
    namespace = env.handle.namespace
    try:
        with pytest.raises(Exception):
            raise RuntimeError("boom")
    finally:
        env.teardown()
    assert env.handle is None
    assert namespace not in env.sandbox._live
