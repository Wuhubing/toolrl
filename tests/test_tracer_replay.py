"""Tests for execution tracing + deterministic replay (SPEC 2.5)."""

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.data.task_generator import TaskGenerator
from toolrl.env.task_env import TaskEnv
from toolrl.env.tracer import RolloutLog
from toolrl.replay.deterministic_replay import replay, verify_replay


def _run_rollout(schema, task, answer):
    env = TaskEnv(task, schema, max_turns=5)
    env.reset()
    for sql in task.gold_sql:
        env.step({"name": "query", "arguments": {"sql": sql}})
    env.step({"name": "finish", "arguments": {"answer": answer}})
    log = env.rollout_log()
    env.teardown()
    return log


def test_log_serialization_roundtrip(generator):
    schema = generator.generate("ecommerce", seed=51)
    task = TaskGenerator().generate_tasks(schema, n=1)[0]
    log = _run_rollout(schema, task, task.ground_truth)
    d = log.to_dict()
    restored = RolloutLog.from_dict(d)
    assert restored.seed == log.seed
    assert restored.namespace == log.namespace
    assert len(restored.trace) == len(log.trace)
    # schema structure round-trips identically
    assert restored.schema.to_dict()["tables"] == log.schema.to_dict()["tables"]
    assert restored.schema.seed_rows == log.schema.seed_rows


def test_replay_reconstructs_pre_rollout_state(generator):
    schema = generator.generate("hr", seed=52)
    task = TaskGenerator().generate_tasks(schema, n=1)[0]
    log = _run_rollout(schema, task, task.ground_truth)

    from toolrl.env.sandbox import Sandbox

    handle = replay(log)
    fresh = replay(log)
    try:
        for table in schema.tables:
            q = f'SELECT * FROM "{table.name}"'
            got = fresh.backend.execute(fresh.backend_handle, q)
            assert got.status == "ok"
    finally:
        Sandbox(fresh.backend).teardown(fresh)
        Sandbox(handle.backend).teardown(handle)


def test_verify_replay_is_deterministic(generator):
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=53)
        task = TaskGenerator().generate_tasks(schema, n=1)[0]
        log = _run_rollout(schema, task, task.ground_truth)
        report = verify_replay(log)
        assert report.deterministic, report.failures
        assert report.replayed_steps >= 2


def test_tracer_captures_timing_per_step(generator):
    schema = generator.generate("logistics", seed=54)
    task = TaskGenerator().generate_tasks(schema, n=1)[0]
    log = _run_rollout(schema, task, task.ground_truth)
    for entry in log.trace:
        assert entry.timestamp >= 0.0
        assert entry.duration_ms >= 0.0
    # every model query is present, in order, with its result
    queries = [e for e in log.trace if e.kind == "query"]
    assert [e.sql for e in queries] == task.gold_sql
