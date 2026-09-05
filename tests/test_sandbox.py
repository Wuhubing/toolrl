"""Tests for the sandbox (SPEC 2.2): spawn, tracing, guaranteed teardown."""

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.env.backend import SQLiteBackend
from toolrl.env.sandbox import Sandbox


def test_spawn_and_execute(generator):
    backend = SQLiteBackend()
    sandbox = Sandbox(backend)
    schema = generator.generate("ecommerce", seed=31)
    handle = sandbox.spawn(schema, task_id="t1")
    res = sandbox.execute(handle, "SELECT COUNT(*) FROM orders")
    assert res.status == "ok" and res.rows[0][0] > 0
    sandbox.teardown(handle)
    backend.close()


def test_trace_records_sql_result_and_timing(generator):
    backend = SQLiteBackend()
    sandbox = Sandbox(backend)
    schema = generator.generate("ecommerce", seed=32)
    handle = sandbox.spawn(schema, task_id="t2")
    sandbox.execute(handle, "SELECT 1 AS one")
    sandbox.execute(handle, "SELECT region FROM customers LIMIT 3")
    kinds = [e.kind for e in handle.log.trace]
    assert kinds[0] == "setup"
    assert kinds.count("query") == 2
    for e in handle.log.trace:
        assert e.duration_ms >= 0.0
    # every query entry captured its result
    query_entries = [e for e in handle.log.trace if e.kind == "query"]
    assert all(e.result.status == "ok" for e in query_entries)
    sandbox.teardown(handle)
    backend.close()


def test_teardown_is_idempotent(generator):
    backend = SQLiteBackend()
    sandbox = Sandbox(backend)
    handle = sandbox.spawn(generator.generate("hr", seed=33))
    sandbox.teardown(handle)
    sandbox.teardown(handle)  # must not raise
    backend.close()


def test_context_manager_guarantees_cleanup(generator):
    backend = SQLiteBackend()
    with Sandbox(backend) as sandbox:
        handle = sandbox.spawn(generator.generate("logistics", seed=34))
        assert handle.namespace in sandbox._live
    assert backend._conns == {}  # backend fully closed


def test_spawn_logs_schema_and_seed(generator):
    backend = SQLiteBackend()
    sandbox = Sandbox(backend)
    schema = generator.generate("ecommerce", seed=35)
    handle = sandbox.spawn(schema, seed=35, task_id="t3")
    assert handle.log.schema.name == schema.name
    assert handle.log.seed == 35
    assert handle.log.namespace
    backend.close()
