"""Tests for the database backend abstraction (SPEC 2.2)."""

import pytest

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.env.backend import SQLiteBackend


def test_spawn_execute_teardown(generator, backend):
    schema = generator.generate("ecommerce", seed=21)
    handle = backend.spawn(schema, "ns1")
    res = backend.execute(handle, "SELECT COUNT(*) FROM customers")
    assert res.status == "ok"
    assert res.rows[0][0] > 0
    backend.teardown(handle)
    # executing after teardown yields a failure result, not a crash
    res = backend.execute(handle, "SELECT 1")
    assert res.status == "error"


def test_namespaces_are_isolated(generator, backend):
    schema = generator.generate("hr", seed=22)
    h1 = backend.spawn(schema, "a")
    h2 = backend.spawn(schema, "b")
    # mutate namespace a only
    backend.execute(h1, "INSERT INTO departments (department_id, name) VALUES (999, 'Leak')")
    c1 = backend.execute(h1, "SELECT COUNT(*) FROM departments").rows[0][0]
    c2 = backend.execute(h2, "SELECT COUNT(*) FROM departments").rows[0][0]
    assert c1 == c2 + 1
    backend.teardown(h1)
    backend.teardown(h2)


def test_duplicate_namespace_rejected(generator, backend):
    schema = generator.generate("ecommerce", seed=23)
    backend.spawn(schema, "dup")
    with pytest.raises(ValueError):
        backend.spawn(schema, "dup")


def test_bad_sql_returns_error_result(generator, backend):
    schema = generator.generate("ecommerce", seed=24)
    handle = backend.spawn(schema, "bad")
    res = backend.execute(handle, "SELECT * FROM nonexistent_table")
    assert res.status == "error"
    assert res.error


def test_close_releases_all(generator, backend):
    schema = generator.generate("ecommerce", seed=25)
    backend.spawn(schema, "x")
    backend.spawn(schema, "y")
    backend.close()
    assert backend._conns == {}


def test_render_ddl_has_create_statements(generator):
    from toolrl.env.backend import render_ddl

    schema = generator.generate("ecommerce", seed=26)
    for dialect in ("sqlite", "postgres"):
        stmts = render_ddl(schema, dialect)
        assert len(stmts) == len(schema.tables)
        assert all(s.upper().startswith("CREATE TABLE") for s in stmts)
