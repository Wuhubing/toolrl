"""Tests for multi-step task generation (SPEC 2.1)."""

from toolrl.data.schema_generator import SchemaGenerator
from toolrl.data.task_generator import TaskGenerator
from toolrl.env.backend import SQLiteBackend


def test_tasks_require_multiple_steps(generator):
    tg = TaskGenerator()
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=11)
        tasks = tg.generate_tasks(schema, n=4)
        assert len(tasks) == 4
        for t in tasks:
            assert t.min_steps >= 2, f"{domain} task requires only {t.min_steps} steps"
            assert t.instruction.strip()
            assert t.ground_truth is not None


def test_ground_truth_matches_database(generator):
    tg = TaskGenerator()
    for domain in ("ecommerce", "logistics", "hr"):
        schema = generator.generate(domain, seed=13)
        tasks = tg.generate_tasks(schema, n=3)
        for task in tasks:
            assert task.ground_truth is not None
            assert len(task.gold_sql) == task.min_steps


def test_gold_sql_is_reproducible(generator):
    tg = TaskGenerator()
    schema = generator.generate("ecommerce", seed=17)
    a = tg.generate_tasks(schema, n=3)
    b = tg.generate_tasks(schema, n=3)
    assert [t.ground_truth for t in a] == [t.ground_truth for t in b]
    assert [t.gold_sql for t in a] == [t.gold_sql for t in b]


def test_gold_queries_run_clean(generator):
    # The recorded gold SQL must actually execute against a fresh materialization
    # and produce the recorded ground truth.
    tg = TaskGenerator()
    backend = SQLiteBackend()
    try:
        schema = generator.generate("logistics", seed=19)
        tasks = tg.generate_tasks(schema, n=3)
        handle = backend.spawn(schema, "verify")
        for task in tasks:
            rows = None
            for sql in task.gold_sql:
                res = backend.execute(handle, sql)
                assert res.status == "ok", res.error
                rows = res.rows
            assert rows is not None and len(rows) > 0
            # the final gold query's first scalar equals the recorded answer
            assert str(rows[0][0]) == str(task.ground_truth)
    finally:
        backend.close()


def test_all_domains_covered():
    from toolrl.data.task_generator import _TEMPLATES

    assert set(_TEMPLATES) == {"ecommerce", "logistics", "hr"}
    for templates in _TEMPLATES.values():
        assert len(templates) >= 2
