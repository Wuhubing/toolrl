"""Database backend abstraction (SPEC 2.2 foundation).

Every environment module talks to a database through the :class:`Backend`
interface so the whole stack is unit-testable **without** a live Postgres or
Docker. Two implementations ship here:

* :class:`SQLiteBackend` — an in-memory fake used by the test suite. Each
  rollout namespace maps to a private ``:memory:`` SQLite connection, so it
  actually executes SQL for real (deterministically) while remaining fully
  isolated.
* :class:`PostgresBackend` — the production backend, which talks to a shared
  Postgres instance (provisioned via Docker) and isolates rollouts via
  ``CREATE SCHEMA`` / ``DROP SCHEMA`` (namespace-per-rollout, see
  ``sandbox.py`` for the justification). ``psycopg`` is imported lazily so
  importing this module never requires Postgres to be installed.

Both backends render the dialect-neutral :class:`~toolrl.data.schema_generator.Schema`
into their own DDL dialect and insert the schema's deterministic seed rows.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from toolrl.data.schema_generator import Schema, _topological_order


@dataclass
class SQLResult:
    """The structured result of executing one SQL statement."""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    rowcount: int = 0
    status: str = "ok"  # "ok" | "error"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": [list(r) for r in self.rows],
            "rowcount": self.rowcount,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SQLResult":
        return cls(
            columns=list(d.get("columns", [])),
            rows=[tuple(r) for r in d.get("rows", [])],
            rowcount=int(d.get("rowcount", 0)),
            status=d.get("status", "ok"),
            error=d.get("error"),
        )

    @classmethod
    def failure(cls, message: str) -> "SQLResult":
        return cls(status="error", error=message)


@dataclass
class BackendHandle:
    """Opaque handle to a provisioned, isolated namespace within a backend."""

    namespace: str
    state: Any = None


class Backend(ABC):
    """Abstract per-rollout database backend."""

    dialect: str = "sqlite"

    @abstractmethod
    def spawn(self, schema: Schema, namespace: str) -> BackendHandle:
        """Provision an isolated namespace and materialize ``schema`` in it."""

    @abstractmethod
    def teardown(self, handle: BackendHandle) -> None:
        """Destroy the namespace; must be safe to call more than once."""

    @abstractmethod
    def execute(self, handle: BackendHandle, sql: str) -> SQLResult:
        """Execute one SQL statement inside the namespace, returning its result."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release backend-wide resources (connection pools, etc.)."""


# --------------------------------------------------------------------------- #
# SQLite in-memory backend (used for tests)
# --------------------------------------------------------------------------- #


class SQLiteBackend(Backend):
    """In-memory fake backend: one ``:memory:`` SQLite connection per namespace."""

    dialect = "sqlite"

    def __init__(self) -> None:
        self._conns: dict[str, sqlite3.Connection] = {}

    def spawn(self, schema: Schema, namespace: str) -> BackendHandle:
        if namespace in self._conns:
            raise ValueError(f"namespace {namespace!r} already exists")
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        self._conns[namespace] = conn
        try:
            for stmt in _render_ddl(schema, "sqlite"):
                conn.execute(stmt)
            for stmt, params in _render_inserts(schema):
                conn.execute(stmt, params)
            conn.commit()
        except Exception:
            self.teardown(BackendHandle(namespace, conn))
            raise
        return BackendHandle(namespace, conn)

    def teardown(self, handle: BackendHandle) -> None:
        conn = self._conns.pop(handle.namespace, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

    def execute(self, handle: BackendHandle, sql: str) -> SQLResult:
        conn = self._conns.get(handle.namespace)
        if conn is None:
            return SQLResult.failure(f"namespace {handle.namespace!r} is not provisioned")
        try:
            cur = conn.execute(sql)
            if cur.description is None:  # non-SELECT (DDL/DML)
                conn.commit()
                return SQLResult(columns=[], rows=[], rowcount=cur.rowcount or 0)
            rows = [tuple(r) for r in cur.fetchall()]
            columns = [d[0] for d in cur.description]
            return SQLResult(columns=columns, rows=rows, rowcount=len(rows))
        except Exception as exc:  # noqa: BLE001 - surface SQL errors as results
            return SQLResult.failure(str(exc))

    def close(self) -> None:
        for ns in list(self._conns):
            self.teardown(BackendHandle(ns))
        self._conns.clear()


# --------------------------------------------------------------------------- #
# Postgres backend (production; psycopg imported lazily)
# --------------------------------------------------------------------------- #


class PostgresBackend(Backend):
    """Production backend: namespace-per-rollout via ``CREATE SCHEMA``.

    Connects to a single shared Postgres instance (provisioned with Docker)
    using ``psycopg``. The ``dsn`` follows libpq syntax, e.g.
    ``postgresql://user:pass@localhost:5432/toolrl``.
    """

    dialect = "postgres"

    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or "postgresql://postgres:postgres@localhost:5432/toolrl"
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg  # lazy: only needed in production

            self._conn = psycopg.connect(self._dsn)
            self._conn.autocommit = True
        return self._conn

    def spawn(self, schema: Schema, namespace: str) -> BackendHandle:
        conn = self._get_conn()
        safe = _quote_ident(namespace)
        conn.execute(f'CREATE SCHEMA {safe}')
        conn.execute(f'SET search_path TO {safe}')
        try:
            for stmt in _render_ddl(schema, "postgres"):
                conn.execute(stmt)
            for stmt, params in _render_inserts(schema, "postgres"):
                conn.execute(stmt, params)
        except Exception:
            self.teardown(BackendHandle(namespace))
            raise
        return BackendHandle(namespace)

    def teardown(self, handle: BackendHandle) -> None:
        try:
            conn = self._get_conn()
            safe = _quote_ident(handle.namespace)
            conn.execute(f'DROP SCHEMA IF EXISTS {safe} CASCADE')
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def execute(self, handle: BackendHandle, sql: str) -> SQLResult:
        try:
            conn = self._get_conn()
            safe = _quote_ident(handle.namespace)
            conn.execute(f'SET search_path TO {safe}')
            cur = conn.execute(sql)
            if cur.description is None:
                return SQLResult(columns=[], rows=[], rowcount=cur.rowcount or 0)
            rows = [tuple(r) for r in cur.fetchall()]
            columns = [d.name for d in cur.description]
            return SQLResult(columns=columns, rows=rows, rowcount=len(rows))
        except Exception as exc:  # noqa: BLE001
            return SQLResult.failure(str(exc))

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()


# --------------------------------------------------------------------------- #
# Dialect-aware DDL rendering
# --------------------------------------------------------------------------- #

_TYPE_MAP = {
    "sqlite": {
        "INTEGER": "INTEGER",
        "TEXT": "TEXT",
        "REAL": "REAL",
        "BOOLEAN": "INTEGER",
        "TIMESTAMP": "TEXT",
    },
    "postgres": {
        "INTEGER": "INTEGER",
        "TEXT": "TEXT",
        "REAL": "DOUBLE PRECISION",
        "BOOLEAN": "BOOLEAN",
        "TIMESTAMP": "TIMESTAMP",
    },
}


def _render_ddl(schema: Schema, dialect: str) -> list[str]:
    """Render ``CREATE TABLE`` statements for ``schema`` in the given dialect."""
    type_map = _TYPE_MAP[dialect]
    stmts: list[str] = []
    for table in _topological_order(schema.tables):
        col_defs = []
        for col in table.columns:
            ctype = type_map[col.type]
            parts = [f"{_quote_ident(col.name)} {ctype}"]
            if not col.nullable:
                parts.append("NOT NULL")
            if col.unique:
                parts.append("UNIQUE")
            col_defs.append(" ".join(parts))
        col_defs.append(f"PRIMARY KEY ({_quote_ident(table.primary_key)})")
        for fk in table.foreign_keys:
            col_defs.append(
                f"FOREIGN KEY ({_quote_ident(fk.column)}) "
                f"REFERENCES {_quote_ident(fk.ref_table)} ({_quote_ident(fk.ref_column)})"
            )
        stmts.append(
            f"CREATE TABLE {_quote_ident(table.name)} (\n  "
            + ",\n  ".join(col_defs)
            + "\n)"
        )
    return stmts


def _render_inserts(schema: Schema, dialect: str = "sqlite") -> list[tuple[str, tuple[Any, ...]]]:
    """Render parameterized ``INSERT`` statements for all seed rows."""
    placeholder = "%s" if dialect == "postgres" else "?"
    stmts: list[tuple[str, tuple[Any, ...]]] = []
    for table in schema.tables:
        rows = schema.seed_rows.get(table.name, [])
        if not rows:
            continue
        cols = [c.name for c in table.columns]
        for row in rows:
            values = tuple(row.get(c) for c in cols)
            stmts.append(
                (
                    f"INSERT INTO {_quote_ident(table.name)} "
                    f"({', '.join(_quote_ident(c) for c in cols)}) "
                    f"VALUES ({', '.join(placeholder for _ in cols)})",
                    values,
                )
            )
    return stmts


def _quote_ident(name: str) -> str:
    """Double-quote an SQL identifier (works for both SQLite and Postgres)."""
    return '"' + name.replace('"', '""') + '"'


def render_ddl(schema: Schema, dialect: str = "sqlite") -> list[str]:
    """Public helper: render DDL for a schema in a given dialect."""
    return _render_ddl(schema, dialect)
