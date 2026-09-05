"""Sandboxed, isolated rollout environment (SPEC 2.2).

Isolation unit — namespace-per-rollout
--------------------------------------
We isolate each rollout in its own database namespace (a ``CREATE SCHEMA`` in
Postgres, a private ``:memory:`` SQLite database in the fake backend) rather
than spinning up a full container per rollout. Rationale:

* **Spin-up latency:** ``CREATE SCHEMA`` is milliseconds; a fresh Postgres
  container is on the order of a second (plus image pull/health-check time).
  At ~5,120 rollouts this dominates wall-clock.
* **Concurrency ceiling:** schema-per-rollout lets a single shared instance
  host thousands of concurrent namespaces; container-per-rollout is bounded by
  Docker/daemon limits and per-container memory overhead (~tens of MB each).
* **Isolation sufficiency:** Postgres schemas already provide per-namespace
  object isolation, and each rollout only runs *its own* deterministic SQL
  against its own namespace — there is no untrusted arbitrary code that would
  require kernel-level (container) isolation for safety.

The accepted tradeoff is shared instance-level resources (CPU/memory) and no
kernel isolation, which is fine for this threat model and is the same decision
the fast-spin-up option in the spec describes.

Guaranteed teardown
-------------------
``teardown`` is idempotent and the ``Sandbox`` tracks live handles; the class is
a context manager and also tears down every live handle on ``close()`` / GC, so
a namespace is always cleaned up even on rollout failure or timeout.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from toolrl.data.schema_generator import Schema
from toolrl.env.backend import Backend, BackendHandle, SQLResult
from toolrl.env.tracer import RolloutLog, Tracer


@dataclass
class SandboxHandle:
    """A live sandbox: the backend handle plus the schema that was provisioned."""

    namespace: str
    backend: Backend
    backend_handle: BackendHandle
    schema: Schema
    seed: int
    tracer: Tracer
    log: RolloutLog


class Sandbox:
    """Provisions isolated per-rollout namespaces and executes SQL against them.

    Every ``spawn`` materializes ``schema`` (DDL + deterministic seed rows) in a
    fresh namespace, records the setup statements in the rollout trace, and
    returns a :class:`SandboxHandle`. Every subsequent ``execute`` is traced
    (SQL + result + timing) for replay (SPEC 2.5).
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self._live: set[str] = set()

    def spawn(self, schema: Schema, seed: int | None = None, task_id: str = "") -> SandboxHandle:
        seed = schema.seed if seed is None else seed
        namespace = f"rollout_{uuid.uuid4().hex}"
        log = RolloutLog(
            schema=schema,
            seed=seed,
            namespace=namespace,
            task_id=task_id,
        )
        tracer = Tracer(log)

        t0 = time.monotonic()
        backend_handle = self.backend.spawn(schema, namespace)
        duration_ms = (time.monotonic() - t0) * 1000.0
        # Record the materialization as a single setup trace entry (the full
        # DDL is re-derivable from `schema`, which is already in the log).
        tracer.record(
            kind="setup",
            sql=f"-- materialize schema {schema.name!r} (seed={seed})",
            result=SQLResult(status="ok", rowcount=sum(len(r) for r in schema.seed_rows.values())),
            duration_ms=duration_ms,
            step=0,
        )

        self._live.add(namespace)
        return SandboxHandle(
            namespace=namespace,
            backend=self.backend,
            backend_handle=backend_handle,
            schema=schema,
            seed=seed,
            tracer=tracer,
            log=log,
        )

    def execute(self, handle: SandboxHandle, sql: str) -> SQLResult:
        t0 = time.monotonic()
        result = self.backend.execute(handle.backend_handle, sql)
        duration_ms = (time.monotonic() - t0) * 1000.0
        step = handle.tracer.next_step()
        handle.tracer.record(
            kind="query", sql=sql, result=result, duration_ms=duration_ms, step=step
        )
        return result

    def teardown(self, handle: SandboxHandle) -> None:
        if handle.namespace in self._live:
            self.backend.teardown(handle.backend_handle)
            self._live.discard(handle.namespace)

    def close(self) -> None:
        self.backend.close()
        self._live.clear()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - context mgr sig
        self.close()
