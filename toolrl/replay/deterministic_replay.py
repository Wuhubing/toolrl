"""Deterministic environment replay (SPEC 2.5).

Given a logged rollout, we can (a) reconstruct the environment to the exact
pre-rollout state, and (b) re-run the recorded SQL and confirm every result is
bit-for-bit identical. Together these let us distinguish "the model made a bad
decision" from "the environment behaved non-deterministically" when debugging
training instability.

Determinism rests on two facts:
* the schema's seed rows are generated deterministically from ``seed`` (see
  :class:`~toolrl.data.schema_generator.SchemaGenerator`), and
* the fake backend (SQLite) and the DDL/insert renderer are deterministic, so
  re-materializing the same schema reproduces the identical DB.
"""

from __future__ import annotations

from dataclasses import dataclass

from toolrl.env.backend import Backend, SQLiteBackend
from toolrl.env.sandbox import Sandbox, SandboxHandle
from toolrl.env.tracer import RolloutLog


@dataclass
class ReplayReport:
    replayed_steps: int
    matching_steps: int
    ok: bool
    failures: list[str]

    @property
    def deterministic(self) -> bool:
        return self.ok


def replay(rollout_log: RolloutLog, backend: Backend | None = None) -> SandboxHandle:
    """Reconstruct the environment to the exact pre-rollout state.

    Re-materializes the rollout's schema (same seed) into a fresh namespace and
    returns a live :class:`SandboxHandle` whose DB is identical to the original
    pre-rollout state. Returns the handle for further inspection or teardown.
    """
    backend = backend if backend is not None else SQLiteBackend()
    sandbox = Sandbox(backend)
    return sandbox.spawn(rollout_log.schema, seed=rollout_log.seed, task_id=rollout_log.task_id)


def verify_replay(
    rollout_log: RolloutLog, backend: Backend | None = None
) -> ReplayReport:
    """Re-run the recorded trace and verify every query reproduces its result.

    Spawns a fresh namespace from the logged schema + seed, then replays each
    recorded SQL statement in order and compares the returned result to the
    recorded one (columns + rows). Setup and submit entries are skipped.
    """
    backend = backend if backend is not None else SQLiteBackend()
    handle = replay(rollout_log, backend)
    failures: list[str] = []
    replayed = 0
    matching = 0
    try:
        for entry in rollout_log.trace:
            if entry.kind != "query":
                continue
            replayed += 1
            got = backend.execute(handle.backend_handle, entry.sql)
            if got.status != entry.result.status:
                failures.append(
                    f"step {entry.step}: status {got.status!r} != {entry.result.status!r}"
                )
                continue
            if list(got.columns) != list(entry.result.columns) or got.rows != entry.result.rows:
                failures.append(f"step {entry.step}: result rows/columns differ")
                continue
            matching += 1
    finally:
        Sandbox(backend).teardown(handle)
    return ReplayReport(
        replayed_steps=replayed,
        matching_steps=matching,
        ok=(replayed == matching) and len(failures) == 0,
        failures=failures,
    )
