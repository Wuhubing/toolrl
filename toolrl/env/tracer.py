"""Execution tracing (SPEC 2.2) and rollout logging (SPEC 2.5).

Every SQL statement executed during a rollout — its result and its timing — is
recorded in a :class:`TraceEntry`. Together with the initial schema + seed,
the trace forms a :class:`RolloutLog` that fully captures a rollout and can be
serialized to JSON for deterministic replay (see ``replay/deterministic_replay.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from toolrl.data.schema_generator import Schema
from toolrl.env.backend import SQLResult


@dataclass
class TraceEntry:
    """One recorded SQL execution (or the final submit action)."""

    step: int                 # 0 == setup, else 1-based tool-call step index
    kind: str                 # "setup" | "query" | "submit"
    sql: str                  # SQL text ("" for submit)
    result: SQLResult
    duration_ms: float
    timestamp: float          # time.monotonic() at execution start

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "kind": self.kind,
            "sql": self.sql,
            "result": self.result.to_dict(),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceEntry":
        return cls(
            step=int(d["step"]),
            kind=d["kind"],
            sql=d.get("sql", ""),
            result=SQLResult.from_dict(d["result"]),
            duration_ms=float(d["duration_ms"]),
            timestamp=float(d["timestamp"]),
        )


@dataclass
class RolloutLog:
    """The complete, serializable record of a single rollout.

    ``schema`` + ``seed`` are the initial state (enough to reconstruct the exact
    pre-rollout DB state), and ``trace`` is the full action trace.
    """

    schema: Schema
    seed: int
    namespace: str
    task_id: str
    trace: list[TraceEntry] = field(default_factory=list)
    final_answer: Any = None
    reward: float = 0.0
    success: bool = False
    done: bool = False
    failure_reason: str | None = None
    turns_used: int = 0
    max_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "seed": self.seed,
            "namespace": self.namespace,
            "task_id": self.task_id,
            "trace": [e.to_dict() for e in self.trace],
            "final_answer": self.final_answer,
            "reward": self.reward,
            "success": self.success,
            "done": self.done,
            "failure_reason": self.failure_reason,
            "turns_used": self.turns_used,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RolloutLog":
        return cls(
            schema=Schema.from_dict(d["schema"]),
            seed=int(d["seed"]),
            namespace=d["namespace"],
            task_id=d["task_id"],
            trace=[TraceEntry.from_dict(e) for e in d.get("trace", [])],
            final_answer=d.get("final_answer"),
            reward=float(d.get("reward", 0.0)),
            success=bool(d.get("success", False)),
            done=bool(d.get("done", False)),
            failure_reason=d.get("failure_reason"),
            turns_used=int(d.get("turns_used", 0)),
            max_turns=int(d.get("max_turns", 0)),
        )


class Tracer:
    """Records timed SQL executions into a :class:`RolloutLog`."""

    def __init__(self, rollout_log: RolloutLog):
        self.log = rollout_log
        self._step = 0

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def record(
        self,
        kind: str,
        sql: str,
        result: SQLResult,
        duration_ms: float,
        step: int | None = None,
    ) -> TraceEntry:
        entry = TraceEntry(
            step=step if step is not None else self._step,
            kind=kind,
            sql=sql,
            result=result,
            duration_ms=duration_ms,
            timestamp=time.monotonic(),
        )
        self.log.trace.append(entry)
        return entry
