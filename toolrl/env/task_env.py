"""Persistent multi-turn task environment (SPEC 2.3).

``TaskEnv`` wraps a :class:`~toolrl.env.sandbox.Sandbox` in a standard RL env
interface: ``step(tool_call) -> (result, done, info)``. The model issues tool
calls (``query`` to run SQL against the sandbox, ``finish`` to submit a final
answer) until it submits, or until the turn limit is hit.

veRL rollout-worker contract
----------------------------
This env is deliberately shaped to be drivable by veRL's rollout worker: a
``reset() -> observation`` (the NL instruction + schema metadata the policy
prompt is built from) followed by repeated ``step(tool_call)`` calls returning
the ``(result, done, info)`` triple, with the task-success reward surfaced in
``info["reward"]``. We intentionally do *not* guess veRL's exact internal API —
only the multi-turn step/reset shape is provided, which is the only thing a
rollout loop needs. Wiring ``info["reward"]`` into veRL's reward interface is
TR-002's job.

Reward function
---------------
The reward is **binary task success only** — ``1.0`` if the submitted final
answer matches ground truth, ``0.0`` otherwise. There is **no partial credit**
for intermediate correct sub-steps. This is deliberate: it keeps the GRPO
objective aligned to "produce the correct final answer" rather than rewarding
the policy for reaching an intermediate state that may not lead to the answer.
(If TR-002 decides otherwise, that is a change to this module, not something to
hand-wave later.)

Turn limit
----------
``max_turns`` is the maximum number of tool calls the model may make. Using the
final turn on a non-``finish`` action — or attempting to step past the limit —
terminates the rollout with ``done=True`` and ``info["failure_reason"] ==
"turn_limit_exhausted"``. Exhaustion is a *defined failure* (``success=False``,
``reward=0``), never a silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolrl.data.schema_generator import Schema
from toolrl.data.task_generator import Task
from toolrl.env.backend import Backend, SQLResult, SQLiteBackend
from toolrl.env.sandbox import Sandbox, SandboxHandle
from toolrl.env.tracer import RolloutLog


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def parse_tool_call(tool_call: ToolCall | dict[str, Any]) -> ToolCall:
    if isinstance(tool_call, ToolCall):
        return tool_call
    if isinstance(tool_call, dict):
        return ToolCall(
            name=str(tool_call.get("name", "")),
            arguments=dict(tool_call.get("arguments") or {}),
        )
    raise TypeError(f"tool_call must be a ToolCall or dict, got {type(tool_call)!r}")


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return str(value).strip().lower()


def answer_matches(submitted: Any, ground_truth: Any) -> bool:
    return normalize_answer(submitted) == normalize_answer(ground_truth)


_EXHAUSTED_REASON = "turn_limit_exhausted"


class TaskEnv:
    """A multi-turn environment over a single task + schema."""

    def __init__(
        self,
        task: Task,
        schema: Schema,
        backend: Backend | None = None,
        max_turns: int = 10,
        sandbox: Sandbox | None = None,
    ):
        self.task = task
        self.schema = schema
        self.backend = backend if backend is not None else SQLiteBackend()
        self.sandbox = sandbox if sandbox is not None else Sandbox(self.backend)
        self.max_turns = max_turns

        self.handle: SandboxHandle | None = None
        self.turns_used = 0
        self.done = False
        self.reward = 0.0
        self.success = False
        self.failure_reason: str | None = None
        self.final_answer: Any = None

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> dict[str, Any]:
        self.handle = self.sandbox.spawn(self.schema, task_id=self.task.task_id)
        self.turns_used = 0
        self.done = False
        self.reward = 0.0
        self.success = False
        self.failure_reason = None
        self.final_answer = None
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        return {
            "instruction": self.task.instruction,
            "schema_name": self.schema.name,
            "tables": [
                {
                    "name": t.name,
                    "columns": [
                        {"name": c.name, "type": c.type} for c in t.columns
                    ],
                    "primary_key": t.primary_key,
                    "foreign_keys": [fk.to_dict() for fk in t.foreign_keys],
                }
                for t in self.schema.tables
            ],
            "tools": [
                {
                    "name": "query",
                    "description": "Execute one SQL statement against the sandbox "
                    "database and return its result rows.",
                    "parameters": {"sql": "string"},
                },
                {
                    "name": "finish",
                    "description": "Submit the final answer to the task.",
                    "parameters": {"answer": "string | number"},
                },
            ],
        }

    # -- step interface ----------------------------------------------------- #

    def step(self, tool_call: ToolCall | dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        if self.handle is None:
            raise RuntimeError("reset() must be called before step()")
        if self.done:
            return self._last_result(), True, self._info()

        if self.turns_used >= self.max_turns:
            self._terminate_failure(_EXHAUSTED_REASON)
            return {
                "tool": "error",
                "error": "turn limit exhausted; no further tool calls allowed",
            }, True, self._info()

        tc = parse_tool_call(tool_call)
        self.turns_used += 1

        if tc.name == "finish":
            return self._finish(tc)
        if tc.name == "query":
            return self._query(tc)

        # Unknown tool: count it as a wasted turn.
        result = {"tool": "error", "error": f"unknown tool {tc.name!r}"}
        if self.turns_used >= self.max_turns:
            self._terminate_failure(_EXHAUSTED_REASON)
            return result, True, self._info()
        return result, False, self._info()

    def _finish(self, tc: ToolCall) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        answer = tc.arguments.get("answer")
        self.final_answer = answer
        self.success = answer_matches(answer, self.task.ground_truth)
        self.reward = 1.0 if self.success else 0.0
        self.done = True

        self.handle.tracer.record(
            kind="submit",
            sql="",
            result=_empty_result(),
            duration_ms=0.0,
            step=self.handle.tracer.next_step(),
        )
        self._finalize_log()
        return {"tool": "finish", "answer": answer, "correct": self.success}, True, self._info()

    def _query(self, tc: ToolCall) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        sql = str(tc.arguments.get("sql", ""))
        res = self.sandbox.execute(self.handle, sql)
        result = {
            "tool": "query",
            "columns": res.columns,
            "rows": [list(r) for r in res.rows],
            "rowcount": res.rowcount,
            "status": res.status,
            "error": res.error,
        }
        if self.turns_used >= self.max_turns:
            self._terminate_failure(_EXHAUSTED_REASON)
            return result, True, self._info()
        return result, False, self._info()

    # -- helpers ------------------------------------------------------------ #

    def _terminate_failure(self, reason: str) -> None:
        self.done = True
        self.success = False
        self.reward = 0.0
        self.failure_reason = reason
        self._finalize_log()

    def _finalize_log(self) -> None:
        log = self.handle.log
        log.final_answer = self.final_answer
        log.reward = self.reward
        log.success = self.success
        log.done = self.done
        log.failure_reason = self.failure_reason
        log.turns_used = self.turns_used
        log.max_turns = self.max_turns

    def _info(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "success": self.success,
            "turns_used": self.turns_used,
            "max_turns": self.max_turns,
            "failure_reason": self.failure_reason,
        }

    def _last_result(self) -> dict[str, Any]:
        return {
            "tool": "finish",
            "answer": self.final_answer,
            "correct": self.success,
        }

    def rollout_log(self) -> RolloutLog:
        assert self.handle is not None
        return self.handle.log

    def teardown(self) -> None:
        if self.handle is not None:
            self.sandbox.teardown(self.handle)
            self.handle = None

    def __enter__(self) -> "TaskEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.teardown()


def _empty_result() -> SQLResult:
    return SQLResult(status="ok", rowcount=0)
