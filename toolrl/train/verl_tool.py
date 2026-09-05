"""veRL tool integration: wrap ``TaskEnv`` (SPEC 2.3) as a veRL ``BaseTool``.

This module is the *only* place ToolRL touches veRL's runtime API. Everything
else (``config.py``, ``grpo.py``) validates config or builds a command string
without importing veRL. The split matters for the interview:

* **What ToolRL implements here** — the :class:`ToolRLTool`, which owns the
  per-rollout :class:`~toolrl.env.task_env.TaskEnv` lifecycle (spawn sandbox on
  ``create``, tear it down on ``release``), dispatches each model tool call to
  ``TaskEnv.step``, and surfaces the task-success reward from ``info["reward"]``.
* **What veRL provides** — the agent loop that drives multi-turn tool calling,
  rollout batching, group-relative advantage computation, and the GRPO update.
  None of that is reimplemented here.

The veRL classes are imported lazily. When veRL is not installed (e.g. a
CPU-only smoke check), this module still imports, but instantiating the tool
raises an informative ``ImportError``.
"""

from __future__ import annotations

import json
from typing import Any

from toolrl.data.schema_generator import Schema
from toolrl.data.task_generator import Task
from toolrl.env.backend import Backend, PostgresBackend, SQLiteBackend
from toolrl.env.sandbox import Sandbox
from toolrl.env.task_env import TaskEnv, ToolCall

# Lazy veRL imports. We bind the real classes when veRL is present, and fall
# back to plain ``object`` so the module (and this class) still import cleanly
# on a machine without veRL.
try:  # pragma: no cover - depends on optional veRL install
    from verl.tools.base_tool import BaseTool as _BaseTool
    from verl.tools.schemas import ToolResponse as _ToolResponse

    _HAS_VERL = True
except Exception:  # noqa: BLE001 - veRL is an optional, heavyweight dependency
    _BaseTool = object  # type: ignore[assignment,misc]
    _ToolResponse = None  # type: ignore[assignment]
    _HAS_VERL = False


def _require_verl() -> None:
    if not _HAS_VERL:
        raise ImportError(
            "veRL is required to run the GRPO stage. Install it per veRL's own "
            "setup instructions (it pins specific torch/vllm/ray versions — do "
            "not guess): see verl.readthedocs.io/en/latest/start/install.html. "
            "The ToolRL --smoke path does not need veRL."
        )


def _backend_from_config(config: dict[str, Any]) -> Backend:
    """Build the DB backend from the tool config (defaults to SQLite)."""
    backend_type = (config or {}).get("backend", "sqlite")
    if backend_type == "postgres":
        dsn = (config or {}).get("dsn")
        return PostgresBackend(dsn=dsn) if dsn else PostgresBackend()
    return SQLiteBackend()


class ToolRLTool(_BaseTool):  # type: ignore[valid-type,misc]
    """A veRL ``BaseTool`` that drives a single multi-turn ``TaskEnv``.

    One tool instance is registered per rollout (``instance_id``). The tool's
    OpenAI schema is a single function, ``toolrl``, with an ``action`` field
    (``query`` or ``finish``) plus the action-specific arguments, matching the
    two logical tools exposed by :class:`~toolrl.env.task_env.TaskEnv`. A single
    function (rather than two) keeps the per-rollout ``TaskEnv`` state in one
    place across ``create`` / ``execute`` / ``release``.
    """

    _ACTIVE: dict[str, TaskEnv] = {}

    def __init__(self, config: dict[str, Any], tool_schema: Any = None):
        _require_verl()
        super().__init__(config, tool_schema)
        self.config = config or {}
        self.max_turns = int(self.config.get("max_turns", 10))
        self.backend = _backend_from_config(self.config)

    # -- lifecycle --------------------------------------------------------- #

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, Any]:
        """Spawn the sandbox and register a live ``TaskEnv`` for this rollout.

        ``kwargs`` carries the per-sample ``create_kwargs`` (serialized
        ``schema`` + ``task``) that the dataset attached to the sample.
        """
        _require_verl()
        if instance_id is None:
            raise ValueError("ToolRLTool.create requires an instance_id")

        schema_dict = kwargs.get("schema")
        task_dict = kwargs.get("task")
        if schema_dict is None or task_dict is None:
            raise ValueError(
                "ToolRLTool.create requires 'schema' and 'task' in create_kwargs; "
                "the GRPO dataset must embed the serialized schema + task per sample"
            )

        schema = Schema.from_dict(schema_dict) if isinstance(schema_dict, dict) else schema_dict
        task = Task(
            task_id=task_dict["task_id"],
            schema_name=task_dict["schema_name"],
            domain=task_dict["domain"],
            instruction=task_dict["instruction"],
            ground_truth=task_dict["ground_truth"],
            min_steps=int(task_dict.get("min_steps", 2)),
            seed=int(task_dict.get("seed", 0)),
            gold_sql=list(task_dict.get("gold_sql", [])),
            metadata=dict(task_dict.get("metadata", {})),
        )

        env = TaskEnv(
            task=task,
            schema=schema,
            backend=self.backend,
            max_turns=self.max_turns,
            sandbox=Sandbox(self.backend),
        )
        env.reset()
        self._ACTIVE[instance_id] = env
        return instance_id, _ToolResponse(text=json.dumps({"status": "ready"}))

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[Any, float, dict]:
        """Dispatch one model tool call to ``TaskEnv.step`` and return the reward."""
        _require_verl()
        env = self._ACTIVE.get(instance_id)
        if env is None:
            return (
                _ToolResponse(text=json.dumps({"error": f"no env for instance {instance_id!r}"})),
                0.0,
                {"success": False},
            )

        action = parameters.get("action", "query")
        if action == "query":
            tc = ToolCall(name="query", arguments={"sql": str(parameters.get("sql", ""))})
        elif action == "finish":
            tc = ToolCall(name="finish", arguments={"answer": parameters.get("answer")})
        else:
            return (
                _ToolResponse(text=json.dumps({"error": f"unknown action {action!r}"})),
                0.0,
                {"success": False},
            )

        result, _, info = env.step(tc)
        reward = float(info["reward"])
        metrics = {
            "success": bool(info["success"]),
            "turns_used": int(info["turns_used"]),
            "failure_reason": info["failure_reason"],
        }
        return _ToolResponse(text=json.dumps(result)), reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Return the final task-success reward accumulated by this rollout."""
        _require_verl()
        env = self._ACTIVE.get(instance_id)
        return float(env.reward) if env is not None else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Tear down the sandbox for this rollout (guaranteed cleanup)."""
        env = self._ACTIVE.pop(instance_id, None)
        if env is not None:
            env.teardown()


def openai_tool_schema() -> dict[str, Any]:
    """The OpenAI function-tool schema for the single ``toolrl`` tool.

    Provided as a plain dict so ``configs/tools.yaml`` can embed it directly;
    veRL parses it into its internal ``OpenAIFunctionToolSchema`` at startup,
    avoiding any dependency on veRL's schema classes in ToolRL code.
    """
    return {
        "type": "function",
        "function": {
            "name": "toolrl",
            "description": (
                "Interact with the sandbox SQL database. Use action='query' to "
                "run one SQL statement and action='finish' to submit the final "
                "answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "finish"],
                        "description": "Which action to perform.",
                    },
                    "sql": {
                        "type": "string",
                        "description": "SQL to execute (only when action='query').",
                    },
                    "answer": {
                        "type": ["string", "number", "null"],
                        "description": "Final answer to submit (only when action='finish').",
                    },
                },
                "required": ["action"],
            },
        },
    }
