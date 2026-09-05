"""Dataset builders for the three stages (SPEC 2.4).

Each builder is deterministic (given a seed) and reuses the TR-001 layers
(schema/task generation + the task environment) so the training data is the
*same* environment contract the GRPO stage rolls out against — no format drift.

* ``build_sft_examples`` — successful (gold) trajectories as ``prompt`` +
  ``completion`` for TRL ``SFTTrainer``.
* ``build_dpo_examples`` — preference pairs ``(prompt, chosen, rejected)`` where
  the chosen response is the gold trajectory and the rejected response is a
  deterministic *failed* trajectory (same queries, wrong final answer), for TRL
  ``DPOTrainer``.
* ``build_grpo_rows`` — the veRL rollout dataset: one parquet row per task
  carrying ``prompt``, ``agent_name="tool_agent"``, ``ground_truth``,
  ``data_source``, and the serialized ``schema`` + ``task`` in ``tools_kwargs``
  so ``ToolRLTool.create`` can reconstruct the environment.

No heavy frameworks are imported here; the parquet writer imports pandas lazily
(only when actually writing files, i.e. on the GPU/data-prep path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolrl.data.schema_generator import Schema, SchemaGenerator, disjoint_split
from toolrl.data.task_generator import Task, TaskGenerator
from toolrl.env.backend import SQLiteBackend
from toolrl.env.task_env import TaskEnv
from toolrl.train.prompts import build_prompt, render_tool_call

TOOL_AGENT_NAME = "tool_agent"
DATA_SOURCE = "toolrl"

# Ground-truth task counts (SPEC): ~5,120 training rollouts and 360 held-out
# test tasks. The canonical schema pool below is sized so the disjoint 80/20
# split yields at least these counts on each side.
DEFAULT_NUM_TRAIN_TASKS = 5120
DEFAULT_NUM_TEST_TASKS = 360
CANONICAL_SCHEMAS_PER_DOMAIN = 900


def _schema_tables(schema: Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "columns": [{"name": c.name, "type": c.type} for c in t.columns],
            "primary_key": t.primary_key,
            "foreign_keys": [fk.to_dict() for fk in t.foreign_keys],
        }
        for t in schema.tables
    ]


_TOOLS = [
    {
        "name": "query",
        "description": "Execute one SQL statement against the sandbox database.",
        "parameters": {"sql": "string"},
    },
    {
        "name": "finish",
        "description": "Submit the final answer to the task.",
        "parameters": {"answer": "string | number"},
    },
]


def prompt_for(task: Task, schema: Schema) -> str:
    """The shared prompt for a task (identical at train and test time)."""
    return build_prompt(task.instruction, _schema_tables(schema), _TOOLS)


def gold_trajectory(task: Task, schema: Schema) -> list[dict[str, str]]:
    """The expert (gold) tool-call transcript for a task, as chat messages.

    Runs the gold SQL against a *real* materialization of the schema so the
    tool results embedded in the transcript are exactly what the environment
    would return (no hand-written results).
    """
    backend = SQLiteBackend()
    env = TaskEnv(task, schema, backend=backend)
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt_for(task, schema)}]
    try:
        env.reset()
        for sql in task.gold_sql:
            call = {"name": "query", "arguments": {"sql": sql}}
            result, _, _ = env.step(call)
            messages.append({"role": "assistant", "content": render_tool_call(call)})
            messages.append({"role": "tool", "content": _result_text(result)})
        finish = {"name": "finish", "arguments": {"answer": task.ground_truth}}
        messages.append({"role": "assistant", "content": render_tool_call(finish)})
    finally:
        env.teardown()
    return messages


def failed_trajectory(task: Task, schema: Schema) -> list[dict[str, str]]:
    """A deterministic *failed* trajectory: same gold queries, wrong final answer."""
    backend = SQLiteBackend()
    env = TaskEnv(task, schema, backend=backend)
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt_for(task, schema)}]
    try:
        env.reset()
        for sql in task.gold_sql:
            call = {"name": "query", "arguments": {"sql": sql}}
            result, _, _ = env.step(call)
            messages.append({"role": "assistant", "content": render_tool_call(call)})
            messages.append({"role": "tool", "content": _result_text(result)})
        wrong = {"name": "finish", "arguments": {"answer": "0"}}
        messages.append({"role": "assistant", "content": render_tool_call(wrong)})
    finally:
        env.teardown()
    return messages


def _result_text(result: dict[str, Any]) -> str:
    if result.get("status") == "error":
        return f"error: {result.get('error')}"
    rows = result.get("rows", [])
    return f"rows={rows}"


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """Flatten a chat transcript to a single text completion (assistant+turns).

    Only the assistant tool calls and the tool results are included after the
    system prompt + user prompt; the completion the SFT stage learns is exactly
    the sequence of tool calls, which is what establishes tool-call syntax.
    """
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


def generate_split(
    seed: int = 42,
    domain: str | None = None,
    num_schemas_per_domain: int = CANONICAL_SCHEMAS_PER_DOMAIN,
) -> tuple[list[tuple[Schema, Task]], list[tuple[Schema, Task]]]:
    """Generate the disjoint train/test ``(schema, task)`` split **once**.

    This is the single source of the train/test partition used by *both* the
    training data builders (SFT/DPO/GRPO) and the eval harness. It generates one
    canonical schema pool (deterministic in ``seed``), splits it at the schema
    level with :func:`~toolrl.data.schema_generator.disjoint_split`, then builds
    one task per schema. Because both train and eval call this function with the
    same seed, the train side and test side are disjoint *by construction* — the
    eval harness is never graded on a schema structure seen during training.

    Returns ``(train_pairs, test_pairs)``.
    """
    generator = SchemaGenerator()
    domains = [domain] if domain else generator.domain_names
    schemas: list[Schema] = []
    for d in domains:
        schemas.extend(generator.generate_many(d, num_schemas_per_domain, seed=seed))

    train_schemas, test_schemas = disjoint_split(schemas, test_ratio=0.2, seed=seed)

    task_generator = TaskGenerator()
    train = [(s, task_generator.generate_tasks(s, n=1, seed=s.seed)[0]) for s in train_schemas]
    test = [(s, task_generator.generate_tasks(s, n=1, seed=s.seed)[0]) for s in test_schemas]
    return train, test


def _train_pairs(num_tasks: int, seed: int, domain: str | None) -> list[tuple[Schema, Task]]:
    train, _ = generate_split(seed=seed, domain=domain)
    return train[:num_tasks]


def build_sft_examples(
    num_tasks: int = 2048,
    seed: int = 42,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Build SFT examples: ``{"prompt": str, "completion": str}``."""
    examples: list[dict[str, Any]] = []
    for schema, task in _train_pairs(num_tasks, seed, domain):
        messages = gold_trajectory(task, schema)
        examples.append({"prompt": messages[0]["content"], "completion": _messages_to_text(messages[1:])})
    return examples


def build_dpo_examples(
    num_tasks: int = 2048,
    seed: int = 42,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Build DPO preference pairs: ``{"prompt", "chosen", "rejected"}``."""
    examples: list[dict[str, Any]] = []
    for schema, task in _train_pairs(num_tasks, seed, domain):
        prompt = prompt_for(task, schema)
        chosen_msgs = gold_trajectory(task, schema)
        rejected_msgs = failed_trajectory(task, schema)
        examples.append(
            {
                "prompt": prompt,
                "chosen": _messages_to_text(chosen_msgs[1:]),
                "rejected": _messages_to_text(rejected_msgs[1:]),
            }
        )
    return examples


def build_grpo_rows(
    num_tasks: int = 2048,
    seed: int = 42,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Build veRL GRPO rollout rows (one per task), ready for parquet writing."""
    rows: list[dict[str, Any]] = []
    for schema, task in _train_pairs(num_tasks, seed, domain):
        rows.append(
            {
                "prompt": prompt_for(task, schema),
                "agent_name": TOOL_AGENT_NAME,
                "ground_truth": str(task.ground_truth),
                "data_source": DATA_SOURCE,
                "extra_info": {
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "toolrl": {
                            "create_kwargs": {
                                "schema": schema.to_dict(),
                                "task": {
                                    "task_id": task.task_id,
                                    "schema_name": task.schema_name,
                                    "domain": task.domain,
                                    "instruction": task.instruction,
                                    "ground_truth": task.ground_truth,
                                    "min_steps": task.min_steps,
                                    "seed": task.seed,
                                    "gold_sql": task.gold_sql,
                                    "metadata": task.metadata,
                                },
                            }
                        }
                    },
                },
            }
        )
    return rows


def write_grpo_parquet(rows: list[dict[str, Any]], path: str | Path) -> str:
    """Write GRPO rollout rows to parquet (pandas imported lazily)."""
    import pandas as pd  # lazy: only needed on the data-prep path

    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)
