"""Prompt construction and tool-call parsing shared by train and eval.

This is the single source of truth for how a task is rendered into the prompt a
policy sees, and how a policy's free-text tool call is parsed back into the
structured ``ToolCall`` the :class:`~toolrl.env.task_env.TaskEnv` expects. Both
the training data builders (``toolrl/train/data.py``) and the eval harness
(``toolrl/eval/harness.py``) use these helpers so the prompt format is identical
at train time and test time.
"""

from __future__ import annotations

import json
import re
from typing import Any

from toolrl.env.task_env import ToolCall

SYSTEM_PROMPT = (
    "You are a database assistant that answers multi-step questions by issuing "
    "tool calls against a private SQL database. You may call 'query' as many "
    "times as needed to inspect the data, then call 'finish' once with your "
    "final answer. Call only 'query' and 'finish'."
)


def render_tables(tables: list[dict[str, Any]]) -> str:
    """Render the observation's ``tables`` field as compact DDL-ish text."""
    lines: list[str] = []
    for t in tables:
        cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
        lines.append(f"TABLE {t['name']}({cols})  PRIMARY KEY {t['primary_key']}")
        for fk in t.get("foreign_keys", []):
            lines.append(
                f"  FK {t['name']}.{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}"
            )
    return "\n".join(lines)


def render_tools(tools: list[dict[str, Any]]) -> str:
    """Render the observation's ``tools`` field as a bulleted tool list."""
    return "\n".join(f"- {t['name']}: {t['description']}" for t in tools)


def build_prompt(
    instruction: str,
    tables: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    """Build the full text prompt for a single task observation."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Database schema:\n{render_tables(tables)}\n\n"
        f"Available tools:\n{render_tools(tools)}\n\n"
        f"Instruction: {instruction}\n"
    )


def build_prompt_from_obs(obs: dict[str, Any]) -> str:
    """Build the prompt directly from a ``TaskEnv.reset()`` observation."""
    return build_prompt(
        instruction=obs["instruction"],
        tables=obs["tables"],
        tools=obs["tools"],
    )


def render_tool_call(tc: ToolCall | dict[str, Any]) -> str:
    """Serialize a tool call to the canonical text form the model is taught."""
    if isinstance(tc, ToolCall):
        name, args = tc.name, tc.arguments
    else:
        name, args = tc.get("name"), tc.get("arguments", {})
    return json.dumps({"name": name, "arguments": args}, sort_keys=True)


def parse_tool_call(text: str) -> ToolCall:
    """Parse a model's free-text tool call into a :class:`ToolCall`.

    Accepts a bare JSON object ``{"name": ..., "arguments": {...}}``, possibly
    wrapped in markdown fences or ``<tool_call>`` tags. Raises ``ValueError`` if
    nothing parseable is found.
    """
    text = (text or "").strip()
    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    tagged = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if tagged:
        candidates.append(tagged.group(1))

    for cand in candidates:
        cand = cand.strip()
        match = re.search(r"\{.*\}", cand, re.DOTALL)
        if not match:
            continue
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "name" in obj:
                return ToolCall(
                    name=str(obj.get("name", "")),
                    arguments=dict(obj.get("arguments") or {}),
                )
        except json.JSONDecodeError:
            continue

    raise ValueError(f"could not parse tool call from: {text!r}")
