"""Evaluation harness (SPEC 2.4 ablation): policy abstractions.

A :class:`Policy` turns an environment observation (and the turn history) into
the next :class:`~toolrl.env.task_env.ToolCall`. Three implementations ship:

* :class:`GoldPolicy` — the oracle: replays ``task.gold_sql`` then submits the
  ground-truth answer. Always correct; used to verify harness plumbing and to
  generate SFT/DPO trajectories without a model.
* :class:`WrongAnswerPolicy` — the oracle's failure mode: replays the gold
  queries then submits a wrong answer; used to generate DPO rejected pairs and
  to sanity-check that a bad policy scores 0.
* :class:`LLMPolicy` — loads a HuggingFace checkpoint (base / post-SFT /
  post-DPO / post-GRPO) and generates tool calls from the prompt. This is the
  only policy that needs a GPU; ``transformers``/``torch`` are imported lazily.
"""

from __future__ import annotations

from typing import Any

from toolrl.data.task_generator import Task
from toolrl.env.task_env import ToolCall
from toolrl.train.prompts import build_prompt_from_obs, parse_tool_call


class Policy:
    """Abstract policy: map an observation + history to the next tool call."""

    def act(self, obs: dict[str, Any], history: list[tuple[ToolCall, dict[str, Any]]]) -> ToolCall:
        raise NotImplementedError


class GoldPolicy(Policy):
    """Oracle policy bound to a specific task (uses gold SQL + ground truth)."""

    def __init__(self, task: Task):
        self.task = task
        self._queries_issued = 0

    def act(self, obs: dict[str, Any], history: list[tuple[ToolCall, dict[str, Any]]]) -> ToolCall:
        if self._queries_issued < len(self.task.gold_sql):
            sql = self.task.gold_sql[self._queries_issued]
            self._queries_issued += 1
            return ToolCall(name="query", arguments={"sql": sql})
        return ToolCall(name="finish", arguments={"answer": self.task.ground_truth})


class WrongAnswerPolicy(Policy):
    """A deterministic failure policy: gold queries, then a wrong final answer."""

    def __init__(self, task: Task):
        self.task = task
        self._queries_issued = 0

    def act(self, obs: dict[str, Any], history: list[tuple[ToolCall, dict[str, Any]]]) -> ToolCall:
        if self._queries_issued < len(self.task.gold_sql):
            sql = self.task.gold_sql[self._queries_issued]
            self._queries_issued += 1
            return ToolCall(name="query", arguments={"sql": sql})
        return ToolCall(name="finish", arguments={"answer": "0"})


class LLMPolicy(Policy):
    """A real model policy driven by a HuggingFace checkpoint.

    Builds the prompt from the observation (identical to the training prompt),
    generates a response, and parses it back into a :class:`ToolCall`. Loads
    ``transformers``/``torch`` lazily so importing this module needs no GPU.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto", max_new_tokens: int = 256):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
        )
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def act(self, obs: dict[str, Any], history: list[tuple[ToolCall, dict[str, Any]]]) -> ToolCall:
        self._load()
        import torch

        prompt = build_prompt_from_obs(obs)
        if history:
            transcript = "\n".join(
                f"assistant: {tc.name}({tc.arguments})\nresult: {r}" for tc, r in history
            )
            prompt = f"{prompt}\n\nConversation so far:\n{transcript}\n\nNext tool call:"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_tool_call(text)
