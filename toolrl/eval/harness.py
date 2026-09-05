"""Evaluation harness (SPEC 2.4): run a policy on the held-out test set.

The harness drives a :class:`~toolrl.env.task_env.TaskEnv` for every held-out
test task and scores the rollout with the same binary task-success reward the
GRPO stage optimizes. It is used by ``ablation.py`` to evaluate the same
checkpoint set — {base, post-SFT, post-DPO, post-GRPO} — on the *identical*
held-out test set, producing the stage-wise accuracy breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from toolrl.data.schema_generator import Schema
from toolrl.data.task_generator import Task
from toolrl.env.backend import Backend, SQLiteBackend
from toolrl.env.task_env import TaskEnv, ToolCall
from toolrl.train.data import generate_split

PolicyFactory = Callable[[Task], Any]


@dataclass
class EvalResult:
    """The result of evaluating one checkpoint/stage on the test set."""

    stage: str
    total: int
    correct: int
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "failures": self.failures,
        }


def build_test_set(
    num_tasks: int = 360,
    seed: int = 42,
    domain: str | None = None,
    num_schemas_per_domain: int = 900,
) -> list[tuple[Schema, Task]]:
    """Build the **held-out test set** (disjoint from training at the schema level).

    Delegates to :func:`toolrl.train.data.generate_split` (the single canonical
    train/test partition) and returns its *test* side, so the test tasks are
    disjoint from the SFT/DPO/GRPO training tasks by construction. This is the
    "identical held-out test set" the ablation harness runs every checkpoint
    against.
    """
    _, test_pairs = generate_split(seed=seed, domain=domain, num_schemas_per_domain=num_schemas_per_domain)
    return test_pairs[:num_tasks]


class Harness:
    """Runs a policy over every held-out test task and reports task-success."""

    def __init__(
        self,
        test_tasks: list[tuple[Schema, Task]],
        backend: Backend | None = None,
        max_turns: int = 10,
    ):
        self.test_tasks = test_tasks
        self.backend = backend if backend is not None else SQLiteBackend()
        self.max_turns = max_turns

    def evaluate(self, stage: str, policy_factory: PolicyFactory) -> EvalResult:
        correct = 0
        failures: list[str] = []
        for schema, task in self.test_tasks:
            policy = policy_factory(task)
            success = self._run_one(task, schema, policy)
            if success:
                correct += 1
            else:
                failures.append(task.task_id)
        return EvalResult(
            stage=stage,
            total=len(self.test_tasks),
            correct=correct,
            failures=failures,
        )

    def _run_one(self, task: Task, schema: Schema, policy: Any) -> bool:
        env = TaskEnv(task, schema, backend=self.backend, max_turns=self.max_turns)
        history: list[tuple[ToolCall, dict[str, Any]]] = []
        try:
            obs = env.reset()
            done = False
            while not done:
                tc = policy.act(obs, history)
                result, done, info = env.step(tc)
                history.append((tc, result))
            return bool(info["success"])
        finally:
            env.teardown()

    def close(self) -> None:
        self.backend.close()
