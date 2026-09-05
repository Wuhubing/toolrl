# ToolRL — data, environment, and replay layers

Post-train a small model to make reliable multi-turn tool calls (multi-step
SQL-style database tasks). This repository implements the layers that feed a
training pipeline (SFT → DPO → GRPO on veRL): **schema/task generation**
(2.1), the **sandboxed rollout environment** (2.2), the **persistent multi-turn
task environment** (2.3), and **deterministic replay** (2.5). It contains no
training code — that is TR-002's scope, built on top of these interfaces.

## Layout

```
toolrl/
  data/
    schema_generator.py   # schema model, domain-parameterized generator, disjoint split
    task_generator.py     # multi-step NL tasks + computed ground truth
  env/
    backend.py            # Backend ABC + SQLite (test) / Postgres (prod) backends
    sandbox.py            # per-rollout namespace provisioning / teardown / tracing
    task_env.py           # multi-turn step(reset) interface + reward
    tracer.py             # per-statement execution trace + RolloutLog serialization
  replay/
    deterministic_replay.py  # replay(rollout_log) + verify_replay
configs/
  schema_domains.yaml     # e-commerce / logistics / HR domain definitions
tests/                    # full unit suite (no Postgres / Docker required)
```

## The disjoint split (correctness keystone)

The train/test split is **schema-level**: test tasks run against schemas whose
*structure* was never seen in training — not merely different queries over a
seen schema. A schema's identity is its structure (tables, columns, types,
constraints, primary keys, foreign keys); the seed rows are *not* part of the
identity.

`disjoint_split` partitions *structural fingerprints*, so the two sides are
disjoint by construction. `assert_disjoint` then verifies the guarantee at two
levels: the exact fingerprint, and a **near-duplicate** fingerprint that
canonicalizes table/column identifiers (so `customers(id, name)` and
`clients(uid, label)` are recognized as the same structure). Any overlap raises
`ValueError`. See `toolrl/data/schema_generator.py`.

## Isolation unit: namespace-per-rollout

Each rollout gets its own database namespace — `CREATE SCHEMA` in Postgres, a
private `:memory:` SQLite database in the test backend — rather than a full
container. Rationale: `CREATE SCHEMA` is milliseconds vs ~seconds for a fresh
Postgres container (dominating wall-clock at ~5,120 rollouts); a single shared
instance hosts thousands of concurrent namespaces vs Docker's per-container
memory overhead and daemon limits; and Postgres schemas already provide
per-namespace object isolation, which is sufficient because each rollout runs
only its own deterministic SQL (no untrusted code needing kernel isolation).
The accepted tradeoff is shared instance-level resources. See
`toolrl/env/sandbox.py`.

## Backend abstraction

Every module talks to the DB through `Backend`, so the whole stack is
unit-testable with an in-memory fake (`SQLiteBackend`) — **no live Postgres or
Docker required to run the test suite**. Production uses `PostgresBackend`
(namespace-per-rollout on a shared instance provisioned via Docker; `psycopg`
is imported lazily so it is never required at test time). The dialect-neutral
`Schema` is rendered to per-dialect DDL and seed rows by the backend.

## Reward function

The reward is **binary task success only** — `1.0` if the final submitted
answer matches ground truth, `0.0` otherwise. There is **no partial credit**
for intermediate correct sub-steps. This keeps the downstream GRPO objective
aligned to "produce the correct final answer". Changing this is a decision for
TR-002 and must be made here, not hand-waved later.

## Turn limit

`max_turns` bounds the number of tool calls. Using the final turn on a
non-`finish` action (or stepping past the limit) terminates with `done=True`
and `info["failure_reason"] == "turn_limit_exhausted"` — a *defined failure*
(`reward=0`), never a silent truncation.

## Environment interface (veRL-ready)

`TaskEnv` exposes `reset() -> observation` (NL instruction + schema metadata +
tool schema) followed by `step(tool_call) -> (result, done, info)`, with the
task-success reward in `info["reward"]`. This is the shape a veRL rollout
worker needs to drive; the exact veRL wiring (into its reward interface) is
deliberately left to TR-002. Tools: `query` (run one SQL statement) and
`finish` (submit the final answer).

## Determinism & replay

Every rollout logs its initial `schema` + `seed` + full action trace
(SQL + result + timing) in a serializable `RolloutLog`.
`replay(rollout_log) -> SandboxHandle` reconstructs the exact pre-rollout DB
state, and `verify_replay(rollout_log)` re-runs the recorded SQL to confirm
bit-identical results — the tool used to distinguish "the model made a bad
decision" from "the environment behaved non-deterministically".

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q
```

Requires Python 3.11+. Tests use only the stdlib `sqlite3` plus PyYAML and
pytest; no Postgres or Docker is needed.

## Design decisions worth defending

- **Schema-level (not query-level) held-out split** — a much stronger
  no-leakage claim; see the fingerprint logic and its tests.
- **Namespace-per-rollout** isolation and its concurrency ceiling — a systems
  decision, documented in `sandbox.py`.
- **Pure task-success reward** — what GRPO is actually optimizing; no partial
  credit.
- **Determinism via seed + structure** — replay is possible because seed rows
  are generated deterministically and the backend/DDL renderer are
  deterministic.
