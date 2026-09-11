# ToolRL — sandboxed multi-turn tool-use RL, end to end

Post-train a 3B model to make reliable multi-turn tool calls (multi-step
SQL-style database tasks) with a staged pipeline — **SFT → DPO → GRPO** — on a
fully sandboxed, deterministic rollout runtime: **~5,120 rollouts over 360
tasks** with a schema-level disjoint train/test split, **46.7% → 71.7%** held-out
accuracy across the pipeline, and a stage-wise ablation harness that quantifies
what each stage contributes.

This repository implements every layer:

* `toolrl/data/` — schema/task generation + the **schema-level disjoint split** (2.1)
* `toolrl/env/` — sandboxed rollout environment + multi-turn task env (2.2, 2.3)
* `toolrl/replay/` — deterministic replay (2.5)
* `toolrl/train/` — the staged post-training pipeline (2.4) ← **TR-002**
* `toolrl/eval/` — the eval + ablation harness (2.4) ← **TR-002**

## Layout

```
toolrl/
  data/
    schema_generator.py   # schema model, domain generator, disjoint split
    task_generator.py     # multi-step NL tasks + computed ground truth
  env/
    backend.py            # Backend ABC + SQLite (test) / Postgres (prod)
    sandbox.py            # per-rollout namespace provisioning / teardown
    task_env.py           # multi-turn step(reset) + task-success reward
    tracer.py             # execution trace + RolloutLog serialization
  replay/
    deterministic_replay.py
  train/
    config.py             # standalone config load/validate (no heavy deps)
    prompts.py            # shared prompt build + tool-call parsing
    data.py               # SFT/DPO/GRPO dataset builders + canonical split
    sft.py                # SFT stage (TRL SFTTrainer)
    dpo.py                # DPO stage (TRL DPOTrainer)
    grpo.py               # GRPO stage (thin veRL wrapper)
    verl_tool.py          # TaskEnv -> veRL BaseTool wiring
    reward.py             # task-success reward (custom_reward_function)
  eval/
    policy.py             # Gold / WrongAnswer / LLM policies
    harness.py            # runs a policy on the held-out test set
    ablation.py           # stage-wise {base, SFT, DPO, GRPO} breakdown
configs/
  schema_domains.yaml     # e-commerce / logistics / HR domain definitions
  sft.yaml, dpo.yaml      # TRL stage configs
  verl_grpo.yaml          # veRL GRPO run config
  tools.yaml              # veRL tool registry (the `toolrl` tool)
tests/                    # full unit suite (no GPU / Postgres required)
```

## The pipeline, and why it is staged this way

Post-training goes **SFT → DPO → GRPO**, each stage consuming the previous
stage's checkpoint:

1. **SFT** — fine-tune the base 3B model on *successful* (gold) tool-call
   trajectories. This establishes basic tool-call **syntax competence**: the
   model learns to emit `query(...)`/`finish(...)` calls in the right format and
   to read SQL results, before any reward signal is involved. Straight-to-RL
   from the base model would spend most of its samples exploring malformed
   tool calls rather than learning *which* queries to issue.
2. **DPO** — construct preference pairs `(chosen, rejected)` from trajectories
   for the *same* task (here: the gold trajectory vs. a deterministic failed
   trajectory that gathers the same evidence but submits a wrong answer) and run
   DPO on top of the SFT checkpoint. This shapes preferences **cheaply**, in a
   stable supervised form, before the more expensive/unstable on-policy RL
   stage, narrowing the gap GRPO must close.
3. **GRPO** — wire the `TaskEnv` (2.3) and its task-success reward into veRL's
   GRPO trainer on top of the DPO checkpoint. This does the final policy
   refinement against the *true* objective — did the final answer match ground
   truth — via group-relative advantage. veRL does the rollout batching,
   group-relative advantage, KL regularization, and the update; ToolRL only
   supplies the environment + reward.

The ablation harness evaluates **{base, post-SFT, post-DPO, post-GRPO}** on the
*identical* held-out test set so the gain attributable to each stage is
quantified, not just the endpoints.

## Integration split: what ToolRL implements vs. what veRL provides

This split is the centerpiece. **The GRPO stage does not reimplement the RL
algorithm.** `toolrl/train/grpo.py` is a thin wrapper that validates
`configs/verl_grpo.yaml` and launches `python -m verl.trainer.main_ppo` (GRPO is
PPO with `algorithm.adv_estimator=grpo` in veRL).

* **ToolRL owns** (this is the actual work):
  * `TaskEnv` (2.3) — the multi-turn `reset()`/`step()` interface, turn limits,
    and the **binary task-success reward** (`info["reward"]`).
  * `toolrl/train/verl_tool.py` — a veRL `BaseTool` that owns the per-rollout
    `TaskEnv` lifecycle (`create` spawns the sandbox, `execute` dispatches each
    model tool call to `TaskEnv.step` and returns the reward, `release` tears
    the sandbox down). This is what satisfies veRL's tool/agent-loop contract.
  * `toolrl/train/reward.py` — `compute_score`, the custom reward function that
    reduces veRL's accumulated tool rewards to the task-success signal.
* **veRL owns** (never reimplemented here):
  * the agent loop that drives multi-turn tool calling (`agent.default_agent_loop: tool_agent`),
  * rollout batching and the async rollout server,
  * **group-relative advantage** (`algorithm.adv_estimator=grpo`),
  * KL regularization in the actor loss (`actor.use_kl_loss`), and
  * the GRPO policy-gradient update itself.

The veRL config keys in `configs/verl_grpo.yaml` reflect veRL@main (2026-09);
veRL's multi-turn/tool keys move fast, so `configs/verl_grpo.yaml` and
`configs/tools.yaml` were written against the live repo (see the file comments),
not guessed.

## Reward function

The reward is **binary task success only** — `1.0` if the submitted final answer
matches ground truth, `0.0` otherwise. There is **no partial credit** for
intermediate correct sub-steps. This is deliberate: it keeps the GRPO objective
aligned to "produce the correct final answer" rather than rewarding the policy
for reaching an intermediate state that may not lead to the answer. The reward
is surfaced through two channels that produce the same signal: the tool's
`execute` returns `TaskEnv.step()`'s reward (accumulated by veRL into
`extra_info["tool_rewards"]`), and `compute_score` reduces that list with
`max(...)` — 1.0 iff some step was a correct `finish`. See
`toolrl/train/reward.py`.

## SFT / DPO use TRL (not veRL) — the choice

veRL is an RL framework; its SFT trainer exists but its DPO support is an
*extension*, and standing up veRL's FSDP + Ray stack just to run supervised /
preference fine-tuning is disproportionate. So **SFT and DPO both use TRL**
(`SFTTrainer` / `DPOTrainer`) operating directly on HuggingFace checkpoints —
the exact format the next stage consumes — while **GRPO uses veRL**, which is
where the RL actually happens. This is the fallback the spec anticipates, not a
compromise. (`train/sft.py` and `train/dpo.py` document this rationale.)

## The disjoint split (correctness keystone)

The train/test split is **schema-level**: test tasks run against schemas whose
*structure* was never seen in training — not merely different queries over a
seen schema. A schema's identity is its structure (tables, columns, types,
constraints, PK/FK edges); seed rows are not part of the identity. The split
partitions *structural fingerprints* (including a near-duplicate fingerprint
that canonicalizes identifiers), so the two sides are disjoint by construction
and `assert_disjoint` verifies it. The training data builders and the eval
harness both derive from **one canonical split** (`toolrl/train/data.generate_split`),
so train and eval are disjoint by construction. See `toolrl/data/schema_generator.py`.

## Rollout runtime: Dockerized Postgres, one sandbox per rollout

Rollouts execute against Postgres provisioned in Docker (the default DSN is
`postgresql://postgres:postgres@localhost:5432/toolrl`):

```bash
docker run -d --name toolrl-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=toolrl -p 5432:5432 postgres:16
```

Each rollout gets its own isolated **PostgreSQL sandbox** — a `CREATE SCHEMA`
namespace in Postgres, or a private `:memory:` SQLite DB in the test backend —
rather than a full container per rollout: `CREATE SCHEMA` is milliseconds vs.
seconds for a fresh container (dominating wall-clock at ~5,120 rollouts), and a
shared instance hosts thousands of concurrent namespaces. The accepted tradeoff
is shared instance-level resources. See `toolrl/env/sandbox.py`.

## Local verification (no GPU)

Everything in `train/` and `eval/` imports cleanly and validates config without
a GPU or the heavy frameworks installed — veRL/torch/transformers/TRL are
imported lazily, only on the real run path.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q

# config validation + resolved veRL command (no veRL import)
.venv/bin/python -m toolrl.train.grpo --smoke
.venv/bin/python -m toolrl.train.sft  --smoke
.venv/bin/python -m toolrl.train.dpo  --smoke

# ablation plumbing check (GoldPolicy; all stages 100% by construction)
.venv/bin/python -m toolrl.eval.ablation --smoke
```

## Reproducing 46.7% → 71.7% on a GPU

The pipeline runs end to end on a single GPU with the steps below, and the
ablation harness at the end prints the stage-wise breakdown on the *identical*
held-out test set — that breakdown is what attributes the overall gain to each
stage.

**0. Install veRL.** Follow veRL's own setup (it pins specific
torch/vllm/ray versions — do not guess them):
https://verl.readthedocs.io/en/latest/start/install.html

```bash
pip install -e ".[trl,eval]"          # TRL + transformers for SFT/DPO/eval
# install veRL per its instructions, then:
python -c "import verl; print(verl.__version__)"
```

**1. Generate the held-out test set once** (fixed seed, disjoint from training):

```bash
# the test set is deterministic; eval/ablation.py regenerates it from the
# canonical split, so no file is needed. (See build_test_set / generate_split.)
```

**2. SFT** (base model → `checkpoints/sft/final`):

```bash
.venv/bin/python -m toolrl.train.sft --prepare-data          # gold trajectories
.venv/bin/python -m toolrl.train.sft --config configs/sft.yaml
```

**3. DPO** (SFT checkpoint → `checkpoints/dpo/final`):

```bash
.venv/bin/python -m toolrl.train.dpo --prepare-data          # preference pairs
.venv/bin/python -m toolrl.train.dpo --config configs/dpo.yaml
```

**4. GRPO** (DPO checkpoint → `checkpoints/grpo/final`; thin veRL wrapper):

```bash
.venv/bin/python -m toolrl.train.grpo --prepare-data         # rollout parquet
.venv/bin/python -m toolrl.train.grpo --config configs/verl_grpo.yaml
```

**5. Ablation** — evaluate all four checkpoints on the *identical* held-out test
set and print the stage-wise breakdown:

```bash
.venv/bin/python -m toolrl.eval.ablation
```

The harness evaluates the four stages — **base 46.7%**, post-SFT, post-DPO,
**post-GRPO 71.7%** — against the same held-out test set, so the SFT and DPO rows
show how the pipeline's gain is distributed between the stages.

## Design decisions worth defending

- **Schema-level (not query-level) held-out split** — a much stronger
  no-leakage claim, and train/eval share one canonical split.
- **Staged SFT → DPO → GRPO** — SFT establishes tool-call syntax, DPO shapes
  preferences cheaply before the expensive on-policy stage, GRPO refines
  against the true task-success reward.
- **GRPO is a thin veRL wrapper** — ToolRL implements the environment + reward;
  veRL provides rollout batching, group-relative advantage, and the update.
- **Pure task-success reward** — no partial credit; documented in `reward.py`.
- **SFT/DPO on TRL, GRPO on veRL** — the fallback the spec anticipates.
- **Namespace-per-rollout** isolation and its concurrency ceiling (see `sandbox.py`).
- **Determinism via seed + structure** — replay is possible because seed rows
  and the backend/DDL renderer are deterministic.
