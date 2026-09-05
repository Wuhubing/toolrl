# ToolRL — Implementation Spec

Same purpose as the ForgeHarness spec: written for handoff to a coding AI, structured so the rebuilt project can actually defend its design choices in an interview, not just reproduce the headline numbers. If your original build differs from what's suggested here, edit this spec first so the code matches what you actually did. Ground truth to stay consistent with: 3B base model, ~5,120 sandboxed rollouts, 360 multi-step database tasks, disjoint train/test schemas, 46.7% → 71.7% test accuracy across the SFT → DPO → GRPO pipeline.

## 1. Goal

Post-train a 3B model to make reliable multi-turn tool calls (specifically multi-step database/SQL-style tasks), using a staged pipeline, with a fully sandboxed and deterministic rollout environment so that measured gains reflect real policy improvement and not environment noise or train/test leakage.

**Foundation: build on veRL, not a from-scratch RL loop.** veRL is the training backbone here — it already provides GRPO (and PPO) implementations, rollout orchestration, and distributed training support, and it's the same framework the published "ToolRL: Reward is All Tool Learning Needs" work is built on, so using it means your training loop follows established, reviewed practice rather than a reimplementation that has to be independently validated for correctness. Your implementation effort should go almost entirely into the environment/reward/schema-generation layers below (2.1–2.3, 2.5), not the policy-gradient machinery itself, which veRL already handles.

Assumed stack: Python, PostgreSQL, Docker, veRL as the RL training framework, with TRL as a fallback for the SFT/DPO stages specifically if veRL's coverage of those is thin. Adjust if different.

## 2. Components

### 2.1 Disjoint train/test schema generation

**What it does:** Generates database schemas (tables, columns, relationships) programmatically, with a hard split so no schema (or near-duplicate schema) appears in both train and test.

Requirements:
- A schema generator that produces varied but structurally-plausible schemas (e.g., parameterized by domain: e-commerce, logistics, HR, etc.)
- A held-out split enforced at the schema level, not the task level (i.e., test tasks use schemas never seen in training, not just different queries on seen schemas)
- Task generation on top of each schema: multi-step natural-language goals requiring 2+ dependent tool calls (e.g., "find the top customer by region, then look up their most recent order")

**Interview angle:** This is the detail that separates a credible RL result from an inflated one. Be ready to explain precisely what "disjoint" means in your setup — schema-level held-out split is a much stronger claim than query-level, and if your original build did query-level, say so accurately rather than letting the spec overstate it.

### 2.2 Sandboxed, isolated rollout environment

**What it does:** Each rollout gets its own PostgreSQL instance (or isolated schema/namespace within one), so concurrent rollouts can't interfere with each other's state, and every rollout starts from a known-clean state.

Requirements:
- `Sandbox.spawn(schema_def) -> SandboxHandle` — provisions an isolated Postgres instance (container-per-rollout, or namespace-per-rollout if you need faster spin-up — pick one and justify it, same tradeoff as ForgeHarness's per-call/per-session decision)
- `Sandbox.teardown(handle)` — guaranteed cleanup even on rollout failure/timeout
- Execution tracing: every SQL statement executed during the rollout, its result, and timing, logged per rollout for later replay/debugging

**Interview angle:** Be ready to state the isolation unit precisely (full container vs schema-level) and the concurrency ceiling that choice implies — this is a systems question, not just an RL question, and interviewers in infra-adjacent roles will probe it.

### 2.3 Persistent multi-turn task environment

**What it does:** Wraps the sandbox with a multi-turn interface: the model issues a tool call, the environment executes it against the sandbox, returns a structured result, and the model continues until it emits a final answer or hits a turn limit.

Requirements:
- `TaskEnv.step(tool_call) -> (result, done, info)` — standard RL env interface
- Turn limit and a defined failure mode when it's hit (counted as a failure, not silently truncated)
- Reward signal: task-level success (did the final answer match ground truth) is the primary signal; consider whether you used any partial credit for intermediate correct sub-steps (worth deciding explicitly, since this affects what GRPO is actually optimizing)

**Interview angle:** Be precise about your reward function. "We only rewarded final task success" vs "we gave partial credit for correct intermediate steps" changes what the RL is actually learning to do, and interviewers will ask this directly.

### 2.4 Staged post-training pipeline

**What it does:** SFT → DPO → GRPO, with each stage's contribution isolated via ablation (evaluate on the same held-out test set after each stage).

**Foundation:** The GRPO stage is where veRL does the heavy lifting — it handles rollout batching, group-relative advantage computation, and the policy-gradient update. Your integration point is implementing 2.3's `TaskEnv.step()` interface to satisfy veRL's environment/rollout-worker contract, and wiring the task-success reward from 2.3 into veRL's reward interface. The SFT and DPO stages can use veRL's trainers directly if it covers them, or fall back to TRL's `SFTTrainer`/`DPOTrainer` on the same checkpoints if veRL's coverage is thin — veRL is primarily RL-focused, so this fallback is expected, not a compromise.

Requirements:
- **SFT stage:** fine-tune the base 3B model on successful rollout trajectories (or human/expert-authored trajectories if available) to establish a reasonable initial policy — via veRL if supported, else TRL's `SFTTrainer`
- **DPO stage:** construct preference pairs from rollouts (e.g., successful vs failed trajectories, or higher-reward vs lower-reward trajectories for the same task) and run DPO on top of the SFT checkpoint — via veRL if supported, else TRL's `DPOTrainer`
- **GRPO stage:** wire your `TaskEnv` (2.3) into veRL's rollout-worker interface and use veRL's GRPO trainer directly on top of the DPO checkpoint, rather than reimplementing group-relative advantage computation
- Ablation harness: evaluate {base model, post-SFT, post-DPO, post-GRPO} on the identical held-out test set, producing the stage-wise breakdown that supports "quantified each stage's contribution"

**Interview angle:** This is the centerpiece of the project. Know cold why you staged it this way rather than going straight to GRPO from the base model (SFT establishes basic tool-call syntax competence; DPO shapes preferences cheaply before the more expensive/unstable RL stage; GRPO does the final policy refinement against the true task-success reward). Also know the actual per-stage numbers if you have them — a bare 46.7% → 71.7% invites "how much of that was SFT alone?" And be ready to explain the veRL integration precisely: what you implemented (environment/reward interface) vs. what veRL provides (rollout orchestration, GRPO update rule) — that split, not the training loop itself, is what's actually yours.

### 2.5 Deterministic environment replay

**What it does:** Given a logged rollout (from 2.2's execution tracing), reconstruct the exact environment state and re-run it to verify reproducibility or debug a specific failure.

Requirements:
- Every rollout's initial schema + seed + full action trace is logged
- `replay(rollout_log) -> SandboxHandle` reconstructs the environment to the exact pre-rollout state
- Used to distinguish "the model made a bad decision" from "the environment behaved non-deterministically" when debugging training instability

**Interview angle:** This is what lets you say the reported gains are trustworthy — be ready to describe a specific instance where replay helped you debug something (even a plausible reconstructed example is more credible than an unsupported claim of "fully reproducible").

## 3. Suggested repo structure

```
toolrl/
  data/
    schema_generator.py
    task_generator.py
  env/
    sandbox.py           # per-rollout Postgres provisioning/teardown
    task_env.py           # multi-turn step() interface
    tracer.py              # execution tracing for replay
  train/
    sft.py                 # veRL trainer if supported, else TRL SFTTrainer
    dpo.py                 # veRL trainer if supported, else TRL DPOTrainer
    grpo.py                 # thin wrapper around veRL's GRPO trainer, wiring in task_env + reward
  eval/
    harness.py             # runs {base, SFT, DPO, GRPO} checkpoints on held-out test set
    ablation.py             # produces the stage-wise breakdown
  replay/
    deterministic_replay.py
  configs/
    schema_domains.yaml     # e.g., e-commerce, logistics, HR domain params
    verl_grpo.yaml           # veRL run config: rollout batch size, group size, KL coefficient, etc.
  README.md
```

Key dependency: `verl` (install per veRL's own setup instructions — it has specific torch/vllm/ray version requirements; check their repo before pinning versions here rather than guessing).

## 4. Build order for the AI executor

0. Install and configure veRL — follow veRL's own environment setup (it has specific torch/vllm/ray version pins; don't guess these) before writing any training code
1. Schema generator + disjoint train/test split logic (get this right first, everything else depends on it)
2. Sandbox provisioning/teardown + task environment `step()` interface, written to satisfy veRL's environment contract
3. Execution tracer + deterministic replay (build this early, it makes debugging every later stage easier)
4. SFT training script (veRL if supported, else TRL) + eval harness against held-out test set
5. DPO stage (preference pair construction + training; veRL if supported, else TRL)
6. GRPO stage: wire `task_env` + reward into veRL's GRPO trainer, don't reimplement the update rule
7. Ablation script producing the full stage-wise breakdown
8. Compare against your reported 46.7% → 71.7% and reconcile any gap by adjusting implementation details, not the numbers

## 5. Open items to confirm before handoff

- Exact base model name/family, if you're comfortable sharing it
- Whether schema-level or query-level held-out split was actually used
- Whether GRPO reward included any partial credit or was pure task-success
- Per-stage accuracy numbers (SFT-only, SFT+DPO), if you have them, to bake into the ablation section
- Isolation unit for rollouts: full container per rollout vs schema-level isolation within a shared instance
- Confirm veRL's current setup requirements (torch/vllm/ray versions) before pinning them in configs — these move fast and the AI executor shouldn't guess at them
