# JSONLLM Deterministic Modular Pipeline

`LLM ingress -> strict events -> deterministic policy/planner/executor modules -> append-only JSONL`

`jsonllm` is now a kernel + modules architecture:

- Kernel (`src/event_pipeline.py` + `src/jsonllm_kernel/`) enforces deterministic invariants.
- Modules (`modules/*`) implement domain-specific policy, planning and execution.

## Deterministic guarantees

- LLM only emits `IntentNormalized` (`module_id=kernel.ingress`).
- Side effects are executed only by deterministic workers.
- Event lineage is mandatory: `IntentNormalized -> IntentAccepted/Rejected -> ActionProposed -> ActionOutcome`.
- `ActionProposed.action_id` must be allowlisted.
- `idempotency_key` prevents duplicate processing.
- `actor` authority and `module_id` provenance are validated.

## Layout

- `src/event_pipeline.py`: kernel CLI orchestrator
- `src/jsonllm_kernel/contracts.py`: event and config schemas
- `src/jsonllm_kernel/module_api.py`: module interfaces/contracts
- `src/jsonllm_kernel/module_loader.py`: `module.toml` validation + loader
- `modules/core_builtin/`: built-in generic policy/planner/executor module
- `modules/math_sum/`: example module (`math.sum.v1`) with tangible computation
- `catalog/allowed-actions.json`: global action allowlist
- `catalog/policy-config.json`: global security policy gate
- `catalog/intent-routes.json`: core module routes

## Install

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
```

## CLI

```bash
python3 src/event_pipeline.py list-modules
python3 src/event_pipeline.py validate examples/intent-event.json
python3 src/event_pipeline.py new-intent --request-text "..." --aggregate-id req-1
python3 src/event_pipeline.py run-policy
python3 src/event_pipeline.py run-planner
python3 src/event_pipeline.py run-executor
```

`--dry-run` is available for `run-policy`, `run-planner`, `run-executor`.

## Module system

Every module must provide `module.toml` and an entrypoint class.

Example `module.toml`:

```toml
module_id = "my_module"
name = "My Module"
version = "1.0.0"
priority = 100
enabled = true
intents = ["my_intent"]
actions = ["my.action.v1"]
permissions = ["filesystem:data/outputs"]

[entrypoint]
module_file = "module.py"
class_name = "MyModule"
```

Entrypoint class should implement methods from `BaseModule` in `src/jsonllm_kernel/module_api.py`:

- `policy(intent_event, ctx) -> PolicyDecision | None`
- `plan(accepted_event, intent_event, ctx) -> PlanProposal | None`
- `execute(proposal_event, ctx) -> ExecutionOutcome | None`

Return `None` when the module does not own that event/intent.

## Tangible example

`math_sum` module performs real deterministic computation and writes output files.

```bash
: > data/events.jsonl
python3 src/event_pipeline.py new-intent \
  --request-text "Calculate the sum of 12.5, 7 and -3" \
  --aggregate-id calc-001
python3 src/event_pipeline.py run-policy
python3 src/event_pipeline.py run-planner
python3 src/event_pipeline.py run-executor
ls -la data/outputs
```

You should see a file like `data/outputs/math_sum__math_sum_v1_*.json` with computed `result`.
