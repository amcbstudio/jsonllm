# JSONLLM Kernel

Deterministic event-sourced kernel for modular LLM automation.

`LLM ingress -> strict events -> deterministic module policy/planner/executor -> append-only JSONL`

## What this is

- A kernel that enforces deterministic invariants (schema, lineage, provenance, idempotency).
- A module system where third parties implement domain logic.
- A self-serve workflow to scaffold and validate modules without maintainer support.

## Core guarantees

- LLM only emits `IntentNormalized` (`module_id=kernel.ingress`).
- Side effects run only in deterministic workers.
- Mandatory lineage: `IntentNormalized -> IntentAccepted/Rejected -> ActionProposed -> ActionOutcome`.
- `ActionProposed.action_id` must be allowlisted.
- `actor` authority and `module_id` provenance are validated.

## Repository layout

- `src/event_pipeline.py`: CLI orchestrator
- `src/jsonllm_kernel/contracts.py`: event/config contracts
- `src/jsonllm_kernel/module_api.py`: module SDK (`BaseModule`, types)
- `src/jsonllm_kernel/module_loader.py`: `module.toml` loader/validator
- `src/jsonllm_kernel/scaffold.py`: module scaffolding templates
- `src/jsonllm_kernel/conformance.py`: module conformance suite
- `modules/`: example modules
- `catalog/`: allowlist + security configs

## Install

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
```

Or install as package/CLI:

```bash
python3 -m pip install -e .
jsonllm --help
```

## CLI

```bash
python3 src/event_pipeline.py list-modules
python3 src/event_pipeline.py init-module my_domain
python3 src/event_pipeline.py test-module --module-id my_domain
python3 src/event_pipeline.py new-intent --request-text "..." --aggregate-id req-1
python3 src/event_pipeline.py run-policy
python3 src/event_pipeline.py run-planner
python3 src/event_pipeline.py run-executor
```

## Module self-serve workflow

1. Scaffold module:

```bash
python3 src/event_pipeline.py init-module my_domain
```

2. Implement `policy/plan/execute` in `modules/my_domain/module.py`.
3. Update `modules/my_domain/module.toml` (intents/actions/permissions).
4. Ensure actions are in `catalog/allowed-actions.json`.
5. Run conformance:

```bash
python3 src/event_pipeline.py test-module --module-id my_domain
```

6. Run end-to-end pipeline.

## Compatibility and standards

- Module API version is declared by `module_api_version` in `module.toml`.
- Loader rejects incompatible module API versions.
- See:
  - `docs/MODULE_COMPATIBILITY.md`
  - `docs/MODULE_COOKBOOK.md`

## Example modules

- `modules/core_builtin`: generic routes for search/extraction/classification.
- `modules/math_sum`: tangible deterministic computation (`math.sum.v1`) writing outputs to `data/outputs/`.
