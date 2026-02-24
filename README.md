# JSONLLM Deterministic Event Pipeline

`LLM -> strict JSON event -> deterministic policy -> deterministic planner -> deterministic executor -> append-only JSONL`

The LLM is only used for ingress normalization (`IntentNormalized`).
Every side effect decision is made by deterministic workers.

## Security posture

- The LLM cannot execute actions directly.
- Only trusted actors can emit specific event types (`actor.id` is enforced by `event_type`).
- Execution requires lineage: `IntentAccepted -> ActionProposed -> ActionOutcome`.
- `ActionProposed.action_id` must exist in the allowlist catalog and args must match typed schema.
- Duplicate `idempotency_key` is rejected.

## Stack

- Python
- `pydantic` for strict event typing/validation
- OpenAI Structured Outputs (`responses.parse` / fallback parse API)
- JSONL append-only event log (`data/events.jsonl`)

## Files

- `src/event_pipeline.py`: ingress + deterministic workers
- `catalog/allowed-actions.json`: allowlist of executable capabilities
- `catalog/policy-config.json`: deterministic policy rules
- `catalog/intent-routes.json`: deterministic intent->action routing + arg bindings
- `examples/*.json`: sample events

## Install

```bash
python3 -m pip install -r requirements.txt
```

Set API key for ingress normalization:

```bash
export OPENAI_API_KEY="your_key"
```

## CLI

Validate one event file:

```bash
python3 src/event_pipeline.py validate examples/intent-event.json
```

Ingress (single LLM entrypoint):

```bash
python3 src/event_pipeline.py new-intent \
  --request-text "Procure dados de divida da empresa XPTO" \
  --model gpt-4.1-mini \
  --aggregate-id request-123
```

Policy worker (deterministic):

```bash
python3 src/event_pipeline.py run-policy
```

Planner worker (deterministic):

```bash
python3 src/event_pipeline.py run-planner
```

Executor worker (deterministic handlers):

```bash
python3 src/event_pipeline.py run-executor
```

Dry-run options (no append):

```bash
python3 src/event_pipeline.py run-policy --dry-run
python3 src/event_pipeline.py run-planner --dry-run
python3 src/event_pipeline.py run-executor --dry-run
```

## Event sequence

1. `IntentNormalized` (actor `normalizer.v1`)
2. `IntentAccepted` or `IntentRejected` (actor `policy.v1`)
3. `ActionProposed` (actor `planner.v1`, only for accepted intents)
4. `ActionOutcome` (actor `executor.v1`)

## Add/remove functionalities

- Remove capability: delete route in `catalog/intent-routes.json` and/or action in `catalog/allowed-actions.json`.
- Add capability: add allowlisted action in `catalog/allowed-actions.json`, then add a matching route in `catalog/intent-routes.json`.
- Policy controls whether an intent is executable via `catalog/policy-config.json`.
