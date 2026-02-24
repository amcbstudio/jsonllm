# JSONLLM Deterministic Event Pipeline (Pydantic + OpenAI Structured Outputs)

This repository implements the pattern:

`LLM -> strict JSON event -> deterministic validation -> append-only JSONL`

The model never executes side effects. It only emits structured events. Deterministic workers validate and decide execution.

Write-path policy: only one CLI entrance can write to the event log.

## Stack

- `pydantic` for strict event validation.
- `openai` structured outputs for intent normalization (`new-intent`).
- JSONL append-only storage (`data/events.jsonl`).

## Event types

- `IntentNormalized`
- `ActionProposed`
- `ActionOutcome`

Every event includes:

- `event_id`
- `event_type`
- `schema_version`
- `timestamp`
- `actor`
- `correlation_id`
- `causation_id`
- `idempotency_key`
- `aggregate_id`
- `payload`

## Files

- `src/event_pipeline.py`: Pydantic models, OpenAI normalization, deterministic append pipeline.
- `catalog/allowed-actions.json`: allowlist catalog (`action_id` + typed args).
- `examples/*.json`: sample valid events.
- `requirements.txt`: Python dependencies.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY="your_key"
```

## CLI

Validate one event file:

```bash
python3 src/event_pipeline.py validate examples/intent-event.json
```

Single write entrance: create `IntentNormalized` from natural language using OpenAI structured outputs:

```bash
python3 src/event_pipeline.py new-intent \
  --request-text "Procure pessoas ligadas a XPTO" \
  --model gpt-4.1-mini \
  --aggregate-id request-456
```

Print generated event without appending:

```bash
python3 src/event_pipeline.py new-intent \
  --request-text "Classifique o documento A" \
  --print-only
```

## Deterministic guardrails

- Event validation is done by Pydantic discriminated unions.
- `ActionProposed.action_id` must exist in the allowlist catalog.
- Action args are type-checked (`string`, `integer`, `string_array`).
- Duplicate `idempotency_key` is rejected on append.
- External/manual append is disabled in the CLI to keep one deterministic ingress path.
