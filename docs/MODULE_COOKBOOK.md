# Module Cookbook

## Quick start

1. Scaffold module:

```bash
python3 src/event_pipeline.py init-module my_domain
```

2. Edit `modules/my_domain/module.toml`:
- Fill intents/actions/permissions.
- Keep `module_api_version = 1`.

3. Implement policy/planner/executor in `modules/my_domain/module.py`.

4. Validate module:

```bash
python3 src/event_pipeline.py test-module --module-id my_domain
```

## Design guidelines

- Keep policy strict and deterministic.
- Return `None` when module does not own the event.
- Keep action args typed and allowlisted.
- Avoid broad permissions.
- Write deterministic outputs with explicit references.

## Common patterns

- Intent-specific module:
  - One intent family.
  - One or few tightly scoped actions.

- Domain bundle module:
  - Multiple related intents.
  - Shared policy logic.

## Anti-patterns

- Dynamic shell execution from module logic.
- Using unvalidated free-text as command.
- Producing events that skip lineage transitions.
