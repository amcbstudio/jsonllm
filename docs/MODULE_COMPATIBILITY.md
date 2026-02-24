# Module Compatibility Policy

## Contract version

- Current module API version: `1`.
- Every module manifest must declare `module_api_version = 1`.
- Loader rejects modules with incompatible API version.

## Stability guarantees

- Patch releases may fix bugs without breaking the `BaseModule` interface.
- Minor releases may add optional fields and optional helper utilities.
- Major releases may change module contracts and require manifest version updates.

## Backward compatibility rules

- Kernel keeps event parsing compatibility for older schema versions where practical.
- New safety constraints can be introduced at kernel level if they do not widen privileges.
- Modules must remain deterministic and should not bypass kernel lineage/policy gates.

## Required manifest fields

- `module_id`
- `module_api_version`
- `name`
- `version`
- `priority`
- `enabled`
- `intents`
- `actions`
- `permissions`
- `[entrypoint]` (`module_file`, `class_name`)

## Required behavior

- `policy()` returns `PolicyDecision` for intents the module declares, or `None` for non-owned intents.
- `plan()` and `execute()` must not produce side effects outside declared permissions.
- `execute()` must be deterministic for the same inputs.

## Upgrade path

1. Bump kernel version.
2. Update this compatibility file when contract changes.
3. Run `jsonllm test-module` for each module before deployment.
