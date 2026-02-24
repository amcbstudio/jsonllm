from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    ActionProposedEvent,
    ActionSpec,
    AllowedActionsCatalog,
    IntentAcceptedEvent,
    IntentNormalizedEvent,
    IntentNormalizedPayload,
    IntentRoute,
    IntentRoutesCatalog,
    PolicyConfig,
    validate_arg_type,
)


@dataclass(frozen=True)
class PolicyDecision:
    accepted: bool
    rule_ids: list[str]
    reasons: list[str] = field(default_factory=list)
    summary: str = ""
    rejection_code: str = "policy_denied"


@dataclass(frozen=True)
class PlanProposal:
    action_id: str
    args: dict[str, Any]
    reason: str
    preconditions: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass(frozen=True)
class ExecutionOutcome:
    status: str
    details: str | None = None
    error_code: str | None = None
    output_ref: str | None = None


@dataclass(frozen=True)
class ModuleContext:
    root: Path
    outputs_path: Path
    actions_catalog: AllowedActionsCatalog
    policy_config: PolicyConfig
    routes_catalog: IntentRoutesCatalog


class ModulePlugin(Protocol):
    module_id: str

    def policy(self, intent_event: IntentNormalizedEvent, ctx: ModuleContext) -> PolicyDecision | None:
        ...

    def plan(
        self,
        accepted_event: IntentAcceptedEvent,
        intent_event: IntentNormalizedEvent,
        ctx: ModuleContext,
    ) -> PlanProposal | None:
        ...

    def execute(self, proposal_event: ActionProposedEvent, ctx: ModuleContext) -> ExecutionOutcome | None:
        ...


class BaseModule:
    module_id = "base"

    def policy(self, intent_event: IntentNormalizedEvent, ctx: ModuleContext) -> PolicyDecision | None:
        return None

    def plan(
        self,
        accepted_event: IntentAcceptedEvent,
        intent_event: IntentNormalizedEvent,
        ctx: ModuleContext,
    ) -> PlanProposal | None:
        return None

    def execute(self, proposal_event: ActionProposedEvent, ctx: ModuleContext) -> ExecutionOutcome | None:
        return None


def route_map(routes_catalog: IntentRoutesCatalog) -> dict[str, IntentRoute]:
    return {route.intent: route for route in routes_catalog.routes}


def resolve_path(source: dict[str, Any], path: str) -> Any | None:
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_binding_value(payload: IntentNormalizedPayload, binding_from: str | None, default: Any | None, const: Any | None) -> Any | None:
    if const is not None:
        return const

    if binding_from is None:
        return None

    if binding_from.startswith("entity:"):
        entity_type = binding_from.split(":", 1)[1]
        for entity in payload.entities:
            if entity.type == entity_type:
                return entity.value
        return default

    payload_dict = payload.model_dump(mode="json", exclude_none=False)
    value = resolve_path(payload_dict, binding_from)
    if value is None:
        return default
    return value


def build_route_args(payload: IntentNormalizedPayload, route: IntentRoute) -> tuple[dict[str, Any], list[str]]:
    args: dict[str, Any] = {}
    errors: list[str] = []

    for arg_name, binding in route.arg_bindings.items():
        value = resolve_binding_value(payload, binding.from_path, binding.default, binding.const)
        if value is None and binding.required:
            errors.append(
                f"arg '{arg_name}' could not be resolved from binding '{binding.from_path}'"
            )
            continue
        args[arg_name] = value

    return args, errors


def validate_args_with_spec(action_spec: ActionSpec, args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_keys = set(action_spec.args.keys())
    provided_keys = set(args.keys())

    missing = sorted(required_keys - provided_keys)
    if missing:
        errors.append("missing args: " + ", ".join(missing))

    unknown = sorted(provided_keys - required_keys)
    if unknown:
        errors.append("unknown args: " + ", ".join(unknown))

    for key, arg_type in action_spec.args.items():
        if key in args and not validate_arg_type(args[key], arg_type):
            errors.append(f"invalid arg type for '{key}', expected {arg_type}")

    return errors
