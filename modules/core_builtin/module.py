from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonllm_kernel.contracts import (
    ActionProposedEvent,
    IntentAcceptedEvent,
    IntentNormalizedEvent,
    make_hash,
)
from jsonllm_kernel.module_api import (
    BaseModule,
    ExecutionOutcome,
    ModuleContext,
    PlanProposal,
    PolicyDecision,
    build_route_args,
    route_map,
    validate_args_with_spec,
)


class CoreBuiltinModule(BaseModule):
    module_id = "core_builtin"

    def _action_spec(self, action_id: str, ctx: ModuleContext):
        return {item.id: item for item in ctx.actions_catalog.actions}.get(action_id)

    def policy(self, intent_event: IntentNormalizedEvent, ctx: ModuleContext) -> PolicyDecision | None:
        intent = intent_event.payload.intent
        if intent not in ctx.policy_config.allowed_intents:
            return None

        routes = route_map(ctx.routes_catalog)
        route = routes.get(intent)
        if route is None:
            return PolicyDecision(
                accepted=False,
                rule_ids=["CB-POL-001"],
                reasons=[f"No deterministic route configured for intent '{intent}'"],
                rejection_code="no_route",
            )

        action_spec = self._action_spec(route.action_id, ctx)
        if action_spec is None:
            return PolicyDecision(
                accepted=False,
                rule_ids=["CB-POL-002"],
                reasons=[f"Route action_id '{route.action_id}' is not allowlisted"],
                rejection_code="action_not_allowlisted",
            )

        args, arg_errors = build_route_args(intent_event.payload, route)
        if arg_errors:
            return PolicyDecision(
                accepted=False,
                rule_ids=["CB-POL-003"],
                reasons=arg_errors,
                rejection_code="invalid_bindings",
            )

        arg_type_errors = validate_args_with_spec(action_spec, args)
        if arg_type_errors:
            return PolicyDecision(
                accepted=False,
                rule_ids=["CB-POL-004"],
                reasons=arg_type_errors,
                rejection_code="invalid_args",
            )

        return PolicyDecision(
            accepted=True,
            rule_ids=["CB-POL-OK"],
            summary="Intent accepted by core built-in policy",
        )

    def plan(
        self,
        accepted_event: IntentAcceptedEvent,
        intent_event: IntentNormalizedEvent,
        ctx: ModuleContext,
    ) -> PlanProposal | None:
        if accepted_event.module_id != self.module_id:
            return None

        route = route_map(ctx.routes_catalog).get(intent_event.payload.intent)
        if route is None:
            return None

        args, arg_errors = build_route_args(intent_event.payload, route)
        if arg_errors:
            return None

        action_spec = self._action_spec(route.action_id, ctx)
        if action_spec is None:
            return None

        if validate_args_with_spec(action_spec, args):
            return None

        return PlanProposal(
            action_id=route.action_id,
            args=args,
            reason=route.reason,
            preconditions=["intent accepted by policy worker"],
            dry_run=False,
        )

    def execute(self, proposal_event: ActionProposedEvent, ctx: ModuleContext) -> ExecutionOutcome | None:
        if proposal_event.module_id != self.module_id:
            return None

        action_id = proposal_event.payload.action_id
        args_dict = proposal_event.payload.args
        digest = make_hash({"action_id": action_id, "args": args_dict})[:16]

        supported = {
            "person.search.v1",
            "company.search.v1",
            "document.extract.v1",
            "record.classify.v1",
        }
        if action_id not in supported:
            return None

        output_ref = self._write_output(ctx.outputs_path, action_id, digest, args_dict)

        if action_id in {"person.search.v1", "company.search.v1"}:
            max_results = args_dict.get("max_results", 10)
            details = f"deterministic search completed (max_results={max_results})"
            return ExecutionOutcome(status="executed", details=details, output_ref=output_ref)

        if action_id == "document.extract.v1":
            fields_count = len(args_dict.get("fields", []))
            details = f"deterministic extraction completed (fields={fields_count})"
            return ExecutionOutcome(status="executed", details=details, output_ref=output_ref)

        taxonomy = args_dict.get("taxonomy", "default")
        details = f"deterministic classification completed (taxonomy={taxonomy})"
        return ExecutionOutcome(status="executed", details=details, output_ref=output_ref)

    def _write_output(self, outputs_path: Path, action_id: str, digest: str, args_dict: dict[str, Any]) -> str:
        outputs_path.mkdir(parents=True, exist_ok=True)
        out_path = outputs_path / f"{self.module_id}__{action_id.replace('.', '_')}_{digest}.json"
        payload = {
            "module_id": self.module_id,
            "action_id": action_id,
            "status": "executed",
            "args": args_dict,
            "digest": digest,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return str(out_path)
