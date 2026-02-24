from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .contracts import (
    ACTOR_NORMALIZER,
    ACTOR_PLANNER,
    ACTOR_POLICY,
    SCHEMA_VERSION,
    ActionProposedEvent,
    ActionProposedPayload,
    Actor,
    IntentAcceptedEvent,
    IntentAcceptedPayload,
    IntentNormalizedEvent,
    IntentNormalizedPayload,
    now_utc,
)
from .module_api import ModuleContext
from .module_loader import ModuleLoadError, load_module_registry


def _sample_request_text(intent: str) -> str:
    if intent == "calculation":
        return "Calculate 12.5 + 7 - 3"
    if intent == "person_search":
        return "Search person John Doe"
    if intent == "company_search":
        return "Search company XPTO debt profile"
    if intent == "document_extraction":
        return "Extract fields from document DOC-1"
    if intent == "classification":
        return "Classify record REC-1 under taxonomy default"
    return f"Sample request for intent {intent}"


def _sample_intent_event(intent: str, aggregate_id: str) -> IntentNormalizedEvent:
    correlation_id = uuid.uuid4()
    return IntentNormalizedEvent(
        event_id=uuid.uuid4(),
        event_type="IntentNormalized",
        schema_version=SCHEMA_VERSION,
        module_id="kernel.ingress",
        timestamp=now_utc(),
        actor=Actor(type="llm", id=ACTOR_NORMALIZER),
        correlation_id=correlation_id,
        causation_id=correlation_id,
        idempotency_key=f"conformance-intent-{intent}-{uuid.uuid4()}",
        aggregate_id=aggregate_id,
        confidence=1.0,
        payload=IntentNormalizedPayload(
            request_text=_sample_request_text(intent),
            intent=intent,
            entities=[],
            requested_outputs=["result"],
            priority="normal",
        ),
    )


def run_module_conformance(
    modules_dir: Path,
    ctx: ModuleContext,
    module_id: str | None = None,
) -> dict[str, Any]:
    try:
        registry = load_module_registry(modules_dir)
    except ModuleLoadError as exc:
        return {"ok": False, "error": str(exc), "modules": []}

    loaded_all = registry.all()
    if module_id:
        loaded = [item for item in loaded_all if item.manifest.module_id == module_id]
        if not loaded:
            return {
                "ok": False,
                "error": f"Module '{module_id}' not found",
                "modules": [],
            }
    else:
        loaded = loaded_all

    allowlisted_actions = {action.id for action in ctx.actions_catalog.actions}

    report: dict[str, Any] = {"ok": True, "modules": []}

    for item in loaded:
        manifest = item.manifest
        plugin = item.plugin
        errors: list[str] = []
        warnings: list[str] = []

        if not manifest.intents:
            warnings.append("manifest.intents is empty")

        if not manifest.actions:
            warnings.append("manifest.actions is empty")

        for action in manifest.actions:
            if action not in allowlisted_actions:
                errors.append(f"manifest action not allowlisted: {action}")

        accepted_checks = 0

        for intent in manifest.intents:
            intent_event = _sample_intent_event(intent, f"conformance-{manifest.module_id}")
            decision = plugin.policy(intent_event, ctx)
            if decision is None:
                errors.append(f"policy returned None for declared intent '{intent}'")
                continue

            if decision.accepted:
                accepted_checks += 1
                accepted_event = IntentAcceptedEvent(
                    event_id=uuid.uuid4(),
                    event_type="IntentAccepted",
                    schema_version=SCHEMA_VERSION,
                    module_id=manifest.module_id,
                    timestamp=now_utc(),
                    actor=Actor(type="service", id=ACTOR_POLICY),
                    correlation_id=intent_event.correlation_id,
                    causation_id=intent_event.event_id,
                    idempotency_key=f"conformance-accept-{uuid.uuid4()}",
                    aggregate_id=intent_event.aggregate_id,
                    payload=IntentAcceptedPayload(
                        intent_event_id=intent_event.event_id,
                        rule_ids=decision.rule_ids,
                        summary=decision.summary or "accepted",
                    ),
                )

                proposal = plugin.plan(accepted_event, intent_event, ctx)
                if proposal is None:
                    errors.append(
                        f"plan returned None after accepted policy for intent '{intent}'"
                    )
                    continue

                if proposal.action_id not in manifest.actions:
                    errors.append(
                        f"plan produced action '{proposal.action_id}' not declared in manifest.actions"
                    )

                proposal_event = ActionProposedEvent(
                    event_id=uuid.uuid4(),
                    event_type="ActionProposed",
                    schema_version=SCHEMA_VERSION,
                    module_id=manifest.module_id,
                    timestamp=now_utc(),
                    actor=Actor(type="service", id=ACTOR_PLANNER),
                    correlation_id=accepted_event.correlation_id,
                    causation_id=accepted_event.event_id,
                    idempotency_key=f"conformance-proposal-{uuid.uuid4()}",
                    aggregate_id=accepted_event.aggregate_id,
                    payload=ActionProposedPayload(
                        accepted_event_id=accepted_event.event_id,
                        action_id=proposal.action_id,
                        args=proposal.args,
                        reason=proposal.reason,
                        preconditions=proposal.preconditions,
                        dry_run=proposal.dry_run,
                    ),
                )

                outcome = plugin.execute(proposal_event, ctx)
                if outcome is None:
                    warnings.append(
                        f"execute returned None for action '{proposal.action_id}'"
                    )
                elif outcome.status not in {"accepted", "rejected", "executed", "failed"}:
                    errors.append(
                        f"execute returned invalid status '{outcome.status}'"
                    )

        if accepted_checks == 0:
            warnings.append("no accepted policy scenario validated")

        module_ok = len(errors) == 0
        if not module_ok:
            report["ok"] = False

        report["modules"].append(
            {
                "module_id": manifest.module_id,
                "ok": module_ok,
                "errors": errors,
                "warnings": warnings,
            }
        )

    return report
