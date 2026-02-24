from __future__ import annotations

import json
import re
from pathlib import Path

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
)


class MathSumModule(BaseModule):
    module_id = "math_sum"

    def _numbers(self, expression: str) -> list[float]:
        return [float(match) for match in re.findall(r"[-+]?\d*\.?\d+", expression)]

    def policy(self, intent_event: IntentNormalizedEvent, ctx: ModuleContext) -> PolicyDecision | None:
        if intent_event.payload.intent != "calculation":
            return None

        expression = intent_event.payload.request_text
        numbers = self._numbers(expression)
        if not numbers:
            return PolicyDecision(
                accepted=False,
                rule_ids=["MS-POL-001"],
                reasons=["No numeric terms were found in the calculation request"],
                rejection_code="no_numbers_found",
            )

        if not any(action.id == "math.sum.v1" for action in ctx.actions_catalog.actions):
            return PolicyDecision(
                accepted=False,
                rule_ids=["MS-POL-002"],
                reasons=["Action 'math.sum.v1' is not allowlisted"],
                rejection_code="action_not_allowlisted",
            )

        return PolicyDecision(
            accepted=True,
            rule_ids=["MS-POL-OK"],
            summary="Calculation intent accepted by math_sum module",
        )

    def plan(
        self,
        accepted_event: IntentAcceptedEvent,
        intent_event: IntentNormalizedEvent,
        ctx: ModuleContext,
    ) -> PlanProposal | None:
        if accepted_event.module_id != self.module_id:
            return None

        return PlanProposal(
            action_id="math.sum.v1",
            args={"expression": intent_event.payload.request_text},
            reason="calculation intent accepted by math_sum module",
            preconditions=["intent accepted by policy worker"],
            dry_run=False,
        )

    def execute(self, proposal_event: ActionProposedEvent, ctx: ModuleContext) -> ExecutionOutcome | None:
        if proposal_event.module_id != self.module_id:
            return None

        if proposal_event.payload.action_id != "math.sum.v1":
            return None

        expression = proposal_event.payload.args.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ExecutionOutcome(status="failed", error_code="invalid_expression")

        numbers = self._numbers(expression)
        if not numbers:
            return ExecutionOutcome(status="failed", error_code="no_numbers_found")

        result = sum(numbers)
        digest = make_hash({"action_id": "math.sum.v1", "args": proposal_event.payload.args})[:16]
        output_ref = self._write_output(ctx.outputs_path, digest, expression, numbers, result)

        return ExecutionOutcome(
            status="executed",
            details=f"deterministic sum completed ({len(numbers)} terms)",
            output_ref=output_ref,
        )

    def _write_output(
        self,
        outputs_path: Path,
        digest: str,
        expression: str,
        numbers: list[float],
        result: float,
    ) -> str:
        outputs_path.mkdir(parents=True, exist_ok=True)
        out_path = outputs_path / f"{self.module_id}__math_sum_v1_{digest}.json"
        payload = {
            "module_id": self.module_id,
            "action_id": "math.sum.v1",
            "status": "executed",
            "expression": expression,
            "numbers": numbers,
            "result": result,
            "digest": digest,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return str(out_path)
