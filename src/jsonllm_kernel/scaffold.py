from __future__ import annotations

from pathlib import Path

from .module_api import MODULE_API_VERSION


def _class_name(module_id: str) -> str:
    parts = [part for part in module_id.replace("-", "_").split("_") if part]
    if not parts:
        return "CustomModule"
    return "".join(part.capitalize() for part in parts) + "Module"


def _module_toml(module_id: str, class_name: str) -> str:
    return (
        f'module_id = "{module_id}"\n'
        f"module_api_version = {MODULE_API_VERSION}\n"
        f'name = "{module_id} module"\n'
        'version = "0.1.0"\n'
        "priority = 100\n"
        "enabled = true\n"
        "intents = []\n"
        "actions = []\n"
        "permissions = []\n\n"
        "[entrypoint]\n"
        'module_file = "module.py"\n'
        f'class_name = "{class_name}"\n'
    )


def _module_py(module_id: str, class_name: str) -> str:
    return f'''from __future__ import annotations

from jsonllm_kernel.contracts import ActionProposedEvent, IntentAcceptedEvent, IntentNormalizedEvent
from jsonllm_kernel.module_api import BaseModule, ExecutionOutcome, ModuleContext, PlanProposal, PolicyDecision


class {class_name}(BaseModule):
    module_id = "{module_id}"

    def policy(self, intent_event: IntentNormalizedEvent, ctx: ModuleContext) -> PolicyDecision | None:
        # Return None when this module does not own the intent.
        return None

    def plan(
        self,
        accepted_event: IntentAcceptedEvent,
        intent_event: IntentNormalizedEvent,
        ctx: ModuleContext,
    ) -> PlanProposal | None:
        # Return None when there is no proposal for this accepted event.
        return None

    def execute(self, proposal_event: ActionProposedEvent, ctx: ModuleContext) -> ExecutionOutcome | None:
        # Return None when this module does not execute this action.
        return None
'''


def _conformance_smoke(module_id: str) -> str:
    return (
        "# Basic conformance checklist\n"
        "# 1. Update module.toml intents/actions/permissions\n"
        "# 2. Implement policy/plan/execute in module.py\n"
        "# 3. Run: python3 src/event_pipeline.py test-module --module-id "
        f"{module_id}\n"
    )


def init_module_scaffold(
    modules_dir: Path,
    module_id: str,
    force: bool = False,
) -> dict[str, str]:
    module_root = modules_dir / module_id
    if module_root.exists() and not force:
        raise RuntimeError(f"Module directory already exists: {module_root}")

    module_root.mkdir(parents=True, exist_ok=True)

    class_name = _class_name(module_id)
    files = {
        "module.toml": _module_toml(module_id, class_name),
        "module.py": _module_py(module_id, class_name),
        "CONFORMANCE.md": _conformance_smoke(module_id),
    }

    written: dict[str, str] = {}
    for name, content in files.items():
        path = module_root / name
        if path.exists() and not force:
            raise RuntimeError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")
        written[name] = str(path)

    return written
