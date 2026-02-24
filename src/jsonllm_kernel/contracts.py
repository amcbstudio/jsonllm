import datetime as dt
import hashlib
import json
import uuid
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 3

ACTOR_NORMALIZER = "normalizer.v1"
ACTOR_POLICY = "policy.v1"
ACTOR_PLANNER = "planner.v1"
ACTOR_EXECUTOR = "executor.v1"


class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["human", "llm", "service"]
    id: str = Field(min_length=1)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    excerpt_hash: str | None = Field(default=None, min_length=1)
    field: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class IntentEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    value: str = Field(min_length=1)


class IntentConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results: int | None = Field(default=None, ge=1, le=100)
    locale: str | None = Field(default=None, min_length=2)


class IntentNormalizedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_text: str = Field(min_length=1)
    intent: Literal[
        "person_search",
        "company_search",
        "document_extraction",
        "classification",
        "calculation",
        "unknown",
    ]
    entities: list[IntentEntity] = Field(default_factory=list)
    constraints: IntentConstraints | None = None
    requested_outputs: list[str] = Field(min_length=1)
    priority: Literal["low", "normal", "high"] = "normal"
    notes: str | None = None

    @field_validator("requested_outputs")
    @classmethod
    def _validate_requested_outputs(cls, value: list[str]) -> list[str]:
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("requested_outputs must contain non-empty strings")
        return value


class IntentAcceptedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_event_id: uuid.UUID
    rule_ids: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)


class IntentRejectedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_event_id: uuid.UUID
    rule_ids: list[str] = Field(min_length=1)
    reasons: list[str] = Field(min_length=1)
    rejection_code: str = Field(min_length=1)


class ActionProposedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_event_id: uuid.UUID | None = None
    action_id: str = Field(min_length=1)
    args: dict[str, Any]
    reason: str = Field(min_length=1)
    dry_run: bool = False
    preconditions: list[str] = Field(default_factory=list)


class ActionOutcomePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_event_id: uuid.UUID
    action_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected", "executed", "failed"]
    executor_id: str = Field(min_length=1)
    details: str | None = None
    error_code: str | None = None
    output_ref: str | None = None


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID
    event_type: str
    schema_version: int = Field(ge=1)
    timestamp: dt.datetime
    module_id: str | None = Field(default=None, min_length=1)
    tenant_id: str | None = Field(default=None, min_length=1)
    actor: Actor
    correlation_id: uuid.UUID
    causation_id: uuid.UUID
    idempotency_key: str = Field(min_length=1)
    aggregate_id: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[Evidence] | None = None


class IntentNormalizedEvent(BaseEvent):
    event_type: Literal["IntentNormalized"]
    payload: IntentNormalizedPayload


class IntentAcceptedEvent(BaseEvent):
    event_type: Literal["IntentAccepted"]
    payload: IntentAcceptedPayload


class IntentRejectedEvent(BaseEvent):
    event_type: Literal["IntentRejected"]
    payload: IntentRejectedPayload


class ActionProposedEvent(BaseEvent):
    event_type: Literal["ActionProposed"]
    payload: ActionProposedPayload


class ActionOutcomeEvent(BaseEvent):
    event_type: Literal["ActionOutcome"]
    payload: ActionOutcomePayload


Event = Annotated[
    Union[
        IntentNormalizedEvent,
        IntentAcceptedEvent,
        IntentRejectedEvent,
        ActionProposedEvent,
        ActionOutcomeEvent,
    ],
    Field(discriminator="event_type"),
]
EVENT_ADAPTER = TypeAdapter(Event)


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    args: dict[str, Literal["string", "integer", "string_array"]]


class AllowedActionsCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    actions: list[ActionSpec] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    allowed_intents: list[str] = Field(default_factory=list)
    blocked_request_terms: list[str] = Field(default_factory=list)
    max_requested_outputs: int = Field(default=10, ge=1, le=100)
    require_route: bool = True


class ArgBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_path: str | None = Field(default=None, alias="from")
    const: Any | None = None
    default: Any | None = None
    required: bool = True

    @model_validator(mode="after")
    def _validate_binding(self) -> "ArgBinding":
        if self.from_path is None and self.const is None:
            raise ValueError("binding must define either 'from' or 'const'")
        return self


class IntentRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    arg_bindings: dict[str, ArgBinding]


class IntentRoutesCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    routes: list[IntentRoute] = Field(default_factory=list)


class OpenAIIntentExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "person_search",
        "company_search",
        "document_extraction",
        "classification",
        "calculation",
        "unknown",
    ]
    entities: list[IntentEntity] = Field(default_factory=list)
    constraints: IntentConstraints | None = None
    requested_outputs: list[str] = Field(min_length=1)
    priority: Literal["low", "normal", "high"] = "normal"
    notes: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1)

    @field_validator("requested_outputs")
    @classmethod
    def _validate_requested_outputs(cls, value: list[str]) -> list[str]:
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("requested_outputs must contain non-empty strings")
        return value


def validation_errors(exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(i) for i in err.get("loc", []))
        msg = err.get("msg", "validation error")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def validate_arg_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str) and bool(value.strip())
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "string_array":
        return isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        )
    return False


def validate_action_against_catalog(
    event: ActionProposedEvent,
    actions_catalog: AllowedActionsCatalog,
) -> list[str]:
    errors: list[str] = []
    actions = {action.id: action for action in actions_catalog.actions}
    spec = actions.get(event.payload.action_id)
    if spec is None:
        return [f"payload.action_id not allowed: {event.payload.action_id}"]

    provided = event.payload.args
    required_keys = set(spec.args.keys())
    provided_keys = set(provided.keys())

    missing = sorted(required_keys - provided_keys)
    if missing:
        errors.append("payload.args missing keys: " + ", ".join(missing))

    unknown = sorted(provided_keys - required_keys)
    if unknown:
        errors.append("payload.args unknown keys: " + ", ".join(unknown))

    for key, expected_type in spec.args.items():
        if key in provided and not validate_arg_type(provided[key], expected_type):
            errors.append(f"payload.args.{key} must be {expected_type}")

    return errors


def parse_event(event_dict: dict[str, Any]) -> tuple[Event | None, list[str]]:
    try:
        event = EVENT_ADAPTER.validate_python(event_dict)
    except ValidationError as exc:
        return None, validation_errors(exc)
    return event, []


def event_json_dict(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json", exclude_none=True)


def make_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_idempotency_key(stage: str, source_event_id: uuid.UUID) -> str:
    return make_hash({"stage": stage, "source_event_id": str(source_event_id)})


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
