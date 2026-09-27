from alloy_server.gen.alloy.v1 import common_pb2 as _common_pb2
from alloy_server.gen.alloy.v1 import query_pb2 as _query_pb2
from alloy_server.gen.alloy.v1 import pipeline_pb2 as _pipeline_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnswerStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANSWER_STATUS_UNSPECIFIED: _ClassVar[AnswerStatus]
    ANSWERED: _ClassVar[AnswerStatus]
    ANSWERED_PARTIAL: _ClassVar[AnswerStatus]
    ANSWERED_UNVERIFIED: _ClassVar[AnswerStatus]
    INSUFFICIENT_EVIDENCE: _ClassVar[AnswerStatus]
    NONE_FOUND_EXHAUSTIVE: _ClassVar[AnswerStatus]

class ProgramState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROGRAM_STATE_UNSPECIFIED: _ClassVar[ProgramState]
    PROVIDED: _ClassVar[ProgramState]
    GENERATED: _ClassVar[ProgramState]
    FAILED: _ClassVar[ProgramState]
    NOT_REQUESTED: _ClassVar[ProgramState]
ANSWER_STATUS_UNSPECIFIED: AnswerStatus
ANSWERED: AnswerStatus
ANSWERED_PARTIAL: AnswerStatus
ANSWERED_UNVERIFIED: AnswerStatus
INSUFFICIENT_EVIDENCE: AnswerStatus
NONE_FOUND_EXHAUSTIVE: AnswerStatus
PROGRAM_STATE_UNSPECIFIED: ProgramState
PROVIDED: ProgramState
GENERATED: ProgramState
FAILED: ProgramState
NOT_REQUESTED: ProgramState

class ProgramDiagnostics(_message.Message):
    __slots__ = ("state", "errors", "attempts", "cache_hit", "model")
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    ATTEMPTS_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    state: ProgramState
    errors: _containers.RepeatedScalarFieldContainer[str]
    attempts: int
    cache_hit: bool
    model: str
    def __init__(self, state: _Optional[_Union[ProgramState, str]] = ..., errors: _Optional[_Iterable[str]] = ..., attempts: _Optional[int] = ..., cache_hit: _Optional[bool] = ..., model: _Optional[str] = ...) -> None: ...

class ReceiptItem(_message.Message):
    __slots__ = ("sensor", "topic", "message", "age_ns", "ok", "reason")
    SENSOR_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    AGE_NS_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    sensor: str
    topic: str
    message: _common_pb2.MessageId
    age_ns: int
    ok: _common_pb2.Truth
    reason: _common_pb2.UnknownReason
    def __init__(self, sensor: _Optional[str] = ..., topic: _Optional[str] = ..., message: _Optional[_Union[_common_pb2.MessageId, _Mapping]] = ..., age_ns: _Optional[int] = ..., ok: _Optional[_Union[_common_pb2.Truth, str]] = ..., reason: _Optional[_Union[_common_pb2.UnknownReason, str]] = ...) -> None: ...

class EvidenceReceipt(_message.Message):
    __slots__ = ("causal_cutoff_ns", "boundary", "max_age_ns", "discovery_mode", "items")
    CAUSAL_CUTOFF_NS_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_NS_FIELD_NUMBER: _ClassVar[int]
    DISCOVERY_MODE_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    causal_cutoff_ns: int
    boundary: _common_pb2.Boundary
    max_age_ns: int
    discovery_mode: str
    items: _containers.RepeatedCompositeFieldContainer[ReceiptItem]
    def __init__(self, causal_cutoff_ns: _Optional[int] = ..., boundary: _Optional[_Union[_common_pb2.Boundary, str]] = ..., max_age_ns: _Optional[int] = ..., discovery_mode: _Optional[str] = ..., items: _Optional[_Iterable[_Union[ReceiptItem, _Mapping]]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("utterance", "program", "pipeline_id", "scope", "availability_config_id", "k", "mode", "window_ids", "presentation")
    UTTERANCE_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_IDS_FIELD_NUMBER: _ClassVar[int]
    PRESENTATION_FIELD_NUMBER: _ClassVar[int]
    utterance: str
    program: _query_pb2.QueryProgram
    pipeline_id: str
    scope: _query_pb2.Scope
    availability_config_id: str
    k: int
    mode: _pipeline_pb2.ExecutionMode
    window_ids: _containers.RepeatedScalarFieldContainer[str]
    presentation: bool
    def __init__(self, utterance: _Optional[str] = ..., program: _Optional[_Union[_query_pb2.QueryProgram, _Mapping]] = ..., pipeline_id: _Optional[str] = ..., scope: _Optional[_Union[_query_pb2.Scope, _Mapping]] = ..., availability_config_id: _Optional[str] = ..., k: _Optional[int] = ..., mode: _Optional[_Union[_pipeline_pb2.ExecutionMode, str]] = ..., window_ids: _Optional[_Iterable[str]] = ..., presentation: _Optional[bool] = ...) -> None: ...

class ResultItem(_message.Message):
    __slots__ = ("candidate", "receipt", "unsupported")
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_FIELD_NUMBER: _ClassVar[int]
    UNSUPPORTED_FIELD_NUMBER: _ClassVar[int]
    candidate: _pipeline_pb2.Candidate
    receipt: EvidenceReceipt
    unsupported: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, candidate: _Optional[_Union[_pipeline_pb2.Candidate, _Mapping]] = ..., receipt: _Optional[_Union[EvidenceReceipt, _Mapping]] = ..., unsupported: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("results", "program_used", "diagnostics", "status", "completeness", "report", "cost", "bundle_id", "notes", "filtered")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_USED_FIELD_NUMBER: _ClassVar[int]
    DIAGNOSTICS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    NOTES_FIELD_NUMBER: _ClassVar[int]
    FILTERED_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[ResultItem]
    program_used: _query_pb2.QueryProgram
    diagnostics: ProgramDiagnostics
    status: AnswerStatus
    completeness: _pipeline_pb2.Completeness
    report: _pipeline_pb2.StageReport
    cost: _pipeline_pb2.CostReport
    bundle_id: str
    notes: _containers.RepeatedScalarFieldContainer[str]
    filtered: _containers.RepeatedCompositeFieldContainer[_pipeline_pb2.FilterRecord]
    def __init__(self, results: _Optional[_Iterable[_Union[ResultItem, _Mapping]]] = ..., program_used: _Optional[_Union[_query_pb2.QueryProgram, _Mapping]] = ..., diagnostics: _Optional[_Union[ProgramDiagnostics, _Mapping]] = ..., status: _Optional[_Union[AnswerStatus, str]] = ..., completeness: _Optional[_Union[_pipeline_pb2.Completeness, str]] = ..., report: _Optional[_Union[_pipeline_pb2.StageReport, _Mapping]] = ..., cost: _Optional[_Union[_pipeline_pb2.CostReport, _Mapping]] = ..., bundle_id: _Optional[str] = ..., notes: _Optional[_Iterable[str]] = ..., filtered: _Optional[_Iterable[_Union[_pipeline_pb2.FilterRecord, _Mapping]]] = ...) -> None: ...
