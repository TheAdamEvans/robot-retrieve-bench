from alloy_server.gen.alloy.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CandidateKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CANDIDATE_KIND_UNSPECIFIED: _ClassVar[CandidateKind]
    WINDOW: _ClassVar[CandidateKind]
    INTERVAL: _ClassVar[CandidateKind]
    MERGED: _ClassVar[CandidateKind]

class Completeness(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPLETENESS_UNSPECIFIED: _ClassVar[Completeness]
    EXHAUSTIVE: _ClassVar[Completeness]
    CANDIDATE_LIMITED: _ClassVar[Completeness]
    COMPLETENESS_UNKNOWN: _ClassVar[Completeness]

class ScoreKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SCORE_KIND_UNSPECIFIED: _ClassVar[ScoreKind]
    BM25: _ClassVar[ScoreKind]
    COSINE: _ClassVar[ScoreKind]
    TRUTH_ORDINAL: _ClassVar[ScoreKind]
    FUSED: _ClassVar[ScoreKind]
    RANK: _ClassVar[ScoreKind]

class FilterReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILTER_REASON_UNSPECIFIED: _ClassVar[FilterReason]
    REQUIRED_CLAUSE_FALSE: _ClassVar[FilterReason]
    TRUNCATED: _ClassVar[FilterReason]
    MERGED_INTO: _ClassVar[FilterReason]

class StageOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STAGE_OUTCOME_UNSPECIFIED: _ClassVar[StageOutcome]
    RAN: _ClassVar[StageOutcome]
    SKIPPED: _ClassVar[StageOutcome]
    UNAVAILABLE: _ClassVar[StageOutcome]
    ABSTAINED: _ClassVar[StageOutcome]

class PipelineRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PIPELINE_ROLE_UNSPECIFIED: _ClassVar[PipelineRole]
    TOP_LEVEL: _ClassVar[PipelineRole]
    GENERATE: _ClassVar[PipelineRole]
    RERANK: _ClassVar[PipelineRole]

class ExecutionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_MODE_UNSPECIFIED: _ClassVar[ExecutionMode]
    SEARCH: _ClassVar[ExecutionMode]
    SCORE_ALL: _ClassVar[ExecutionMode]
CANDIDATE_KIND_UNSPECIFIED: CandidateKind
WINDOW: CandidateKind
INTERVAL: CandidateKind
MERGED: CandidateKind
COMPLETENESS_UNSPECIFIED: Completeness
EXHAUSTIVE: Completeness
CANDIDATE_LIMITED: Completeness
COMPLETENESS_UNKNOWN: Completeness
SCORE_KIND_UNSPECIFIED: ScoreKind
BM25: ScoreKind
COSINE: ScoreKind
TRUTH_ORDINAL: ScoreKind
FUSED: ScoreKind
RANK: ScoreKind
FILTER_REASON_UNSPECIFIED: FilterReason
REQUIRED_CLAUSE_FALSE: FilterReason
TRUNCATED: FilterReason
MERGED_INTO: FilterReason
STAGE_OUTCOME_UNSPECIFIED: StageOutcome
RAN: StageOutcome
SKIPPED: StageOutcome
UNAVAILABLE: StageOutcome
ABSTAINED: StageOutcome
PIPELINE_ROLE_UNSPECIFIED: PipelineRole
TOP_LEVEL: PipelineRole
GENERATE: PipelineRole
RERANK: PipelineRole
EXECUTION_MODE_UNSPECIFIED: ExecutionMode
SEARCH: ExecutionMode
SCORE_ALL: ExecutionMode

class StageScore(_message.Message):
    __slots__ = ("stage_id", "impl", "value", "kind")
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    IMPL_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    stage_id: str
    impl: str
    value: float
    kind: ScoreKind
    def __init__(self, stage_id: _Optional[str] = ..., impl: _Optional[str] = ..., value: _Optional[float] = ..., kind: _Optional[_Union[ScoreKind, str]] = ...) -> None: ...

class StageTouch(_message.Message):
    __slots__ = ("stage_id", "action")
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    stage_id: str
    action: str
    def __init__(self, stage_id: _Optional[str] = ..., action: _Optional[str] = ...) -> None: ...

class ClauseResult(_message.Message):
    __slots__ = ("clause_id", "doc", "required", "truth", "reason", "value", "unit")
    CLAUSE_ID_FIELD_NUMBER: _ClassVar[int]
    DOC_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    TRUTH_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    clause_id: str
    doc: str
    required: bool
    truth: _common_pb2.Truth
    reason: _common_pb2.UnknownReason
    value: float
    unit: str
    def __init__(self, clause_id: _Optional[str] = ..., doc: _Optional[str] = ..., required: _Optional[bool] = ..., truth: _Optional[_Union[_common_pb2.Truth, str]] = ..., reason: _Optional[_Union[_common_pb2.UnknownReason, str]] = ..., value: _Optional[float] = ..., unit: _Optional[str] = ...) -> None: ...

class NamedAnchor(_message.Message):
    __slots__ = ("name", "t_ns", "lo_ns", "hi_ns")
    NAME_FIELD_NUMBER: _ClassVar[int]
    T_NS_FIELD_NUMBER: _ClassVar[int]
    LO_NS_FIELD_NUMBER: _ClassVar[int]
    HI_NS_FIELD_NUMBER: _ClassVar[int]
    name: str
    t_ns: int
    lo_ns: int
    hi_ns: int
    def __init__(self, name: _Optional[str] = ..., t_ns: _Optional[int] = ..., lo_ns: _Optional[int] = ..., hi_ns: _Optional[int] = ...) -> None: ...

class Candidate(_message.Message):
    __slots__ = ("candidate_id", "kind", "recording_id", "seed", "resolved", "lineage", "scores", "clauses", "evidence", "anchors", "completeness", "spatial_basis", "filtered", "window_id", "member_ids")
    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StageScore
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StageScore, _Mapping]] = ...) -> None: ...
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    RECORDING_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_FIELD_NUMBER: _ClassVar[int]
    LINEAGE_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    CLAUSES_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    ANCHORS_FIELD_NUMBER: _ClassVar[int]
    COMPLETENESS_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_BASIS_FIELD_NUMBER: _ClassVar[int]
    FILTERED_FIELD_NUMBER: _ClassVar[int]
    WINDOW_ID_FIELD_NUMBER: _ClassVar[int]
    MEMBER_IDS_FIELD_NUMBER: _ClassVar[int]
    candidate_id: str
    kind: CandidateKind
    recording_id: str
    seed: _common_pb2.Interval
    resolved: _common_pb2.Interval
    lineage: _containers.RepeatedCompositeFieldContainer[StageTouch]
    scores: _containers.MessageMap[str, StageScore]
    clauses: _containers.RepeatedCompositeFieldContainer[ClauseResult]
    evidence: _containers.RepeatedCompositeFieldContainer[_common_pb2.MessageId]
    anchors: _containers.RepeatedCompositeFieldContainer[NamedAnchor]
    completeness: Completeness
    spatial_basis: _common_pb2.SpatialBasis
    filtered: bool
    window_id: str
    member_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, candidate_id: _Optional[str] = ..., kind: _Optional[_Union[CandidateKind, str]] = ..., recording_id: _Optional[str] = ..., seed: _Optional[_Union[_common_pb2.Interval, _Mapping]] = ..., resolved: _Optional[_Union[_common_pb2.Interval, _Mapping]] = ..., lineage: _Optional[_Iterable[_Union[StageTouch, _Mapping]]] = ..., scores: _Optional[_Mapping[str, StageScore]] = ..., clauses: _Optional[_Iterable[_Union[ClauseResult, _Mapping]]] = ..., evidence: _Optional[_Iterable[_Union[_common_pb2.MessageId, _Mapping]]] = ..., anchors: _Optional[_Iterable[_Union[NamedAnchor, _Mapping]]] = ..., completeness: _Optional[_Union[Completeness, str]] = ..., spatial_basis: _Optional[_Union[_common_pb2.SpatialBasis, str]] = ..., filtered: _Optional[bool] = ..., window_id: _Optional[str] = ..., member_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class FilterRecord(_message.Message):
    __slots__ = ("candidate_id", "stage_id", "reason", "detail")
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    candidate_id: str
    stage_id: str
    reason: FilterReason
    detail: str
    def __init__(self, candidate_id: _Optional[str] = ..., stage_id: _Optional[str] = ..., reason: _Optional[_Union[FilterReason, str]] = ..., detail: _Optional[str] = ...) -> None: ...

class CostReport(_message.Message):
    __slots__ = ("bytes_read", "bytes_by_layer", "bytes_served_from_cache", "raw_ratio", "llm_calls", "prompt_tokens", "cached_prompt_tokens", "reasoning_tokens", "completion_tokens", "usd_est", "encoder_forward_ms", "wall_ms")
    class BytesByLayerEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    BYTES_READ_FIELD_NUMBER: _ClassVar[int]
    BYTES_BY_LAYER_FIELD_NUMBER: _ClassVar[int]
    BYTES_SERVED_FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    RAW_RATIO_FIELD_NUMBER: _ClassVar[int]
    LLM_CALLS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    USD_EST_FIELD_NUMBER: _ClassVar[int]
    ENCODER_FORWARD_MS_FIELD_NUMBER: _ClassVar[int]
    WALL_MS_FIELD_NUMBER: _ClassVar[int]
    bytes_read: int
    bytes_by_layer: _containers.ScalarMap[str, int]
    bytes_served_from_cache: int
    raw_ratio: float
    llm_calls: int
    prompt_tokens: int
    cached_prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    usd_est: float
    encoder_forward_ms: float
    wall_ms: float
    def __init__(self, bytes_read: _Optional[int] = ..., bytes_by_layer: _Optional[_Mapping[str, int]] = ..., bytes_served_from_cache: _Optional[int] = ..., raw_ratio: _Optional[float] = ..., llm_calls: _Optional[int] = ..., prompt_tokens: _Optional[int] = ..., cached_prompt_tokens: _Optional[int] = ..., reasoning_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., usd_est: _Optional[float] = ..., encoder_forward_ms: _Optional[float] = ..., wall_ms: _Optional[float] = ...) -> None: ...

class StageReport(_message.Message):
    __slots__ = ("stage_id", "impl", "role", "candidates_in", "candidates_out", "filtered_by_reason", "truncated", "cost", "triggered_by", "children", "outcome", "outcome_detail")
    class FilteredByReasonEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    IMPL_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_IN_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_OUT_FIELD_NUMBER: _ClassVar[int]
    FILTERED_BY_REASON_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    TRIGGERED_BY_FIELD_NUMBER: _ClassVar[int]
    CHILDREN_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_DETAIL_FIELD_NUMBER: _ClassVar[int]
    stage_id: str
    impl: str
    role: PipelineRole
    candidates_in: int
    candidates_out: int
    filtered_by_reason: _containers.ScalarMap[str, int]
    truncated: int
    cost: CostReport
    triggered_by: str
    children: _containers.RepeatedCompositeFieldContainer[StageReport]
    outcome: StageOutcome
    outcome_detail: str
    def __init__(self, stage_id: _Optional[str] = ..., impl: _Optional[str] = ..., role: _Optional[_Union[PipelineRole, str]] = ..., candidates_in: _Optional[int] = ..., candidates_out: _Optional[int] = ..., filtered_by_reason: _Optional[_Mapping[str, int]] = ..., truncated: _Optional[int] = ..., cost: _Optional[_Union[CostReport, _Mapping]] = ..., triggered_by: _Optional[str] = ..., children: _Optional[_Iterable[_Union[StageReport, _Mapping]]] = ..., outcome: _Optional[_Union[StageOutcome, str]] = ..., outcome_detail: _Optional[str] = ...) -> None: ...

class StageSpec(_message.Message):
    __slots__ = ("stage_id", "impl", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STAGE_ID_FIELD_NUMBER: _ClassVar[int]
    IMPL_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    stage_id: str
    impl: str
    params: _containers.ScalarMap[str, str]
    def __init__(self, stage_id: _Optional[str] = ..., impl: _Optional[str] = ..., params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PipelineSpec(_message.Message):
    __slots__ = ("pipeline_id", "role", "generators", "rankers", "final_k", "spec_version", "stub", "doc")
    PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    GENERATORS_FIELD_NUMBER: _ClassVar[int]
    RANKERS_FIELD_NUMBER: _ClassVar[int]
    FINAL_K_FIELD_NUMBER: _ClassVar[int]
    SPEC_VERSION_FIELD_NUMBER: _ClassVar[int]
    STUB_FIELD_NUMBER: _ClassVar[int]
    DOC_FIELD_NUMBER: _ClassVar[int]
    pipeline_id: str
    role: PipelineRole
    generators: _containers.RepeatedCompositeFieldContainer[StageSpec]
    rankers: _containers.RepeatedCompositeFieldContainer[StageSpec]
    final_k: int
    spec_version: str
    stub: bool
    doc: str
    def __init__(self, pipeline_id: _Optional[str] = ..., role: _Optional[_Union[PipelineRole, str]] = ..., generators: _Optional[_Iterable[_Union[StageSpec, _Mapping]]] = ..., rankers: _Optional[_Iterable[_Union[StageSpec, _Mapping]]] = ..., final_k: _Optional[int] = ..., spec_version: _Optional[str] = ..., stub: _Optional[bool] = ..., doc: _Optional[str] = ...) -> None: ...
