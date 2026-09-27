from alloy_server.gen.alloy.v1 import common_pb2 as _common_pb2
from alloy_server.gen.alloy.v1 import query_pb2 as _query_pb2
from alloy_server.gen.alloy.v1 import answer_pb2 as _answer_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LabelSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LABEL_SOURCE_UNSPECIFIED: _ClassVar[LabelSource]
    COMPUTED: _ClassVar[LabelSource]
    AGENT_PROVISIONAL: _ClassVar[LabelSource]
    HUMAN: _ClassVar[LabelSource]
    DERIVED: _ClassVar[LabelSource]

class Bucket(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BUCKET_UNSPECIFIED: _ClassVar[Bucket]
    ZERO: _ClassVar[Bucket]
    ONE_TWO: _ClassVar[Bucket]
    THREE_FIVE: _ClassVar[Bucket]
    SIX_PLUS: _ClassVar[Bucket]
    UNSURE: _ClassVar[Bucket]
LABEL_SOURCE_UNSPECIFIED: LabelSource
COMPUTED: LabelSource
AGENT_PROVISIONAL: LabelSource
HUMAN: LabelSource
DERIVED: LabelSource
BUCKET_UNSPECIFIED: Bucket
ZERO: Bucket
ONE_TWO: Bucket
THREE_FIVE: Bucket
SIX_PLUS: Bucket
UNSURE: Bucket

class EvalQuery(_message.Message):
    __slots__ = ("query_id", "intent_group_id", "query_set", "utterance", "scope", "oracle_program", "expected_status", "expect_abstain", "intent")
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    INTENT_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_SET_FIELD_NUMBER: _ClassVar[int]
    UTTERANCE_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    ORACLE_PROGRAM_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_STATUS_FIELD_NUMBER: _ClassVar[int]
    EXPECT_ABSTAIN_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    query_id: str
    intent_group_id: str
    query_set: str
    utterance: str
    scope: _query_pb2.Scope
    oracle_program: _query_pb2.QueryProgram
    expected_status: _answer_pb2.AnswerStatus
    expect_abstain: bool
    intent: str
    def __init__(self, query_id: _Optional[str] = ..., intent_group_id: _Optional[str] = ..., query_set: _Optional[str] = ..., utterance: _Optional[str] = ..., scope: _Optional[_Union[_query_pb2.Scope, _Mapping]] = ..., oracle_program: _Optional[_Union[_query_pb2.QueryProgram, _Mapping]] = ..., expected_status: _Optional[_Union[_answer_pb2.AnswerStatus, str]] = ..., expect_abstain: _Optional[bool] = ..., intent: _Optional[str] = ...) -> None: ...

class Judgment(_message.Message):
    __slots__ = ("intent_group_id", "window_id", "grade", "rationale", "refs", "source", "judge", "job_id")
    INTENT_GROUP_ID_FIELD_NUMBER: _ClassVar[int]
    WINDOW_ID_FIELD_NUMBER: _ClassVar[int]
    GRADE_FIELD_NUMBER: _ClassVar[int]
    RATIONALE_FIELD_NUMBER: _ClassVar[int]
    REFS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    JUDGE_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    intent_group_id: str
    window_id: str
    grade: int
    rationale: str
    refs: _containers.RepeatedCompositeFieldContainer[_common_pb2.MessageId]
    source: LabelSource
    judge: str
    job_id: str
    def __init__(self, intent_group_id: _Optional[str] = ..., window_id: _Optional[str] = ..., grade: _Optional[int] = ..., rationale: _Optional[str] = ..., refs: _Optional[_Iterable[_Union[_common_pb2.MessageId, _Mapping]]] = ..., source: _Optional[_Union[LabelSource, str]] = ..., judge: _Optional[str] = ..., job_id: _Optional[str] = ...) -> None: ...

class WindowAttributes(_message.Message):
    __slots__ = ("segment_id", "recording_id", "span", "persons_in_corridor", "stationary_group", "doorway_traversal", "vehicle_present", "vehicle_interaction", "bicycle", "indoor", "turn_visible", "caption", "refs", "judge", "job_id")
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    RECORDING_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    PERSONS_IN_CORRIDOR_FIELD_NUMBER: _ClassVar[int]
    STATIONARY_GROUP_FIELD_NUMBER: _ClassVar[int]
    DOORWAY_TRAVERSAL_FIELD_NUMBER: _ClassVar[int]
    VEHICLE_PRESENT_FIELD_NUMBER: _ClassVar[int]
    VEHICLE_INTERACTION_FIELD_NUMBER: _ClassVar[int]
    BICYCLE_FIELD_NUMBER: _ClassVar[int]
    INDOOR_FIELD_NUMBER: _ClassVar[int]
    TURN_VISIBLE_FIELD_NUMBER: _ClassVar[int]
    CAPTION_FIELD_NUMBER: _ClassVar[int]
    REFS_FIELD_NUMBER: _ClassVar[int]
    JUDGE_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    segment_id: str
    recording_id: str
    span: _common_pb2.Interval
    persons_in_corridor: Bucket
    stationary_group: _common_pb2.Truth
    doorway_traversal: _common_pb2.Truth
    vehicle_present: _common_pb2.Truth
    vehicle_interaction: _common_pb2.Truth
    bicycle: _common_pb2.Truth
    indoor: _common_pb2.Truth
    turn_visible: _common_pb2.Truth
    caption: str
    refs: _containers.RepeatedCompositeFieldContainer[_common_pb2.MessageId]
    judge: str
    job_id: str
    def __init__(self, segment_id: _Optional[str] = ..., recording_id: _Optional[str] = ..., span: _Optional[_Union[_common_pb2.Interval, _Mapping]] = ..., persons_in_corridor: _Optional[_Union[Bucket, str]] = ..., stationary_group: _Optional[_Union[_common_pb2.Truth, str]] = ..., doorway_traversal: _Optional[_Union[_common_pb2.Truth, str]] = ..., vehicle_present: _Optional[_Union[_common_pb2.Truth, str]] = ..., vehicle_interaction: _Optional[_Union[_common_pb2.Truth, str]] = ..., bicycle: _Optional[_Union[_common_pb2.Truth, str]] = ..., indoor: _Optional[_Union[_common_pb2.Truth, str]] = ..., turn_visible: _Optional[_Union[_common_pb2.Truth, str]] = ..., caption: _Optional[str] = ..., refs: _Optional[_Iterable[_Union[_common_pb2.MessageId, _Mapping]]] = ..., judge: _Optional[str] = ..., job_id: _Optional[str] = ...) -> None: ...
