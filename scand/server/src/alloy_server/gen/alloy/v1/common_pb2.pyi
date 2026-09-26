from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Truth(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRUTH_UNSPECIFIED: _ClassVar[Truth]
    TRUTH_UNKNOWN: _ClassVar[Truth]
    TRUTH_TRUE: _ClassVar[Truth]
    TRUTH_FALSE: _ClassVar[Truth]

class UnknownReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_REASON_UNSPECIFIED: _ClassVar[UnknownReason]
    NOT_INDEXED: _ClassVar[UnknownReason]
    NOT_APPLICABLE: _ClassVar[UnknownReason]
    OUTSIDE_COVERAGE: _ClassVar[UnknownReason]
    STALE: _ClassVar[UnknownReason]
    FUTURE_MEASUREMENT: _ClassVar[UnknownReason]
    HEADER_MISSING: _ClassVar[UnknownReason]
    CALIBRATION_UNCERTAIN: _ClassVar[UnknownReason]
    NO_BINDING: _ClassVar[UnknownReason]
    UNEXPRESSIBLE: _ClassVar[UnknownReason]

class Boundary(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BOUNDARY_UNSPECIFIED: _ClassVar[Boundary]
    STRICT_BEFORE: _ClassVar[Boundary]
    INCLUSIVE: _ClassVar[Boundary]

class SpatialBasis(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SPATIAL_BASIS_UNSPECIFIED: _ClassVar[SpatialBasis]
    SPATIAL_NONE: _ClassVar[SpatialBasis]
    RECORDED_TF: _ClassVar[SpatialBasis]
    NOMINAL: _ClassVar[SpatialBasis]
    ESTIMATED: _ClassVar[SpatialBasis]
TRUTH_UNSPECIFIED: Truth
TRUTH_UNKNOWN: Truth
TRUTH_TRUE: Truth
TRUTH_FALSE: Truth
UNKNOWN_REASON_UNSPECIFIED: UnknownReason
NOT_INDEXED: UnknownReason
NOT_APPLICABLE: UnknownReason
OUTSIDE_COVERAGE: UnknownReason
STALE: UnknownReason
FUTURE_MEASUREMENT: UnknownReason
HEADER_MISSING: UnknownReason
CALIBRATION_UNCERTAIN: UnknownReason
NO_BINDING: UnknownReason
UNEXPRESSIBLE: UnknownReason
BOUNDARY_UNSPECIFIED: Boundary
STRICT_BEFORE: Boundary
INCLUSIVE: Boundary
SPATIAL_BASIS_UNSPECIFIED: SpatialBasis
SPATIAL_NONE: SpatialBasis
RECORDED_TF: SpatialBasis
NOMINAL: SpatialBasis
ESTIMATED: SpatialBasis

class MessageId(_message.Message):
    __slots__ = ("recording_id", "topic", "topic_ordinal", "payload_sha256_128")
    RECORDING_ID_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    TOPIC_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_SHA256_128_FIELD_NUMBER: _ClassVar[int]
    recording_id: str
    topic: str
    topic_ordinal: int
    payload_sha256_128: bytes
    def __init__(self, recording_id: _Optional[str] = ..., topic: _Optional[str] = ..., topic_ordinal: _Optional[int] = ..., payload_sha256_128: _Optional[bytes] = ...) -> None: ...

class Interval(_message.Message):
    __slots__ = ("start_ns", "end_ns")
    START_NS_FIELD_NUMBER: _ClassVar[int]
    END_NS_FIELD_NUMBER: _ClassVar[int]
    start_ns: int
    end_ns: int
    def __init__(self, start_ns: _Optional[int] = ..., end_ns: _Optional[int] = ...) -> None: ...
