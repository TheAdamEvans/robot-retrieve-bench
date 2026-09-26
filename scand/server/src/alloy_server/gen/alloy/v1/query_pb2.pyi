from alloy_server.gen.alloy.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Unit(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNIT_UNSPECIFIED: _ClassVar[Unit]
    DIMENSIONLESS: _ClassVar[Unit]
    RATIO: _ClassVar[Unit]
    PERCENT: _ClassVar[Unit]
    M: _ClassVar[Unit]
    CM: _ClassVar[Unit]
    MPS: _ClassVar[Unit]
    KMH: _ClassVar[Unit]
    MPS2: _ClassVar[Unit]
    DEG: _ClassVar[Unit]
    DEG_PER_S: _ClassVar[Unit]
    S: _ClassVar[Unit]
    MS: _ClassVar[Unit]

class Comparator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPARATOR_UNSPECIFIED: _ClassVar[Comparator]
    GTE: _ClassVar[Comparator]
    GT: _ClassVar[Comparator]
    LTE: _ClassVar[Comparator]
    LT: _ClassVar[Comparator]
    EQ: _ClassVar[Comparator]

class EventKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVENT_KIND_UNSPECIFIED: _ClassVar[EventKind]
    THRESHOLD: _ClassVar[EventKind]
    CHANGE: _ClassVar[EventKind]
    ONSET: _ClassVar[EventKind]
    EXTREMUM: _ClassVar[EventKind]
    RETURN_TO: _ClassVar[EventKind]
    TRACK_APPEAR: _ClassVar[EventKind]
    TRACK_DISAPPEAR: _ClassVar[EventKind]

class Direction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DIRECTION_UNSPECIFIED: _ClassVar[Direction]
    DOWN: _ClassVar[Direction]
    UP: _ClassVar[Direction]

class AnchorPoint(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANCHOR_POINT_UNSPECIFIED: _ClassVar[AnchorPoint]
    START: _ClassVar[AnchorPoint]
    END: _ClassVar[AnchorPoint]
    EXTREMUM_POINT: _ClassVar[AnchorPoint]

class RelationKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RELATION_KIND_UNSPECIFIED: _ClassVar[RelationKind]
    AFTER: _ClassVar[RelationKind]
    BEFORE: _ClassVar[RelationKind]
    WITHIN: _ClassVar[RelationKind]
    DURING: _ClassVar[RelationKind]

class Quantifier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUANTIFIER_UNSPECIFIED: _ClassVar[Quantifier]
    ALL: _ClassVar[Quantifier]
    FIRST: _ClassVar[Quantifier]
    ARGMIN: _ClassVar[Quantifier]
    ARGMAX: _ClassVar[Quantifier]
UNIT_UNSPECIFIED: Unit
DIMENSIONLESS: Unit
RATIO: Unit
PERCENT: Unit
M: Unit
CM: Unit
MPS: Unit
KMH: Unit
MPS2: Unit
DEG: Unit
DEG_PER_S: Unit
S: Unit
MS: Unit
COMPARATOR_UNSPECIFIED: Comparator
GTE: Comparator
GT: Comparator
LTE: Comparator
LT: Comparator
EQ: Comparator
EVENT_KIND_UNSPECIFIED: EventKind
THRESHOLD: EventKind
CHANGE: EventKind
ONSET: EventKind
EXTREMUM: EventKind
RETURN_TO: EventKind
TRACK_APPEAR: EventKind
TRACK_DISAPPEAR: EventKind
DIRECTION_UNSPECIFIED: Direction
DOWN: Direction
UP: Direction
ANCHOR_POINT_UNSPECIFIED: AnchorPoint
START: AnchorPoint
END: AnchorPoint
EXTREMUM_POINT: AnchorPoint
RELATION_KIND_UNSPECIFIED: RelationKind
AFTER: RelationKind
BEFORE: RelationKind
WITHIN: RelationKind
DURING: RelationKind
QUANTIFIER_UNSPECIFIED: Quantifier
ALL: Quantifier
FIRST: Quantifier
ARGMIN: Quantifier
ARGMAX: Quantifier

class Quantity(_message.Message):
    __slots__ = ("value", "unit")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    value: float
    unit: Unit
    def __init__(self, value: _Optional[float] = ..., unit: _Optional[_Union[Unit, str]] = ...) -> None: ...

class AnchorRef(_message.Message):
    __slots__ = ("event", "point")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    POINT_FIELD_NUMBER: _ClassVar[int]
    event: str
    point: AnchorPoint
    def __init__(self, event: _Optional[str] = ..., point: _Optional[_Union[AnchorPoint, str]] = ...) -> None: ...

class EventSpec(_message.Message):
    __slots__ = ("name", "kind", "feature", "comparator", "threshold", "change", "direction", "within", "min_duration", "from_below", "sustain", "fraction", "reference", "required", "doc")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    COMPARATOR_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    WITHIN_FIELD_NUMBER: _ClassVar[int]
    MIN_DURATION_FIELD_NUMBER: _ClassVar[int]
    FROM_BELOW_FIELD_NUMBER: _ClassVar[int]
    SUSTAIN_FIELD_NUMBER: _ClassVar[int]
    FRACTION_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DOC_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: EventKind
    feature: str
    comparator: Comparator
    threshold: Quantity
    change: Quantity
    direction: Direction
    within: Quantity
    min_duration: Quantity
    from_below: Quantity
    sustain: Quantity
    fraction: float
    reference: AnchorRef
    required: bool
    doc: str
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[EventKind, str]] = ..., feature: _Optional[str] = ..., comparator: _Optional[_Union[Comparator, str]] = ..., threshold: _Optional[_Union[Quantity, _Mapping]] = ..., change: _Optional[_Union[Quantity, _Mapping]] = ..., direction: _Optional[_Union[Direction, str]] = ..., within: _Optional[_Union[Quantity, _Mapping]] = ..., min_duration: _Optional[_Union[Quantity, _Mapping]] = ..., from_below: _Optional[_Union[Quantity, _Mapping]] = ..., sustain: _Optional[_Union[Quantity, _Mapping]] = ..., fraction: _Optional[float] = ..., reference: _Optional[_Union[AnchorRef, _Mapping]] = ..., required: _Optional[bool] = ..., doc: _Optional[str] = ...) -> None: ...

class Relation(_message.Message):
    __slots__ = ("parent", "child", "kind", "min_gap", "max_gap", "required", "doc")
    PARENT_FIELD_NUMBER: _ClassVar[int]
    CHILD_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    MIN_GAP_FIELD_NUMBER: _ClassVar[int]
    MAX_GAP_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DOC_FIELD_NUMBER: _ClassVar[int]
    parent: AnchorRef
    child: AnchorRef
    kind: RelationKind
    min_gap: Quantity
    max_gap: Quantity
    required: bool
    doc: str
    def __init__(self, parent: _Optional[_Union[AnchorRef, _Mapping]] = ..., child: _Optional[_Union[AnchorRef, _Mapping]] = ..., kind: _Optional[_Union[RelationKind, str]] = ..., min_gap: _Optional[_Union[Quantity, _Mapping]] = ..., max_gap: _Optional[_Union[Quantity, _Mapping]] = ..., required: _Optional[bool] = ..., doc: _Optional[str] = ...) -> None: ...

class Selection(_message.Message):
    __slots__ = ("quantifier", "by_feature", "by_event")
    QUANTIFIER_FIELD_NUMBER: _ClassVar[int]
    BY_FEATURE_FIELD_NUMBER: _ClassVar[int]
    BY_EVENT_FIELD_NUMBER: _ClassVar[int]
    quantifier: Quantifier
    by_feature: str
    by_event: str
    def __init__(self, quantifier: _Optional[_Union[Quantifier, str]] = ..., by_feature: _Optional[str] = ..., by_event: _Optional[str] = ...) -> None: ...

class ReceiptSpec(_message.Message):
    __slots__ = ("cutoff", "sensors", "max_age", "boundary")
    CUTOFF_FIELD_NUMBER: _ClassVar[int]
    SENSORS_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    cutoff: AnchorRef
    sensors: _containers.RepeatedScalarFieldContainer[str]
    max_age: Quantity
    boundary: _common_pb2.Boundary
    def __init__(self, cutoff: _Optional[_Union[AnchorRef, _Mapping]] = ..., sensors: _Optional[_Iterable[str]] = ..., max_age: _Optional[_Union[Quantity, _Mapping]] = ..., boundary: _Optional[_Union[_common_pb2.Boundary, str]] = ...) -> None: ...

class Scope(_message.Message):
    __slots__ = ("recording_ids", "robots")
    RECORDING_IDS_FIELD_NUMBER: _ClassVar[int]
    ROBOTS_FIELD_NUMBER: _ClassVar[int]
    recording_ids: _containers.RepeatedScalarFieldContainer[str]
    robots: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, recording_ids: _Optional[_Iterable[str]] = ..., robots: _Optional[_Iterable[str]] = ...) -> None: ...

class QueryProgram(_message.Message):
    __slots__ = ("scope", "primary_event", "events", "relations", "selection", "return_anchors", "receipt", "context_before", "context_after", "unexpressible", "abstain_if_insufficient")
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_EVENT_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    RETURN_ANCHORS_FIELD_NUMBER: _ClassVar[int]
    RECEIPT_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_BEFORE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_AFTER_FIELD_NUMBER: _ClassVar[int]
    UNEXPRESSIBLE_FIELD_NUMBER: _ClassVar[int]
    ABSTAIN_IF_INSUFFICIENT_FIELD_NUMBER: _ClassVar[int]
    scope: Scope
    primary_event: str
    events: _containers.RepeatedCompositeFieldContainer[EventSpec]
    relations: _containers.RepeatedCompositeFieldContainer[Relation]
    selection: Selection
    return_anchors: _containers.RepeatedCompositeFieldContainer[AnchorRef]
    receipt: ReceiptSpec
    context_before: Quantity
    context_after: Quantity
    unexpressible: _containers.RepeatedScalarFieldContainer[str]
    abstain_if_insufficient: bool
    def __init__(self, scope: _Optional[_Union[Scope, _Mapping]] = ..., primary_event: _Optional[str] = ..., events: _Optional[_Iterable[_Union[EventSpec, _Mapping]]] = ..., relations: _Optional[_Iterable[_Union[Relation, _Mapping]]] = ..., selection: _Optional[_Union[Selection, _Mapping]] = ..., return_anchors: _Optional[_Iterable[_Union[AnchorRef, _Mapping]]] = ..., receipt: _Optional[_Union[ReceiptSpec, _Mapping]] = ..., context_before: _Optional[_Union[Quantity, _Mapping]] = ..., context_after: _Optional[_Union[Quantity, _Mapping]] = ..., unexpressible: _Optional[_Iterable[str]] = ..., abstain_if_insufficient: _Optional[bool] = ...) -> None: ...
