from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Basis(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BASIS_UNSPECIFIED: _ClassVar[Basis]
    RECORDED: _ClassVar[Basis]
    URDF_NOMINAL: _ClassVar[Basis]
    DATASHEET_NOMINAL: _ClassVar[Basis]
    ESTIMATED_BAND: _ClassVar[Basis]

class SensorKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SENSOR_KIND_UNSPECIFIED: _ClassVar[SensorKind]
    CAMERA: _ClassVar[SensorKind]
    LIDAR_3D: _ClassVar[SensorKind]
    LIDAR_2D: _ClassVar[SensorKind]
    ODOMETRY: _ClassVar[SensorKind]
    IMU: _ClassVar[SensorKind]
    TF: _ClassVar[SensorKind]
    JOYSTICK: _ClassVar[SensorKind]
    CAMERA_INFO: _ClassVar[SensorKind]

class TwistFrame(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TWIST_FRAME_UNSPECIFIED: _ClassVar[TwistFrame]
    TWIST_IN_ODOM: _ClassVar[TwistFrame]
    TWIST_IN_BODY: _ClassVar[TwistFrame]

class SmoothingMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SMOOTHING_METHOD_UNSPECIFIED: _ClassVar[SmoothingMethod]
    TRAILING_MEDIAN: _ClassVar[SmoothingMethod]
    GAIT_SYNC_MEAN: _ClassVar[SmoothingMethod]
BASIS_UNSPECIFIED: Basis
RECORDED: Basis
URDF_NOMINAL: Basis
DATASHEET_NOMINAL: Basis
ESTIMATED_BAND: Basis
SENSOR_KIND_UNSPECIFIED: SensorKind
CAMERA: SensorKind
LIDAR_3D: SensorKind
LIDAR_2D: SensorKind
ODOMETRY: SensorKind
IMU: SensorKind
TF: SensorKind
JOYSTICK: SensorKind
CAMERA_INFO: SensorKind
TWIST_FRAME_UNSPECIFIED: TwistFrame
TWIST_IN_ODOM: TwistFrame
TWIST_IN_BODY: TwistFrame
SMOOTHING_METHOD_UNSPECIFIED: SmoothingMethod
TRAILING_MEDIAN: SmoothingMethod
GAIT_SYNC_MEAN: SmoothingMethod

class Band(_message.Message):
    __slots__ = ("nominal", "lo", "hi", "basis")
    NOMINAL_FIELD_NUMBER: _ClassVar[int]
    LO_FIELD_NUMBER: _ClassVar[int]
    HI_FIELD_NUMBER: _ClassVar[int]
    BASIS_FIELD_NUMBER: _ClassVar[int]
    nominal: float
    lo: float
    hi: float
    basis: Basis
    def __init__(self, nominal: _Optional[float] = ..., lo: _Optional[float] = ..., hi: _Optional[float] = ..., basis: _Optional[_Union[Basis, str]] = ...) -> None: ...

class CameraModel(_message.Message):
    __slots__ = ("width", "height", "hfov_deg", "height_m", "pitch_down_deg", "intrinsics_recorded")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    HFOV_DEG_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_M_FIELD_NUMBER: _ClassVar[int]
    PITCH_DOWN_DEG_FIELD_NUMBER: _ClassVar[int]
    INTRINSICS_RECORDED_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    hfov_deg: Band
    height_m: Band
    pitch_down_deg: Band
    intrinsics_recorded: bool
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., hfov_deg: _Optional[_Union[Band, _Mapping]] = ..., height_m: _Optional[_Union[Band, _Mapping]] = ..., pitch_down_deg: _Optional[_Union[Band, _Mapping]] = ..., intrinsics_recorded: _Optional[bool] = ...) -> None: ...

class SensorSpec(_message.Message):
    __slots__ = ("name", "kind", "topics", "nominal_hz", "display_rotation_deg", "display_names", "camera", "twist_frame", "note")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    NOMINAL_HZ_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_ROTATION_DEG_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAMES_FIELD_NUMBER: _ClassVar[int]
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    TWIST_FRAME_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: SensorKind
    topics: _containers.RepeatedScalarFieldContainer[str]
    nominal_hz: float
    display_rotation_deg: _containers.RepeatedScalarFieldContainer[int]
    display_names: _containers.RepeatedScalarFieldContainer[str]
    camera: CameraModel
    twist_frame: TwistFrame
    note: str
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[SensorKind, str]] = ..., topics: _Optional[_Iterable[str]] = ..., nominal_hz: _Optional[float] = ..., display_rotation_deg: _Optional[_Iterable[int]] = ..., display_names: _Optional[_Iterable[str]] = ..., camera: _Optional[_Union[CameraModel, _Mapping]] = ..., twist_frame: _Optional[_Union[TwistFrame, str]] = ..., note: _Optional[str] = ...) -> None: ...

class SpeedSmoothing(_message.Message):
    __slots__ = ("method", "window_s", "gait_cycles")
    METHOD_FIELD_NUMBER: _ClassVar[int]
    WINDOW_S_FIELD_NUMBER: _ClassVar[int]
    GAIT_CYCLES_FIELD_NUMBER: _ClassVar[int]
    method: SmoothingMethod
    window_s: float
    gait_cycles: int
    def __init__(self, method: _Optional[_Union[SmoothingMethod, str]] = ..., window_s: _Optional[float] = ..., gait_cycles: _Optional[int] = ...) -> None: ...

class Footprint(_message.Message):
    __slots__ = ("length_m", "width_m", "height_m", "basis")
    LENGTH_M_FIELD_NUMBER: _ClassVar[int]
    WIDTH_M_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_M_FIELD_NUMBER: _ClassVar[int]
    BASIS_FIELD_NUMBER: _ClassVar[int]
    length_m: float
    width_m: float
    height_m: float
    basis: Basis
    def __init__(self, length_m: _Optional[float] = ..., width_m: _Optional[float] = ..., height_m: _Optional[float] = ..., basis: _Optional[_Union[Basis, str]] = ...) -> None: ...

class Corridor(_message.Message):
    __slots__ = ("half_width_m", "length_m", "basis")
    HALF_WIDTH_M_FIELD_NUMBER: _ClassVar[int]
    LENGTH_M_FIELD_NUMBER: _ClassVar[int]
    BASIS_FIELD_NUMBER: _ClassVar[int]
    half_width_m: float
    length_m: float
    basis: Basis
    def __init__(self, half_width_m: _Optional[float] = ..., length_m: _Optional[float] = ..., basis: _Optional[_Union[Basis, str]] = ...) -> None: ...

class EmbodimentProfile(_message.Message):
    __slots__ = ("embodiment_id", "robot", "locomotion", "footprint", "max_speed_mps", "sensors", "corridor", "clearance_sensor", "speed_smoothing", "signature_topics", "absent", "profile_version")
    EMBODIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_FIELD_NUMBER: _ClassVar[int]
    LOCOMOTION_FIELD_NUMBER: _ClassVar[int]
    FOOTPRINT_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_MPS_FIELD_NUMBER: _ClassVar[int]
    SENSORS_FIELD_NUMBER: _ClassVar[int]
    CORRIDOR_FIELD_NUMBER: _ClassVar[int]
    CLEARANCE_SENSOR_FIELD_NUMBER: _ClassVar[int]
    SPEED_SMOOTHING_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_TOPICS_FIELD_NUMBER: _ClassVar[int]
    ABSENT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    embodiment_id: str
    robot: str
    locomotion: str
    footprint: Footprint
    max_speed_mps: Band
    sensors: _containers.RepeatedCompositeFieldContainer[SensorSpec]
    corridor: Corridor
    clearance_sensor: str
    speed_smoothing: SpeedSmoothing
    signature_topics: _containers.RepeatedScalarFieldContainer[str]
    absent: _containers.RepeatedScalarFieldContainer[str]
    profile_version: str
    def __init__(self, embodiment_id: _Optional[str] = ..., robot: _Optional[str] = ..., locomotion: _Optional[str] = ..., footprint: _Optional[_Union[Footprint, _Mapping]] = ..., max_speed_mps: _Optional[_Union[Band, _Mapping]] = ..., sensors: _Optional[_Iterable[_Union[SensorSpec, _Mapping]]] = ..., corridor: _Optional[_Union[Corridor, _Mapping]] = ..., clearance_sensor: _Optional[str] = ..., speed_smoothing: _Optional[_Union[SpeedSmoothing, _Mapping]] = ..., signature_topics: _Optional[_Iterable[str]] = ..., absent: _Optional[_Iterable[str]] = ..., profile_version: _Optional[str] = ...) -> None: ...

class TopicProfile(_message.Message):
    __slots__ = ("topic", "msgtype", "count", "hz", "max_gap_s", "gaps_over_2x_period", "header_present", "median_capture_to_receipt_ms", "p99_capture_to_receipt_ms", "future_stamped_frac", "max_future_ms")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    MSGTYPE_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    HZ_FIELD_NUMBER: _ClassVar[int]
    MAX_GAP_S_FIELD_NUMBER: _ClassVar[int]
    GAPS_OVER_2X_PERIOD_FIELD_NUMBER: _ClassVar[int]
    HEADER_PRESENT_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_CAPTURE_TO_RECEIPT_MS_FIELD_NUMBER: _ClassVar[int]
    P99_CAPTURE_TO_RECEIPT_MS_FIELD_NUMBER: _ClassVar[int]
    FUTURE_STAMPED_FRAC_FIELD_NUMBER: _ClassVar[int]
    MAX_FUTURE_MS_FIELD_NUMBER: _ClassVar[int]
    topic: str
    msgtype: str
    count: int
    hz: float
    max_gap_s: float
    gaps_over_2x_period: int
    header_present: bool
    median_capture_to_receipt_ms: float
    p99_capture_to_receipt_ms: float
    future_stamped_frac: float
    max_future_ms: float
    def __init__(self, topic: _Optional[str] = ..., msgtype: _Optional[str] = ..., count: _Optional[int] = ..., hz: _Optional[float] = ..., max_gap_s: _Optional[float] = ..., gaps_over_2x_period: _Optional[int] = ..., header_present: _Optional[bool] = ..., median_capture_to_receipt_ms: _Optional[float] = ..., p99_capture_to_receipt_ms: _Optional[float] = ..., future_stamped_frac: _Optional[float] = ..., max_future_ms: _Optional[float] = ...) -> None: ...

class GaitEstimate(_message.Message):
    __slots__ = ("period_s", "frequency_hz", "peak_to_median", "windows")
    PERIOD_S_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_HZ_FIELD_NUMBER: _ClassVar[int]
    PEAK_TO_MEDIAN_FIELD_NUMBER: _ClassVar[int]
    WINDOWS_FIELD_NUMBER: _ClassVar[int]
    period_s: float
    frequency_hz: float
    peak_to_median: float
    windows: int
    def __init__(self, period_s: _Optional[float] = ..., frequency_hz: _Optional[float] = ..., peak_to_median: _Optional[float] = ..., windows: _Optional[int] = ...) -> None: ...

class IntakeReport(_message.Message):
    __slots__ = ("recording_id", "source", "source_sha256", "source_bytes", "embodiment_id", "profile_version", "missing_signature_topics", "unprofiled_topics", "topics", "gait", "anomalies", "duration_s", "intake_version")
    RECORDING_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SHA256_FIELD_NUMBER: _ClassVar[int]
    SOURCE_BYTES_FIELD_NUMBER: _ClassVar[int]
    EMBODIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    MISSING_SIGNATURE_TOPICS_FIELD_NUMBER: _ClassVar[int]
    UNPROFILED_TOPICS_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    GAIT_FIELD_NUMBER: _ClassVar[int]
    ANOMALIES_FIELD_NUMBER: _ClassVar[int]
    DURATION_S_FIELD_NUMBER: _ClassVar[int]
    INTAKE_VERSION_FIELD_NUMBER: _ClassVar[int]
    recording_id: str
    source: str
    source_sha256: str
    source_bytes: int
    embodiment_id: str
    profile_version: str
    missing_signature_topics: _containers.RepeatedScalarFieldContainer[str]
    unprofiled_topics: _containers.RepeatedScalarFieldContainer[str]
    topics: _containers.RepeatedCompositeFieldContainer[TopicProfile]
    gait: GaitEstimate
    anomalies: _containers.RepeatedScalarFieldContainer[str]
    duration_s: float
    intake_version: str
    def __init__(self, recording_id: _Optional[str] = ..., source: _Optional[str] = ..., source_sha256: _Optional[str] = ..., source_bytes: _Optional[int] = ..., embodiment_id: _Optional[str] = ..., profile_version: _Optional[str] = ..., missing_signature_topics: _Optional[_Iterable[str]] = ..., unprofiled_topics: _Optional[_Iterable[str]] = ..., topics: _Optional[_Iterable[_Union[TopicProfile, _Mapping]]] = ..., gait: _Optional[_Union[GaitEstimate, _Mapping]] = ..., anomalies: _Optional[_Iterable[str]] = ..., duration_s: _Optional[float] = ..., intake_version: _Optional[str] = ...) -> None: ...
