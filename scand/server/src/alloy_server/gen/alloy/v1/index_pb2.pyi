from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StageScope(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STAGE_SCOPE_UNSPECIFIED: _ClassVar[StageScope]
    PER_RECORDING: _ClassVar[StageScope]
    CORPUS: _ClassVar[StageScope]
STAGE_SCOPE_UNSPECIFIED: StageScope
PER_RECORDING: StageScope
CORPUS: StageScope

class SourceSpec(_message.Message):
    __slots__ = ("impl", "raw_dir", "index_csv", "aliases", "id_template", "held_out")
    class AliasesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    IMPL_FIELD_NUMBER: _ClassVar[int]
    RAW_DIR_FIELD_NUMBER: _ClassVar[int]
    INDEX_CSV_FIELD_NUMBER: _ClassVar[int]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    ID_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    HELD_OUT_FIELD_NUMBER: _ClassVar[int]
    impl: str
    raw_dir: str
    index_csv: str
    aliases: _containers.ScalarMap[str, str]
    id_template: str
    held_out: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, impl: _Optional[str] = ..., raw_dir: _Optional[str] = ..., index_csv: _Optional[str] = ..., aliases: _Optional[_Mapping[str, str]] = ..., id_template: _Optional[str] = ..., held_out: _Optional[_Iterable[str]] = ...) -> None: ...

class IndexStage(_message.Message):
    __slots__ = ("name", "impl", "params", "enabled", "requires_flag")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    IMPL_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    REQUIRES_FLAG_FIELD_NUMBER: _ClassVar[int]
    name: str
    impl: str
    params: _containers.ScalarMap[str, str]
    enabled: bool
    requires_flag: bool
    def __init__(self, name: _Optional[str] = ..., impl: _Optional[str] = ..., params: _Optional[_Mapping[str, str]] = ..., enabled: _Optional[bool] = ..., requires_flag: _Optional[bool] = ...) -> None: ...

class SinkSpec(_message.Message):
    __slots__ = ("impl", "path")
    IMPL_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    impl: str
    path: str
    def __init__(self, impl: _Optional[str] = ..., path: _Optional[str] = ...) -> None: ...

class IndexSpec(_message.Message):
    __slots__ = ("index_id", "source", "stages", "sink")
    INDEX_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    STAGES_FIELD_NUMBER: _ClassVar[int]
    SINK_FIELD_NUMBER: _ClassVar[int]
    index_id: str
    source: SourceSpec
    stages: _containers.RepeatedCompositeFieldContainer[IndexStage]
    sink: SinkSpec
    def __init__(self, index_id: _Optional[str] = ..., source: _Optional[_Union[SourceSpec, _Mapping]] = ..., stages: _Optional[_Iterable[_Union[IndexStage, _Mapping]]] = ..., sink: _Optional[_Union[SinkSpec, _Mapping]] = ...) -> None: ...

class ArtifactRecord(_message.Message):
    __slots__ = ("key", "stage", "version", "recording_id", "params_sha256", "input_keys", "paths", "content_key", "wall_s", "created_at", "info")
    class InfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    KEY_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    RECORDING_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMS_SHA256_FIELD_NUMBER: _ClassVar[int]
    INPUT_KEYS_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_KEY_FIELD_NUMBER: _ClassVar[int]
    WALL_S_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    key: str
    stage: str
    version: str
    recording_id: str
    params_sha256: str
    input_keys: _containers.RepeatedScalarFieldContainer[str]
    paths: _containers.RepeatedScalarFieldContainer[str]
    content_key: str
    wall_s: float
    created_at: str
    info: _containers.ScalarMap[str, str]
    def __init__(self, key: _Optional[str] = ..., stage: _Optional[str] = ..., version: _Optional[str] = ..., recording_id: _Optional[str] = ..., params_sha256: _Optional[str] = ..., input_keys: _Optional[_Iterable[str]] = ..., paths: _Optional[_Iterable[str]] = ..., content_key: _Optional[str] = ..., wall_s: _Optional[float] = ..., created_at: _Optional[str] = ..., info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SinkManifest(_message.Message):
    __slots__ = ("index_id", "spec_sha256", "artifacts")
    class ArtifactsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ArtifactRecord
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ArtifactRecord, _Mapping]] = ...) -> None: ...
    INDEX_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_SHA256_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    index_id: str
    spec_sha256: str
    artifacts: _containers.MessageMap[str, ArtifactRecord]
    def __init__(self, index_id: _Optional[str] = ..., spec_sha256: _Optional[str] = ..., artifacts: _Optional[_Mapping[str, ArtifactRecord]] = ...) -> None: ...
