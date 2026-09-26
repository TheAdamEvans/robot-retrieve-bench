#!/usr/bin/env bash
# Regenerate Python classes + mypy stubs from server/proto. Run from scand/.
set -euo pipefail
out=server/src/alloy_server/gen
rm -rf "$out/alloy"
uv run python -m grpc_tools.protoc -I server/proto --python_out="$out" --pyi_out="$out" server/proto/alloy/v1/*.proto
touch "$out/__init__.py" "$out/alloy/__init__.py" "$out/alloy/v1/__init__.py"
# protoc emits absolute imports (`from alloy.v1 import ...`); make them package-relative
sed -i '' -E 's/^from alloy\.v1 import/from alloy_server.gen.alloy.v1 import/' "$out"/alloy/v1/*_pb2.py "$out"/alloy/v1/*_pb2.pyi
