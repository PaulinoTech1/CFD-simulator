#!/usr/bin/env bash
# Build the host path first. Call nvcc only if it exists.
# Never adds --use_fast_math.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARCH="${ARCH:-sm_86}"
HOST_BIN="${HOST_BIN:-karman3d-host}"
DEV_BIN="${DEV_BIN:-karman3d}"

echo "host path: g++ -O2 -std=c++17 -DKARMAN_CPU"
g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o "$HOST_BIN" kernels/karman3d.cu
echo "host binary: $HOST_BIN"

if command -v nvcc >/dev/null 2>&1; then
  echo "device path: nvcc -O3 -arch=$ARCH"
  nvcc -O3 -arch="$ARCH" -o "$DEV_BIN" kernels/karman3d.cu
  echo "device binary: $DEV_BIN"
else
  echo "GPU compiler not found, host path only"
fi
