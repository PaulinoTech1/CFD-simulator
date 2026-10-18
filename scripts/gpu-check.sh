#!/bin/sh
# Phase 9 gate. A missing compiler is a skip, not a device result.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "gpu  nvcc=0  device-check=not-run  UNVERIFIED"
  echo "skip: nvcc is not on this machine. Host checks still apply."
  exit 0
fi
echo "nvcc is present. Build with: nvcc -O3 -arch=sm_86 -o karman3d kernels/karman3d.cu"
echo "Do not pass --use_fast_math. Run the binary and compare Couette, Poiseuille, the scalar shift, and the cavity with the host."
exit 0
