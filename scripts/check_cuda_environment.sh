#!/usr/bin/env bash
set -u

echo "=== host/user ==="
hostname
whoami
id

echo
echo "=== nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo
echo "=== /dev/nvidia* ==="
shopt -s nullglob
nvidia_devices=(/dev/nvidia*)
shopt -u nullglob
if [ "${#nvidia_devices[@]}" -eq 0 ]; then
  echo "no /dev/nvidia* devices visible"
else
  ls -l "${nvidia_devices[@]}"
fi

echo
echo "=== environment ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-}"

echo
echo "=== conda/python ==="
conda env list || true
conda run -n sam301 which python || true
conda run -n sam301 python --version || true

echo
echo "=== pytorch cuda ==="
conda run -n sam301 python -c 'import os, torch; print("torch_file:", torch.__file__); print("torch_version:", torch.__version__); print("torch_cuda_version:", torch.version.cuda); print("cuda_available:", torch.cuda.is_available()); print("device_count:", torch.cuda.device_count()); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' || true

echo
echo "=== torch build config ==="
conda run -n sam301 python -c 'import torch; print(torch.__config__.show())' || true

echo
echo "=== sam3 import ==="
PYTHONPATH=/home/book/sam301 conda run -n sam301 python -c 'import sam3; print("sam3_file:", sam3.__file__)' || true
