#!/usr/bin/env python3
"""
GPU検出デバッグスクリプト
Dockerコンテナ内で実行して、GPUが正しく認識されているか確認する
"""

import torch
import os
import subprocess

print("=" * 60)
print("GPU Detection Debug Script")
print("=" * 60)

# 環境変数の確認
print("\n[Environment Variables]")
env_vars = ['HSA_OVERRIDE_GFX_VERSION', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
            'ROCM_VERSION', 'HIP_PLATFORM', 'GPU_DEVICE_ORDINAL']
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"  {var}: {value}")

# PyTorchのバージョン情報
print("\n[PyTorch Version Info]")
print(f"  torch.__version__: {torch.__version__}")
print(f"  torch.version.cuda: {getattr(torch.version, 'cuda', 'N/A')}")
print(f"  torch.version.hip: {getattr(torch.version, 'hip', 'N/A')}")

# GPU検出状態
print("\n[GPU Detection]")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"  torch.cuda.current_device(): {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - Major/Minor: {props.major}.{props.minor}")
else:
    print("  No CUDA/ROCm devices detected")

# HIPの確認
print("\n[HIP Check]")
if hasattr(torch, 'hip'):
    print(f"  torch.hip available: Yes")
    if hasattr(torch.hip, 'is_available'):
        print(f"  torch.hip.is_available(): {torch.hip.is_available()}")
else:
    print(f"  torch.hip available: No")

# システムコマンドでデバイスを確認
print("\n[System Device Check]")
try:
    # ROCmデバイスの確認
    result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("  rocm-smi output:")
        print(result.stdout[:500])  # 最初の500文字のみ表示
    else:
        print("  rocm-smi not found or failed")
except Exception as e:
    print(f"  rocm-smi error: {e}")

# /dev/kfd と /dev/dri の確認
print("\n[Device Files]")
if os.path.exists('/dev/kfd'):
    print("  /dev/kfd: EXISTS")
else:
    print("  /dev/kfd: NOT FOUND")

if os.path.exists('/dev/dri'):
    print("  /dev/dri: EXISTS")
    try:
        dri_files = os.listdir('/dev/dri')
        print(f"    Contents: {dri_files}")
    except Exception as e:
        print(f"    Error listing /dev/dri: {e}")
else:
    print("  /dev/dri: NOT FOUND")

# 簡単なテスト
print("\n[Simple GPU Test]")
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = x * 2
        print(f"  GPU computation test: SUCCESS")
        print(f"  Result: {y.cpu().numpy()}")
    else:
        print("  GPU not available for testing")
except Exception as e:
    print(f"  GPU computation test FAILED: {e}")

print("\n" + "=" * 60)