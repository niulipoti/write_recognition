"""
第 1 步 · 环境自检脚本
运行方式：python code/01_check_env.py
作用：检查 Python、PyTorch、torchvision 是否安装成功。
"""
import sys

print("=" * 50)
print("Python 版本:", sys.version.split()[0])
print("Python 路径:", sys.executable)
print("=" * 50)

try:
    import numpy
    print(f"numpy      OK  {numpy.__version__}")
except ImportError:
    print("numpy      未安装！请运行: pip install numpy")

try:
    import matplotlib
    print(f"matplotlib OK  {matplotlib.__version__}")
except ImportError:
    print("matplotlib 未安装！请运行: pip install matplotlib")

try:
    import torch
    print(f"PyTorch    OK  {torch.__version__}")
except ImportError:
    print("PyTorch 未安装！请运行: pip install torch torchvision")
    sys.exit(1)

try:
    import torchvision
    print(f"torchvision OK {torchvision.__version__}")
except ImportError:
    print("torchvision 未安装！请运行: pip install torchvision")
    sys.exit(1)

print("-" * 50)
print("CUDA（显卡加速）可用:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("本教程将使用设备:", device, "（CPU 就完全够用）")
print("=" * 50)
print("环境自检通过，可以开始第 2 步啦！")
