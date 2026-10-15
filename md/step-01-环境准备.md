# 第 1 步 · 环境准备

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐☆☆☆☆ | 约 30 分钟 | 无 | [code/01_check_env.py](../code/01_check_env.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 检查并安装好 Python 环境
- [ ] 创建并激活项目专属的虚拟环境
- [ ] 安装 PyTorch 等所有依赖
- [ ] 运行环境自检脚本，确认一切就绪

## 📚 知识准备

### 什么是 Python，为什么要用它学 AI？

Python 是目前 AI 领域使用最广泛的编程语言，语法简单、生态强大。我们教程里的所有代码都是 Python。

### 什么是虚拟环境（venv）？

想象你在做菜：每个项目是一道菜，需要的调料（第三方库）各不相同。如果所有菜共用一间厨房（系统 Python），A 菜需要老抽 1.0、B 菜需要老抽 2.0，就会打架。**虚拟环境就是给每个项目一间独立的厨房**，互不干扰。

- 创建：`python -m venv .venv` —— 在本项目里建一个叫 `.venv` 的独立环境
- 激活：`.venv\Scripts\Activate.ps1`（Windows PowerShell）—— 之后所有 `pip install` 都装进这个环境
- 退出：`deactivate`

### 什么是 PyTorch？

PyTorch 是 Meta（Facebook）开源的深度学习框架，是目前**学术界和工业界最流行**的框架之一。它帮我们做两件大事：

1. **张量（Tensor）计算**：张量可以理解为「多维数组」，图像、模型参数都是张量
2. **自动求导（Autograd）**：帮我们自动计算梯度，实现反向传播（第 5 步会用到）

另一个常见框架是 TensorFlow/Keras，原理相通，学会一个再学另一个很快。

### 需要显卡（GPU）吗？

**不需要。** 显卡能加速训练，但 MNIST 数据集很小，用 CPU 训练几分钟也能完成。本教程全程 CPU 可跑，第 9 步的 CNN 也只需几分钟。等你以后跑大模型，再考虑 GPU 不迟。

## 🛠️ 动手实践

### 1.1 检查 Python 是否已安装

打开终端（Windows：按 `Win + R`，输入 `powershell` 回车；macOS/Linux：打开 Terminal），输入：

```bash
python --version
```

- 显示 `Python 3.9.x` ~ `3.12.x` 之类 → 已安装，跳到 1.2
- 显示 `Python 3.13` 以上 → 也可以用，但个别库可能还没适配，建议装 3.11 或 3.12
- 提示「不是内部或外部命令」→ 没装。去 [python.org](https://www.python.org/downloads/) 下载安装包，**安装时务必勾选底部的 “Add Python to PATH”**，装完重开终端再试一次

> 💡 Windows 有时 `python` 会跳转到微软商店。改用 `py -3 --version` 试试；如果 `py` 也找不到，说明没装好。

### 1.2 创建并激活虚拟环境

把终端切到本项目目录（`cd 手写数字识别教程`），执行：

```bash
# 创建虚拟环境（只需做一次）
python -m venv .venv

# 激活（每次打开新终端都要做一次）
.venv\Scripts\Activate.ps1
```

激活成功后，命令行最前面会出现绿色的 `(.venv)` 字样，说明你现在在虚拟环境里了。

> 💡 若 PowerShell 提示「禁止运行脚本」，先执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，输入 Y 回车，再重新激活。
> macOS/Linux 的激活命令是 `source .venv/bin/activate`。

### 1.3 安装依赖

激活虚拟环境后，执行（国内用户建议加上 `-i` 镜像参数，速度快很多）：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这条命令会读取项目里的 `requirements.txt`，一次装齐 5 个库：

| 库 | 作用 |
| :--- | :--- |
| torch | PyTorch 深度学习框架本体 |
| torchvision | PyTorch 官方视觉工具包（负责下载、加载 MNIST 数据集） |
| numpy | 数值计算基础库 |
| matplotlib | 画图库（可视化数据、画学习曲线） |
| scikit-learn | 机器学习工具库（本教程用它画混淆矩阵） |

下载约几百 MB，耐心等待。看到 `Successfully installed ...` 就成功了。

### 1.4 运行环境自检脚本

```bash
python code/01_check_env.py
```

**预期输出**（版本号可能略有不同）：

```
==================================================
Python 版本: 3.11.9
Python 路径: ...\手写数字识别教程\.venv\Scripts\python.exe
==================================================
numpy      OK  2.1.2
matplotlib OK  3.9.2
PyTorch    OK  2.5.1
torchvision OK 0.20.1
--------------------------------------------------
CUDA（显卡加速）可用: False
本教程将使用设备: cpu （CPU 就完全够用）
==================================================
环境自检通过，可以开始第 2 步啦！
```

完整脚本如下（每一行都有注释，请读一遍再跑）：

```python
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
```

## ✅ 完成自检

逐条确认：

- [ ] `python --version` 能打印出 3.9 ~ 3.12 的版本号
- [ ] 终端提示符前出现了 `(.venv)`
- [ ] `pip install -r requirements.txt` 以 `Successfully installed` 结尾
- [ ] `python code/01_check_env.py` 输出「环境自检通过」，没有红色报错
- [ ] 能说出虚拟环境的作用（一句话即可）

全部打勾，进入下一步！

## 🐛 常见问题

**Q：`pip install` 特别慢或超时？**
A：加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（清华镜像），或换成阿里镜像 `https://mirrors.aliyun.com/pypi/simple/`。科学上网时可以把 `-i` 去掉用官方源。

**Q：激活虚拟环境提示「因为在此系统上禁止运行脚本」？**
A：执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，输入 `Y` 回车，然后重新激活。

**Q：`python` 命令打不开？**
A：重装 Python 时勾选 “Add Python to PATH”；或改用 `py` 命令（如 `py -3 --version`、`py -3 -m venv .venv`）。

**Q：CUDA 显示 False，会影响学习吗？**
A：完全不影响。本教程所有代码 CPU 都能流畅运行，显卡只是锦上添花。

## 📌 本步小结

1. 虚拟环境 = 项目专属的「独立厨房」，用 `python -m venv .venv` 创建、`Activate.ps1` 激活
2. PyTorch 是我们要用的深度学习框架，负责张量计算和自动求导
3. 环境自检通过 = 万事俱备，只欠数据

## 🔗 下一步

👉 [第 2 步 · 认识数据](md/step-02-认识数据.md)：下载 MNIST 数据集，亲手「看」一眼 AI 要吃的数据长什么样。
