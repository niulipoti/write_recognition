# 第 3 步 · 数据预处理

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐☆☆☆ | 约 30 分钟 | [第 2 步](md/step-02-认识数据.md) | [code/03_preprocess.py](../code/03_preprocess.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 理解「预处理」为什么是 AI 项目的重要一环
- [ ] 把 PIL 图片转成 PyTorch 张量（Tensor），并做归一化
- [ ] 划分训练集 / 验证集 / 测试集，说清三者的区别
- [ ] 用 DataLoader 把数据打包成一个个 batch

## 📚 知识准备

### 什么是张量（Tensor）？

张量是 PyTorch 里最核心的数据结构，可以理解为「带自动求导能力的多维数组」：

| 概念 | 例子 | 张量形状 |
| :--- | :--- | :--- |
| 0 维张量 | 一个数：`3.14` | `[]` |
| 1 维张量 | 一个列表：`[1, 2, 3]` | `[3]` |
| 2 维张量 | 一张灰度图：28×28 个像素 | `[28, 28]` |
| 4 维张量 | 一批（batch）灰度图 | `[64, 1, 28, 28]` |

4 维张量的四个数字含义：`[一批几张图, 颜色通道数, 高, 宽]`。灰度图通道 = 1；彩色图 RGB 通道 = 3。**本教程后面会反复见到 `[64, 1, 28, 28]`，请记住它**。

### 为什么要归一化？

原始像素值是 0~255 的整数，直接把大数值喂给神经网络会让训练不稳定、收敛慢。归一化就是把像素缩放到**均值为 0、标准差为 1** 的小范围：

```
归一化后的值 = (原始值/255 - 0.1307) / 0.3081
```

其中 0.1307 和 0.3081 是对 MNIST 训练集统计出的**像素均值**和**标准差**（业界通用值，直接用即可）。归一化后数据落在 -1 到 2 左右，模型训练又快又稳。

### 训练集 / 验证集 / 测试集：为什么分三份？

- **训练集（train）**：模型刷题用。模型直接看答案（标签）来调整自己
- **验证集（val）**：每次考完试自查用。训练时每轮结束在验证集上打分，用来调超参数、判断有没有过拟合，**模型不直接学它**
- **测试集（test）**：最终大考用。全部训练结束后才用，模拟真实世界的新题

打个比方：训练集是练习册（可反复做），验证集是模拟考（检验学习效果），测试集是高考（只考一次，见真章）。MNIST 自带训练/测试划分，本教程再从 6 万训练集里切 1 万张当验证集。

### 什么是 batch 和 DataLoader？

一次训练 6 万张图很慢，我们**每次取一小批（batch）**来训练，比如 64 张。这样既快，又能让梯度更新更稳定。`DataLoader` 就是自动做这件事的工具：按 batch 切分、随机打乱（shuffle）、需要时多进程加速。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 3.1 运行预处理脚本

```bash
python code/03_preprocess.py
```

```python
"""
第 3 步 · 数据预处理
运行方式：python code/03_preprocess.py
作用：把原始图像转成张量、归一化、划分训练/验证集、打包成 DataLoader。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 1. 预处理流水线：转张量 + 归一化
# MNIST 均值为 0.1307、标准差为 0.3081（对整个训练集统计得到的经验值）
transform = transforms.Compose([
    transforms.ToTensor(),                       # PIL 图像 -> [0,1] 浮点张量，形状 [1, 28, 28]
    transforms.Normalize((0.1307,), (0.3081,)),  # (x - mean) / std
])

train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

# 2. 从 6 万训练集里切出 1 万张作为验证集（训练 5 万 + 验证 1 万）
val_size = 10000
train_set, val_set = random_split(
    train_set, [50000, val_size],
    generator=torch.Generator().manual_seed(42),  # 固定随机种子：每次切分结果一致
)

# 3. 打包成 DataLoader：自动分批、打乱、多进程读取
BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print("训练批次数:", len(train_loader), "| 验证批次数:", len(val_loader), "| 测试批次数:", len(test_loader))

# 4. 取出一个 batch 看看长什么样
images, labels = next(iter(train_loader))
print("一个 batch 的图像张量形状:", images.shape)   # [64, 1, 28, 28]
print("一个 batch 的标签张量形状:", labels.shape)   # [64]
print("归一化后的像素值范围:", round(float(images.min()), 3), "~", round(float(images.max()), 3))

# 5. 把归一化后的图像"还原"显示一张，确认数据没有被弄坏
inv_norm = transforms.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,))  # 归一化的逆运算
plt.imshow(inv_norm(images[0]).squeeze(), cmap="gray")
plt.title(f"restored image, label: {labels[0].item()}")
plt.axis("off")
plt.savefig(OUT_DIR / "03_preprocessed_sample.png", dpi=150)
print("已保存预处理示例 -> outputs/03_preprocessed_sample.png")
plt.close("all")
```

**预期输出**：

```
训练批次数: 782 | 验证批次数: 157 | 测试批次数: 157
一个 batch 的图像张量形状: torch.Size([64, 1, 28, 28])
一个 batch 的标签张量形状: torch.Size([64])
归一化后的像素值范围: -0.424 ~ 2.821
```

### 3.2 逐段讲解

**① transform 流水线**

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

`Compose` 把多个处理步骤串成一条流水线，数据按顺序流过去：

- `ToTensor()`：PIL 图片 → 浮点张量，像素从 0~255 整数变成 0~1 小数，形状变为 `[1, 28, 28]`（通道数放最前面，这是 PyTorch 的约定）
- `Normalize((mean,), (std,))`：执行 `(x - mean) / std`

把这个 `transform` 传给 `datasets.MNIST(transform=transform)`，之后取出的每个样本就**自动完成预处理**。

**② 切分验证集**

```python
train_set, val_set = random_split(train_set, [50000, 10000], generator=...)
```

`random_split` 按 `[50000, 10000]` 的比例把 6 万张随机切成两份。`manual_seed(42)` 固定随机种子——种子相同，随机结果就相同，保证你每次运行切出的验证集一致，实验结果可复现。

**③ DataLoader**

```python
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
```

- `batch_size=64`：每批 64 张，训练集 5 万张 ÷ 64 ≈ 782 批（最后一批不足 64 张）
- `shuffle=True`：**训练集**每轮打乱顺序（防止模型记住顺序）；验证/测试集不需要打乱

`next(iter(train_loader))` 取第一个 batch，返回 `(图像张量, 标签张量)`。

**④ 逆归一化还原**

`Normalize((-0.1307/0.3081,), (1/0.3081,))` 是归一化的逆运算（自己代入公式算算就明白）。用它还原出一张图显示，确认「预处理没把数据弄坏」——这是检查预处理是否正确的小技巧。

### 3.3 小实验

1. 把 `BATCH_SIZE` 改成 128，观察批次数变化（782 → 391），想一想为什么
2. 把 `shuffle=True` 改成 `False`，看两个 batch 的标签顺序有什么变化
3. 试着去掉归一化（只留 `ToTensor()`），观察像素值范围变化，想想模型会怎么样

## ✅ 完成自检

- [ ] 脚本跑通，输出与「预期输出」一致
- [ ] 能说出 `[64, 1, 28, 28]` 每个数字的含义
- [ ] 能写出归一化公式，并说出归一化的目的
- [ ] 能用一句话区分训练集、验证集、测试集
- [ ] 能说出 batch_size=64 和 shuffle=True 分别是什么意思

## 🐛 常见问题

**Q：为什么 ToTensor 之后范围是 0~1，而不是 0~255？**
A：神经网络喜欢小数值输入。除以 255 后数据尺度统一，配合归一化效果更好。

**Q：0.1307 和 0.3081 哪来的？**
A：是对 MNIST 训练集像素统计出的均值与标准差，全行业通用的经验值，直接用。以后做其他数据集时，可以自己统计 `data.mean()` 和 `data.std()` 得到。

**Q：`random_split` 每次运行结果都一样吗？**
A：因为固定了 `manual_seed(42)`，一样。换个种子数字试试，切分会变。复现实验靠的就是固定种子。

**Q：为什么验证集和测试集不 shuffle？**
A：评估时顺序不影响结果，不打乱还省时间。只有训练要打乱，防止模型记住数据顺序「作弊」。

## 📌 本步小结

1. 预处理流水线：PIL 图 → 张量（0~1）→ 归一化（均值 0、标准差 1），通过 `transform` 参数自动生效
2. 数据分三份：5 万训练、1 万验证、1 万测试，各司其职
3. DataLoader 按 batch=64 自动分批，训练集每轮打乱

## 🔗 下一步

👉 [第 4 步 · 构建模型](md/step-04-构建模型.md)：亲手搭出人生第一个神经网络——一个由 784 个输入、10 个输出的「数字认读机」。
