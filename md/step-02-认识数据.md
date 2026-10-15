# 第 2 步 · 认识数据

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐☆☆☆☆ | 约 25 分钟 | [第 1 步](md/step-01-环境准备.md) | [code/02_explore_data.py](../code/02_explore_data.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 说清「数据集、样本、标签」是什么
- [ ] 用 torchvision 一键下载、加载 MNIST 数据集
- [ ] 看懂一张图片在计算机里其实是一个 28×28 的数字矩阵
- [ ] 可视化样本并观察数据分布

## 📚 知识准备

### 机器学习三件套：数据、模型、算法

所有 AI 项目都可以概括成一句话：**用数据训练模型，让模型学会规律**。

- **数据集（Dataset）**：一堆带答案的例题。MNIST 就是 7 万张手写数字图片，每张都标好了它是几
- **样本（Sample）**：数据集里的一条数据 = 一张图片 + 它的答案
- **标签（Label）**：样本的答案，这里是 0~9 的数字
- **特征（Feature）**：用来预测的输入，这里是图片的像素

就像学生刷题：图片是题目，标签是答案，模型就是那个学生。我们之后做的「训练」，就是让模型反复刷题。

### MNIST 是什么？

MNIST（Modified NIST）是 1998 年发布的手写数字数据集，由美国国家标准与技术研究院（NIST）的原始数据整理而来：

| 部分 | 数量 | 用途 |
| :--- | :--- | :--- |
| 训练集（train） | 60,000 张 | 给模型「刷题」用 |
| 测试集（test） | 10,000 张 | 「考试」用——模型从没见过的题 |

每张图片都是 **28×28 像素的灰度图**（只有一个通道，值 0=黑 ~ 255=白）。

### 图片在计算机里就是一个矩阵

比如数字「1」可能长这样（示意，非真实数据）：

```
0   0   0  80 255  80   0   0
0   0   0   0 255   0   0   0
0   0   0   0 255   0   0   0
0   0   0   0 255   0   0   0
...
```

计算机没有「看图」的能力，它看到的就是这样一个数字矩阵。**AI 的本质就是从矩阵中找规律**。

### torchvision 是什么？

torchvision 是 PyTorch 官方的「视觉工具包」，提供：常用数据集（含 MNIST）的一键下载加载、图像预处理工具、预训练模型等。本教程主要用它的 `datasets` 和 `transforms` 两个模块。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 2.1 下载并加载数据

新建终端（记得先激活 `.venv`），在项目根目录运行：

```bash
python code/02_explore_data.py
```

第一次运行会自动下载数据（约 12 MB），保存在 `data/` 目录，之后运行都走本地缓存。脚本内容如下，先读一遍再运行：

```python
"""
第 2 步 · 认识 MNIST 数据集
运行方式：python code/02_explore_data.py
作用：下载并加载 MNIST，打印基本信息，可视化样本和类别分布。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets

# 项目根目录：无论从哪个目录运行本脚本，都能正确定位到项目文件夹
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 1. 下载并加载 MNIST（第一次运行会自动下载，之后使用本地缓存）
train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True)
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True)

# 2. 基本信息
print("训练集样本数:", len(train_set))
print("测试集样本数:", len(test_set))
image, label = train_set[0]
print("第 0 个样本：图像类型", type(image).__name__, "| 尺寸", image.size, "| 标签", label)
print("标签含义：0~9 共 10 个数字")

# 3. 可视化 5x5 共 25 个样本
fig, axes = plt.subplots(5, 5, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img, lab = train_set[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"label: {lab}", fontsize=10)
    ax.axis("off")
fig.suptitle("MNIST: 25 sample images")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_sample_digits.png", dpi=150)
print("已保存样本图 -> outputs/02_sample_digits.png")

# 4. 各类别数量分布
labels = [train_set[i][1] for i in range(len(train_set))]
plt.figure(figsize=(6, 4))
plt.hist(labels, bins=range(11), rwidth=0.8, align="left", color="#4C72B0")
plt.xticks(range(10))
plt.xlabel("digit label")
plt.ylabel("count")
plt.title("MNIST training set: label distribution")
plt.savefig(OUT_DIR / "02_label_dist.png", dpi=150)
print("已保存分布图 -> outputs/02_label_dist.png")
plt.close("all")
```

**预期输出**：

```
训练集样本数: 60000
测试集样本数: 10000
第 0 个样本：图像类型 Image | 尺寸 (28, 28) | 标签 5
标签含义：0~9 共 10 个数字
已保存样本图 -> outputs/02_sample_digits.png
已保存分布图 -> outputs/02_label_dist.png
```

### 2.2 逐段讲解

**① 定位项目目录**

```python
ROOT = Path(__file__).resolve().parent.parent
```

`__file__` 是脚本自己的路径，`.resolve()` 转成绝对路径，向上两级 `.parent.parent` 就是项目根目录。之后 `data/`、`outputs/` 都挂在它下面——这样**无论在哪个目录运行脚本都不会找错位置**。

**② 加载数据集**

```python
train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True)
```

- `root`：数据保存位置
- `train=True`：要训练集；`train=False` 要测试集
- `download=True`：本地没有就自动下载

加载后 `train_set` 是一个可以按下标访问的「数据集对象」，`train_set[0]` 返回 `(图片, 标签)` 二元组。图片是 PIL 的 `Image` 对象，尺寸 28×28。

**③ 可视化样本**：`plt.subplots(5, 5)` 生成 5 行 5 列的网格，循环把 25 张图贴上去。`cmap="gray"` 指定灰度显示。

**④ 类别分布**：统计 6 万张图里每个数字出现多少次，画成柱状图。分布大约均匀（每个数字 5~7 千张），说明数据集是「平衡」的——这是好事，模型不会偏科。

### 2.3 亲手看一眼数据

打开 `outputs/` 目录，双击查看刚生成的两张图：

- `02_sample_digits.png`：25 个手写数字，注意它们**粗细、倾斜、位置各不相同**——这是真实世界的常态，也说明任务有难度
- `02_label_dist.png`：0~9 的分布柱状图

再打开 `data/MNIST/raw/`，你会看到 4 个 `.gz` 压缩文件——那就是整个数据集的原始字节形态（暂时不用管，第 3 步会解释 torchvision 如何把它们变成训练数据）。

### 2.4 小实验：修改代码观察

把 `train_set[0]` 改成 `train_set[59999]`、把 `subplots(5, 5)` 改成 `subplots(2, 8)`，重新运行，观察变化。改坏了也没关系，代码可以随时重写。

## ✅ 完成自检

- [ ] 脚本一次跑通，无红色报错
- [ ] `data/MNIST/raw/` 下能看到 4 个数据文件
- [ ] 能打开并看懂 `outputs/` 里的两张图
- [ ] 能回答：MNIST 训练集多少张？每张图多大？标签是什么？
- [ ] 能回答：为什么说「图片在计算机里就是矩阵」？

## 🐛 常见问题

**Q：卡在 Downloading ... 很久不动 / 下载失败？**
A：数据源在国外，网络不稳定。① 多试几次；② 挂代理；③ 手动下载：从 `https://ossci-datasets.s3.amazonaws.com/mnist/` 下载 4 个 `.gz` 文件（`train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`），放进 `data/MNIST/raw/`，重新运行脚本即可（脚本检测到文件存在会跳过下载）。

**Q：运行后没有弹出图片窗口？**
A：本教程的脚本都采用「保存图片到 outputs/」的方式，不弹窗，方便在任何环境运行。自己打开 `outputs/` 文件夹看图即可。想弹窗的话，在脚本末尾加一行 `plt.show()`。

**Q：图片上文字是英文，能改成中文吗？**
A：能，但 matplotlib 默认字体不支持中文会显示方块，需要额外配置字体，教程里统一用英文标签避免麻烦。

## 📌 本步小结

1. AI 项目 = 用数据训练模型；MNIST 有 6 万训练 + 1 万测试张 28×28 灰度数字图
2. 图片在计算机里是数字矩阵，AI 就是从矩阵里找规律
3. torchvision 的 `datasets.MNIST` 一行代码搞定下载和加载

## 🔗 下一步

👉 [第 3 步 · 数据预处理](md/step-03-数据预处理.md)：把原始图片变成模型爱吃的「标准口粮」——张量和 DataLoader。
