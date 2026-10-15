# 第 6 步 · 评估模型

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐⭐☆☆ | 约 30 分钟 | [第 5 步](md/step-05-训练模型.md) | [code/06_evaluate.py](../code/06_evaluate.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 理解「准确率 97%」到底意味着什么，以及它的局限
- [ ] 读懂并生成混淆矩阵（Confusion Matrix）
- [ ] 计算每个数字类别各自的准确率
- [ ] 把模型认错的样本找出来，分析错因

## 📚 知识准备

### 准确率的局限：只看总分是不够的

第 5 步得到验证准确率约 97%，但「总分」掩盖了很多细节：

- 是每个数字都认识 97%，还是「7 全认识、3 全不认识」平均出来的 97%？
- 3% 的错题长什么样？是笔迹潦草还是标签标错？

就像考试只看总分不知道偏科。**评估模型要像医生体检，不能只看体重。**

### 混淆矩阵：一张表看清所有错法

混淆矩阵是一个 10×10 的表格：**行 = 真实标签，列 = 预测标签**。

```
                预测
          0    1    2   ...   9
真实 0 [ 968   0    2   ...   1 ]   ← 数字 0 的 980 张图：968 张认对，2 张被认成 2
     1 [   0 1120    1   ...   0 ]
     2 [   4   1 1005   ...   3 ]
    ...
     9 [   0   0    4   ... 998 ]
```

- **对角线**：每个类别认对的张数，越亮越好
- **非对角线**：认错的去向——比如「4 经常被认成 9」，一眼就能看到
- 每行之和 = 该类别真实样本数

### 精确率与召回率（先建立直觉，不背公式）

- **精确率（Precision）**：模型说「这是 3」时，有多少真的是 3 —— 信口开河少不少
- **召回率（Recall）**：真正的 3 里，模型抓到了多少 —— 漏网之鱼多不多

两个指标往往此消彼长。本步先学会**逐类准确率**（= 每类的召回率）和混淆矩阵，精确率的概念有个印象即可。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 6.1 运行评估脚本

**前提**：已完成第 5 步，`outputs/model_mlp.pth` 存在。

```bash
python code/06_evaluate.py
```

```python
"""
第 6 步 · 评估模型
运行方式：python code/06_evaluate.py
作用：加载训练好的模型，在测试集上生成混淆矩阵、逐类准确率，并展示认错的样本。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


class MLP(nn.Module):
    """与第 4 步完全相同的模型"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# 1. 加载测试集（与训练时完全相同的预处理）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 2. 加载训练好的模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
model.load_state_dict(torch.load(OUT_DIR / "model_mlp.pth", map_location=device))
model.eval()
print("模型加载成功")

# 3. 在测试集上收集全部预测结果
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc = (all_preds == all_labels).mean()
print(f"测试集准确率: {test_acc:.4f}")

# 4. 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix on test set (row=truth, col=prediction)")
plt.savefig(OUT_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("已保存混淆矩阵 -> outputs/06_confusion_matrix.png")

# 5. 逐类准确率：对角线 ÷ 该行总数
per_class = cm.diagonal() / cm.sum(axis=1)
for i in range(10):
    print(f"数字 {i}: 样本 {cm.sum(axis=1)[i]:5d} 张, 准确率 {per_class[i]:.4f}")
print(f"最难认的数字: {per_class.argmin()} ({per_class.min():.4f})")

# 6. 找出最多 20 张认错的样本并展示
errors = []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        for i in range(images.size(0)):
            if preds[i] != labels[i]:
                errors.append((images[i], labels[i].item(), preds[i].item()))
            if len(errors) >= 20:
                break
        if len(errors) >= 20:
            break

inv_norm = transforms.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,))  # 逆归一化用于显示
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for ax, (img, truth, pred) in zip(axes.flat, errors):
    ax.imshow(inv_norm(img).squeeze(), cmap="gray")
    ax.set_title(f"truth {truth} / pred {pred}", color="red", fontsize=10)
    ax.axis("off")
fig.suptitle("20 misclassified samples (truth / prediction)")
fig.tight_layout()
fig.savefig(OUT_DIR / "06_error_samples.png", dpi=150)
plt.close("all")
print("已保存错误样本 -> outputs/06_error_samples.png")
```

**预期输出**（真实运行结果，数值有小波动）：

```
模型加载成功
测试集准确率: 0.9771
已保存混淆矩阵 -> outputs/06_confusion_matrix.png
数字 0: 样本   980 张, 准确率 0.9908
数字 1: 样本  1135 张, 准确率 0.9921
数字 2: 样本  1032 张, 准确率 0.9767
数字 3: 样本  1010 张, 准确率 0.9772
数字 4: 样本   982 张, 准确率 0.9878
数字 5: 样本   892 张, 准确率 0.9641
数字 6: 样本   958 张, 准确率 0.9739
数字 7: 样本  1028 张, 准确率 0.9844
数字 8: 样本   974 张, 准确率 0.9671
数字 9: 样本  1009 张, 准确率 0.9534
最难认的数字: 9 (0.9534)
已保存错误样本 -> outputs/06_error_samples.png
```

> 💡 逐类准确率每次训练都会有小幅变化，「最难认的数字」通常在 **8、9、5、7** 之间轮换——因为它们互相之间长得最像。你运行的数值和这里不同是正常的，重要的是能看懂指标的含义。

### 6.2 逐段讲解

**① 加载模型三步曲**

```python
model = MLP()                                # 先建一个结构相同的空模型
model.load_state_dict(torch.load(...))       # 把第 5 步学到的参数装进去
model.eval()                                 # 评估模式
```

加载模型的铁律：**先有结构，再装参数**。`state_dict` 的详细原理第 7 步讲。

**② 收集全部预测**

跑完整个测试集，把 1 万个预测结果和真实标签分别存进两个列表，转成 numpy 数组方便统计。`argmax(dim=1)` 取每行最高分下标 = 预测数字。

**③ 混淆矩阵**

`confusion_matrix(真实, 预测)` 一行生成 10×10 矩阵；`ConfusionMatrixDisplay` 把它画成热力图。打开 `outputs/06_confusion_matrix.png`：对角线非常亮（认对的），少数暗点散布在对角线外（认错的）。

**④ 逐类准确率**

`cm.diagonal()` 取对角线（认对数），除以每行总和（该类真实样本数）。**观察你的模型最怕哪个数字**——常见答案是 8，因为 8 和 3、5、9 长得像。

**⑤ 错误样本分析**

遍历时把「预测 ≠ 真实」的样本连图带标签收集 20 张，画成 4×5 网格。打开 `outputs/06_error_samples.png` 观察：这些错题是不是真的很难？很多连人类都要犹豫一下（潦草、歪斜、断开），模型认错情有可原——这是**定性分析**，AI 工程师的日常。

### 6.3 小实验

1. 数一数错误样本图里，哪些错误「人类也会犯」，哪些「莫名其妙」
2. 把混淆矩阵里最大的非对角线格子找出来（比如「4→9」），猜测原因
3. 如果你在第 5 步训练了 20 个 epoch 的过拟合模型，用它跑本步脚本，对比混淆矩阵变化

## ✅ 完成自检

- [ ] 脚本跑通，测试准确率与第 5 步验证准确率接近（差 1 个百分点以内都正常）
- [ ] 能向别人解释混淆矩阵的行、列、对角线分别代表什么
- [ ] 能说出自己的模型最难认哪个数字、最容易和谁混淆
- [ ] 能说出「只看总准确率」有什么不足

## 🐛 常见问题

**Q：报错 `FileNotFoundError: outputs/model_mlp.pth`？**
A：还没训练过。先跑第 5 步 `python code/05_train.py`，再回来。

**Q：测试集准确率和验证集差了不少？**
A：差 1% 以内正常（数据不同）。差得多说明第 5 步在验证集上调过参数导致「验证集过拟合」——本教程没有这个操作，所以不会出现。

**Q：混淆矩阵的字太小看不清？**
A：把 `figsize` 调大，或在 `disp.plot(...)` 后加 `plt.tight_layout()`。图已经保存在 outputs/，可以放大查看。

**Q：为什么要 `bbox_inches="tight"`？**
A：防止标签被裁掉，让图片四周自动留白。

## 📌 本步小结

1. 总准确率是「总分」，混淆矩阵和逐类准确率才是「体检报告」，能暴露偏科
2. 混淆矩阵：行 = 真实、列 = 预测，对角线 = 认对数
3. 分析错误样本是定性评估，很多错题人类也会犯——评估模型也是在理解任务的难度

## 🔗 下一步

👉 [第 7 步 · 保存与加载模型](md/step-07-保存与加载模型.md)：搞懂 `state_dict` 到底存了什么，并写出「输入一张图、输出一个答案」的推理函数。
