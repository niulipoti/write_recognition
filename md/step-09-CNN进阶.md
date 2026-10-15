# 第 9 步 · CNN 进阶

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐⭐⭐☆ | 约 40 分钟 | [第 8 步](md/step-08-识别自己的手写数字.md) | [code/09_cnn.py](../code/09_cnn.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 用直觉理解卷积和池化：为什么 CNN 天生适合图像
- [ ] 搭建一个卷积神经网络（CNN）并训练到 98% 以上准确率
- [ ] 对比 MLP 与 CNN 的参数量和准确率，体会「结构决定上限」
- [ ] 认识「更少参数、更好效果」的深度学习设计哲学

## 📚 知识准备

### MLP 看图像有什么问题？

第 4 步的 MLP 把图片**摊平成 784 个互不相关的数字**，等于把一张图撕成像素碎片再研究。它损失了最重要的信息：**空间结构**——哪个像素在哪个像素旁边。数字「7」的一横一竖只要整体平移两格，784 个数字全变了，MLP 完全没法利用「平移不变性」。

### 卷积（Convolution）：带着窗口找特征

卷积层不摊平图片，而是拿一个小的**卷积核（kernel）**在图上滑动，像拿着放大镜逐格扫描：

```
原始图 5x5                    3x3 卷积核（检测边缘）      输出特征图
 0 0 0 0 0
 0 1 1 1 0       滑动窗口      [-1 -1 -1
 0 1 1 1 0   ──────────────►    -1  8 -1 ]   ──►   检测到的边缘
 0 1 1 1 0    逐格相乘再相加     [-1 -1 -1]
 0 0 0 0 0
```

- 卷积核里的数（-1、8、-1…）就是**要训练的参数**，一个核学会一种特征：横线、竖线、拐角…
- **滑动 + 共享**：同一个核扫遍全图，所以「无论 7 写在哪，都能认出那一横」——这就是平移不变性
- 本步用 8 个核 → 得到 8 张特征图（feature map），第二层再用 16 个核从特征图里找更复杂的特征

### 池化（Pooling）：给特征图瘦身

池化层把特征图每 2×2 格压缩成 1 格，取最大值（MaxPool）：

- 尺寸减半：28×28 → 14×14 → 7×7，计算量骤减
- 只留「最强信号」：某处只要有笔画特征，取最大就能留住
- 进一步强化平移不变性

### 本步的 CNN 结构

```
输入 [1, 28, 28]
 → Conv(1→8 个核) + ReLU        [8, 28, 28]   找低级特征：边缘、笔画
 → MaxPool(2)                   [8, 14, 14]   瘦身
 → Conv(8→16 个核) + ReLU       [16, 14, 14]  找高级特征：笔画组合
 → MaxPool(2)                   [16, 7, 7]    再瘦身
 → Flatten                      [784]
 → Linear(784 → 64) + ReLU      分类头
 → Linear(64 → 10)              输出 10 个分数
```

规律：**卷积负责「看」，全连接负责「判断」**——CNN 先把图像提炼成特征图，最后再摊平交给分类头。

### 为什么 CNN 参数更少却更强？

| 模型 | 参数量 | 测试准确率（本教程实测区间） |
| :--- | :--- | :--- |
| MLP | 109,386 | 约 97.0% ~ 98.0% |
| CNN | **约 52,138** | 约 **98.0% ~ 99.0%** |

CNN 参数少一半，是因为卷积核**共享参数**（一个 3×3 核只有 9 个数，扫遍全图）；效果更好，是因为它**利用了空间结构**。这背后的哲学：**把人类对问题的先验知识（图像有局部结构）做进模型结构里，模型就不用自己从头摸索**。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 9.1 运行 CNN 脚本

```bash
python code/09_cnn.py
```

CPU 上约 5~10 分钟，比 MLP 慢（卷积计算量大），耐心等待。

```python
"""
第 9 步 · 卷积神经网络 CNN
运行方式：python code/09_cnn.py
作用：训练 CNN，与 MLP 对比参数量和测试准确率。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


class CNN(nn.Module):
    """两个卷积块 + 全连接分类头"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # [N,1,28,28] -> [N,8,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [N,8,14,14]
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> [N,16,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [N,16,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),                   # -> [N,64]
            nn.ReLU(),
            nn.Linear(64, 10),                           # -> [N,10]
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """与第 4 步完全相同的模型，用于对比"""
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


# 1. 数据准备（与第 5 步完全一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
train_set, val_set = random_split(
    train_set, [50000, 10000],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# 2. 训练（与第 5 步完全相同的五步曲——这就是框架的威力：换个模型，套路不变）
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_loss, train_acc = running_loss / total, correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")

# 3. 保存 CNN 与学习曲线
torch.save(model.state_dict(), OUT_DIR / "model_cnn.pth")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history["train_loss"], marker="o", label="train")
ax1.plot(history["val_loss"], marker="o", label="val")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
ax1.set_title("CNN loss curve"); ax1.legend()
ax2.plot(history["train_acc"], marker="o", label="train")
ax2.plot(history["val_acc"], marker="o", label="val")
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
ax2.set_title("CNN accuracy curve"); ax2.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "09_cnn_curves.png", dpi=150)
print("已保存 CNN 学习曲线 -> outputs/09_cnn_curves.png")
print("已保存 CNN 模型 -> outputs/model_cnn.pth")


# 4. 终极对比：MLP vs CNN
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def test_accuracy(m):
    m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            correct += (m(images).argmax(dim=1).cpu() == labels).sum().item()
            total += labels.size(0)
    return correct / total

mlp = MLP().to(device)
mlp.load_state_dict(torch.load(OUT_DIR / "model_mlp.pth", map_location=device))
cnn_acc = test_accuracy(model)

print("\n===== 终极对比 =====")
print(f"MLP : 参数量 {count_params(mlp):>7,} | 测试准确率 {test_accuracy(mlp):.4f}")
print(f"CNN : 参数量 {count_params(model):>7,} | 测试准确率 {cnn_acc:.4f}")
```

**预期输出**（真实运行结果，准确率有 ±0.5% 的正常波动）：

```
Epoch 1/5 | train loss 0.2656 acc 0.9213 | val loss 0.1211 acc 0.9610
Epoch 2/5 | train loss 0.0788 acc 0.9764 | val loss 0.0751 acc 0.9775
Epoch 3/5 | train loss 0.0558 acc 0.9827 | val loss 0.0605 acc 0.9810
Epoch 4/5 | train loss 0.0444 acc 0.9860 | val loss 0.0554 acc 0.9829
Epoch 5/5 | train loss 0.0362 acc 0.9883 | val loss 0.0636 acc 0.9801
已保存 CNN 学习曲线 -> outputs/09_cnn_curves.png
已保存 CNN 模型 -> outputs/model_cnn.pth

===== 终极对比 =====
MLP : 参数量 109,386 | 测试准确率 0.9771
CNN : 参数量  52,138 | 测试准确率 0.9821
```

注意：CNN 只训练到**第 2 个 epoch** 验证准确率就达到 97.75%，已经追平 MLP 训练 5 轮的水平（97.16%）；而它的参数量只有 MLP 的**一半**——结构优势一目了然。

> 💡 关于「99%」：网络上常看到 MNIST 99% 以上的成绩，那通常来自更多 epoch、数据增强或更深的网络。本教程用最朴素的结构跑 5 个 epoch，**98.2% 就是正常且优秀的结果**。与其纠结零点几个百分点，不如先把流程和原理吃透——想冲 99%，看本步的「小实验」和第 10 步的挑战任务。

### 9.2 逐段讲解

**① Conv2d 的参数**

```python
nn.Conv2d(1, 8, kernel_size=3, padding=1)
```

- `1`：输入通道数（灰度图 1 通道）
- `8`：输出通道数 = 卷积核个数 = 特征图张数
- `kernel_size=3`：3×3 的卷积核
- `padding=1`：边缘补一圈 0，让输出尺寸和输入保持一致（28 不变）

**② 尺寸变化的规律**

`MaxPool2d(2)` 每 2×2 取最大，尺寸减半。CNN 里数据流形状变化全写在注释里了，跟着注释走一遍：28 → 14 → 7，通道 1 → 8 → 16。

**③ 训练循环原封不动**

对比第 5 步：数据准备、五步曲、评估函数**一行未改**，只换了 `model`。这就是框架抽象的威力——「训练套路」与「模型结构」解耦，以后你学任何新模型（Transformer 等）都只需换模型类。

**④ 对比实验的设计**

`count_params` 和 `test_accuracy` 两个小函数把「模型体检」抽象出来，公平对比两个模型。注意 MLP 用第 5 步训练的权重，CNN 用本步的权重。

### 9.3 小实验

1. 卷积核数 8/16 改成 32/64：准确率涨了多少？训练时间翻了几倍？（CPU 上请耐心）
2. 去掉两个池化层（28×28 直接 Flatten）：参数量爆炸到多少？还敢训练吗？
3. 把 CNN 的 `padding=1` 去掉，观察尺寸变化和报错，理解 padding 的作用
4. 用 `model_cnn.pth` 替换第 8 步脚本里加载的权重文件，画板识别的正确率会更高——试试看！

## ✅ 完成自检

- [ ] 脚本跑通，CNN 测试准确率 ≥ 98%
- [ ] 能用「滑动的放大镜」向别人解释卷积
- [ ] 能说出池化的两个作用（瘦身、保留最强信号）
- [ ] 能画出本步 CNN 的结构草图，标出每步的通道数和尺寸
- [ ] 能解释「为什么 CNN 参数更少却更准」

## 🐛 常见问题

**Q：CNN 训练比 MLP 慢很多，正常吗？**
A：正常。卷积是滑动计算，运算量比矩阵乘法大。CPU 上 5 个 epoch 约 5~10 分钟都正常。

**Q：kernel_size 为什么是 3，不是 5 或 7？**
A：3×3 是实践中最常用的尺寸：感受野够用、参数少、可以堆多层来扩大视野。试试改成 5 观察差异。

**Q：准确率没到 98%？**
A：① 训练随机性，重跑一次；② 检查 `EPOCHS=5` 是否被改小；③ 确认用了 `padding=1`（没有 padding 会损失边缘信息）。

**Q：为什么验证准确率会比训练准确率低？**
A：老问题——验证集是「闭卷考」。CNN 参数少，泛化更好，两者差距通常比 MLP 还小。

## 📌 本步小结

1. 卷积 = 滑动窗口找特征，核共享参数带来平移不变性；池化 = 瘦身 + 保留最强信号
2. CNN = 卷积「看」特征 + 全连接「判」类别；本步结构：8 核 → 池化 → 16 核 → 池化 → 64 → 10
3. 对比实验的结论：**结构决定上限**。CNN 用 MLP 一半的参数，拿到 98%+ 的准确率——把先验知识做进结构，是深度学习的核心设计哲学

## 🔗 下一步

👉 [第 10 步 · 总结与学习路线](md/step-10-总结与学习路线.md)：回顾前面的收获，拿到下一步的学习地图和挑战任务。
