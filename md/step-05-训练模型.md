# 第 5 步 · 训练模型

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐⭐⭐☆ | 约 40 分钟 | [第 4 步](md/step-04-构建模型.md) | [code/05_train.py](../code/05_train.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 理解损失函数：模型「错得有多离谱」的尺子
- [ ] 理解梯度下降：模型「下山找最低点」的过程
- [ ] 写出完整的训练循环：前向传播 → 算损失 → 反向传播 → 更新参数
- [ ] 看懂学习曲线，判断模型是否在正常学习

## 📚 知识准备

### 损失函数（Loss Function）：一把尺子

训练的目标是让模型预测得准。怎么算「准」？用损失函数量一量**预测和正确答案差多远**——差值就是损失（loss），越小越好。

本教程用 **交叉熵损失（CrossEntropyLoss）**，分类任务的事实标准：

- 模型对 10 个数字各打一个分数（logits）
- 交叉熵内部先做 softmax 把分数变概率，再算「真实答案对应的概率」取负对数
- 模型信心满满答对 → 损失接近 0；信心十足答错 → 损失巨大，会被狠狠惩罚

```
真实标签 = 3
模型输出概率 = [0.01, 0.02, 0.05, 0.80, 0.02, 0.03, 0.02, 0.02, 0.01, 0.02]
                            ↑ 数字 3 的概率 0.80
损失 = -ln(0.80) ≈ 0.22   （小，很好）
```

### 梯度下降：蒙眼下山

把损失想象成一座山的高度，模型参数就是你的位置，目标是**走到最低点**：

1. 站在山坡上，感受脚下往哪个方向最陡（求**梯度**）
2. 朝下坡方向迈一小步（**更新参数**，步长 = 学习率 × 梯度）
3. 重复 1~2，直到谷底

**反向传播（backward）** 就是 PyTorch 自动完成的「感受坡度」——第 3 步说过 PyTorch 的核心能力是自动求导，`loss.backward()` 一行代码就求出了所有 10.9 万个参数各自的梯度。

### 优化器（Optimizer）：迈步的教练

知道方向后，怎么迈步、迈多大，由优化器决定。本教程用 **Adam**，目前最常用的优化器：自动给每个参数调整步长，收敛快、对新手友好。经典的选择是 SGD（随机梯度下降），简单但慢。两者都值得认识。

- **学习率（learning rate，lr）**：步长基数，本教程取 0.001。太大 → 迈过头来回震荡；太小 → 走到天黑还没到谷底

### epoch、batch、iteration 三个时间单位

- **batch**：一批数据（64 张）
- **iteration**：训练一个 batch 的过程（782 个 iteration = 看完一遍训练集）
- **epoch**：把**整个训练集完整过一遍**（= 782 个 iteration），本轮循环 5 个 epoch

### 过拟合：死记硬背的陷阱

如果只盯着训练集刷题，模型可能把题背下来，遇到新题就傻眼——这叫**过拟合**。对策：每个 epoch 结束后在**验证集**上打分。训练损失降、验证损失也降 → 健康；验证损失开始上升 → 过拟合前兆，该停了。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 5.1 运行训练脚本

```bash
python code/05_train.py
```

CPU 上 5 个 epoch 大约 2~5 分钟，请耐心等待。

```python
"""
第 5 步 · 训练模型
运行方式：python code/05_train.py
作用：完整训练循环：损失函数 + 优化器 + 反向传播，画学习曲线，保存模型。
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
OUT_DIR.mkdir(exist_ok=True)


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


# 1. 数据准备（与第 3 步完全一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
train_set, val_set = random_split(
    train_set, [50000, 10000],
    generator=torch.Generator().manual_seed(42),
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 2. 模型、损失函数、优化器：训练三件套
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()                    # 内置 softmax 的交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam 优化器，学习率 0.001

EPOCHS = 5
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


def evaluate(model, loader):
    """在给定数据上评估损失和准确率（不更新参数）"""
    model.eval()                                   # 切换到评估模式（关掉 Dropout 等训练特性）
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():                          # 不计算梯度：评估不需要，省内存省时间
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# 3. 训练循环：整个教程的心脏
for epoch in range(1, EPOCHS + 1):
    model.train()                                  # 切回训练模式
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:            # 782 个 batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # ① 清空上一步的梯度
        logits = model(images)                     # ② 前向传播：得到预测分数
        loss = loss_fn(logits, labels)             # ③ 计算损失
        loss.backward()                            # ④ 反向传播：自动求每个参数的梯度
        optimizer.step()                           # ⑤ 更新参数：迈出下山的一步
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

# 4. 画学习曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history["train_loss"], marker="o", label="train")
ax1.plot(history["val_loss"], marker="o", label="val")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
ax1.set_title("Loss curve"); ax1.legend()
ax2.plot(history["train_acc"], marker="o", label="train")
ax2.plot(history["val_acc"], marker="o", label="val")
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
ax2.set_title("Accuracy curve"); ax2.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "05_learning_curves.png", dpi=150)
print("已保存学习曲线 -> outputs/05_learning_curves.png")

# 5. 保存模型权重（第 7 步详解保存原理）
torch.save(model.state_dict(), OUT_DIR / "model_mlp.pth")
print("已保存模型权重 -> outputs/model_mlp.pth")
```

**预期输出**（真实运行结果，数值会有小波动）：

```
Epoch 1/5 | train loss 0.2878 acc 0.9150 | val loss 0.1584 acc 0.9533
Epoch 2/5 | train loss 0.1202 acc 0.9634 | val loss 0.1243 acc 0.9601
Epoch 3/5 | train loss 0.0841 acc 0.9739 | val loss 0.1365 acc 0.9595
Epoch 4/5 | train loss 0.0619 acc 0.9801 | val loss 0.1001 acc 0.9683
Epoch 5/5 | train loss 0.0497 acc 0.9838 | val loss 0.0931 acc 0.9716
已保存学习曲线 -> outputs/05_learning_curves.png
已保存模型权重 -> outputs/model_mlp.pth
```

验证集准确率约 **97%**——第 4 步还全是随机数的模型，训练 5 分钟后就认出了 97% 的手写数字（测试集上约 97.7%）。

### 5.2 逐段讲解：训练循环五步曲

训练的核心就是下面五行，请背下来——**所有深度学习框架的训练循环都是这个套路**：

```python
optimizer.zero_grad()   # ① 清空梯度（否则会累加上一次的）
logits = model(images)  # ② 前向传播：模型给出预测
loss = loss_fn(logits, labels)  # ③ 算损失：预测与答案差多远
loss.backward()         # ④ 反向传播：自动求梯度
optimizer.step()        # ⑤ 更新参数：朝下坡方向迈一步
```

**评估函数 evaluate 的两个细节**：

- `model.eval()` / `model.train()`：切换模型模式。本模型没有 Dropout/BatchNorm，暂时不影响结果，但这是**必须养成的好习惯**
- `torch.no_grad()`：评估时关掉梯度计算，速度快、省内存

**指标是怎么算的**：

- `logits.argmax(dim=1)`：取每行 10 个分数里最大的那个的下标 = 预测的数字
- 与真实标签 `labels` 逐位比较，求平均得准确率
- 损失按样本数加权平均：`loss.item() * labels.size(0)` 累加后除以总数

### 5.3 看懂学习曲线

打开 `outputs/05_learning_curves.png`：

- **左图（loss）**：两条线都快速下降 → 模型在快速学习；训练损失低于验证损失 → 正常现象（验证数据没见过）
- **右图（accuracy）**：快速爬升到 97% 左右进入平台期 → 模型基本学会了
- 若验证损失在第 4~5 个 epoch 出现抬头迹象，就是轻微过拟合前兆——但本轮无伤大雅

### 5.4 小实验（重点！）

改代码重跑，观察曲线变化，每个实验都值得做：

| 实验 | 改动 | 预期现象 |
| :--- | :--- | :--- |
| 1 | `EPOCHS = 5` 改成 `20` | 后期验证损失抬头 → 亲眼看到过拟合 |
| 2 | `lr=1e-3` 改成 `lr=1.0` | 损失乱跳不收敛 → 学习率太大 |
| 3 | `lr=1e-3` 改成 `lr=1e-5` | 损失几乎不动 → 学习率太小 |
| 4 | 把 Adam 换成 `torch.optim.SGD(model.parameters(), lr=0.01)` | 收敛明显变慢 |
| 5 | `batch_size` 改成 256 | 每 epoch 更快，收敛略慢 |

## ✅ 完成自检

- [ ] 脚本跑通，最终验证准确率 ≥ 95%
- [ ] 能默写训练循环五步（zero_grad → forward → loss → backward → step）
- [ ] 能用「下山」的比喻向别人讲清梯度下降
- [ ] 能说出 epoch、batch、iteration 的区别
- [ ] 至少完成 1 个小实验，并能解释观察到的现象
- [ ] 能在 `outputs/` 找到学习曲线和 `model_mlp.pth`

## 🐛 常见问题

**Q：训练时 CPU 占用 100%、风扇狂转，正常吗？**
A：完全正常，训练就是满负荷计算。笔记本建议插电源跑。

**Q：loss 变成 nan 了？**
A：学习率太大，数值爆炸。把 lr 调回 1e-3 或更小。

**Q：准确率停在 10% 左右不动？**
A：10 类随机猜的准确率正好是 10%。可能：数据没归一化、标签没对上、学习率太小。对照第 3 步检查 transform。

**Q：为什么训练损失比验证损失还低？**
A：正常。模型见过训练数据，还边做边改（更新参数）；验证集是「闭卷考」，分数低一点天经地义。

**Q：重新跑一遍，结果和文档不完全一样？**
A：正常。训练有随机性（参数初始化、数据顺序），准确率在 96%~98% 波动都健康。想完全复现需要固定全部随机种子（进阶技巧，先了解概念即可）。

## 📌 本步小结

1. 训练 = 反复执行五步曲：清梯度 → 前向 → 算损失 → 反向求梯度 → 优化器更新参数
2. 交叉熵损失量「错得多离谱」，Adam 优化器负责高效下山，学习率决定步长
3. 每个 epoch 后看验证集表现：损失降、准确率升，曲线是判断模型健康度的体检报告

## 🔗 下一步

👉 [第 6 步 · 评估模型](md/step-06-评估模型.md)：97% 之外还有 3% 认错了什么？用混淆矩阵和错误样本给模型做一次深度体检。
