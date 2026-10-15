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
