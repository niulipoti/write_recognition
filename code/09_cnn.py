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
