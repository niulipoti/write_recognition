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
