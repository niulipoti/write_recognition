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
