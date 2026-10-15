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
