"""
第 7 步 · 保存与加载模型 + 单张图片推理
运行方式：python code/07_save_load.py
作用：演示 state_dict 的保存/加载，并对一张测试图做推理，画出概率条形图。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

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


# 1. 加载模型（结构 + 参数，缺一不可）
model = MLP()
model.load_state_dict(torch.load(OUT_DIR / "model_mlp.pth", map_location="cpu"))
model.eval()
print("模型加载成功")

# 2. 看一眼 state_dict 里存了什么
for name, tensor in model.state_dict().items():
    print(f"{name:<14s} 形状 {list(tensor.shape)}")

# 3. 定义推理函数：输入一张图，输出答案和置信度
def predict(model, image_tensor):
    """输入 [1, 28, 28] 张量，返回 (预测类别, 10 个类别的概率)"""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))      # 补上 batch 维：-> [1,1,28,28]
        probs = torch.softmax(logits, dim=1).squeeze() # 分数 -> 概率
    return probs.argmax().item(), probs

# 4. 拿测试集第 7 张图试试
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
image, label = test_set[7]

pred, probs = predict(model, image)
print(f"真实标签: {label} | 模型预测: {pred} | 置信度: {probs[pred]:.2%}")

# 5. 画图：左边原图，右边概率条形图
inv_norm = transforms.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.imshow(inv_norm(image).squeeze(), cmap="gray")
ax1.set_title(f"input image (true label: {label})")
ax1.axis("off")
ax2.bar(range(10), probs.numpy(), color="#4C72B0")
ax2.set_xticks(range(10))
ax2.set_xlabel("digit")
ax2.set_ylabel("probability")
ax2.set_title(f"prediction: {pred}")
fig.tight_layout()
fig.savefig(OUT_DIR / "07_prediction_probs.png", dpi=150)
plt.close("all")
print("已保存推理结果图 -> outputs/07_prediction_probs.png")
