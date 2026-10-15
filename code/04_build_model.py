"""
第 4 步 · 构建第一个神经网络
运行方式：python code/04_build_model.py
作用：定义 MLP 模型，打印结构、参数量，并用随机输入测试前向传播。
"""
import torch
from torch import nn


class MLP(nn.Module):
    """多层感知机：输入 784 维 -> 128 -> 64 -> 输出 10 维"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 把 [N, 1, 28, 28] 拉平成 [N, 784]
        self.fc1 = nn.Linear(784, 128)       # 全连接层 1：784 -> 128
        self.relu1 = nn.ReLU()               # 激活函数
        self.fc2 = nn.Linear(128, 64)        # 全连接层 2：128 -> 64
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)         # 输出层：64 -> 10 个分数（logits）

    def forward(self, x):
        """前向传播：描述数据从输入到输出的完整旅程"""
        x = self.flatten(x)                  # [N,1,28,28] -> [N,784]
        x = self.relu1(self.fc1(x))          # [N,784] -> [N,128] -> ReLU
        x = self.relu2(self.fc2(x))          # [N,128] -> [N,64] -> ReLU
        x = self.fc3(x)                      # [N,64] -> [N,10]，注意：不加 softmax
        return x


model = MLP()
print(model)

# 统计参数量
total = sum(p.numel() for p in model.parameters())
print("可训练参数量:", total)

# 用随机 batch 测试前向传播是否能跑通
dummy = torch.randn(2, 1, 28, 28)            # 假装是 2 张 28x28 的灰度图
out = model(dummy)
print("输入形状:", dummy.shape, "-> 输出形状:", out.shape)
print("第一张图得到的 10 个分数:", [round(v, 2) for v in out[0].tolist()])
