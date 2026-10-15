# 第 7 步 · 保存与加载模型

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐☆☆☆ | 约 25 分钟 | [第 5 步](md/step-05-训练模型.md) | [code/07_save_load.py](../code/07_save_load.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 理解 `state_dict` 里存的到底是什么
- [ ] 正确保存与加载模型，避开常见坑
- [ ] 写出「输入一张图、输出答案和置信度」的推理函数
- [ ] 理解训练（training）与推理（inference）的区别

## 📚 知识准备

### state_dict：模型的「大脑记忆」

训练 5 分钟后，模型从随机数变成了「会认数字的大脑」。这份「记忆」就是 10.9 万个参数的具体数值，在 PyTorch 里打包成一个字典对象，叫 **state_dict**：

```
{
  'fc1.weight': 形状 [128, 784] 的张量,   ← 第 1 层的 100,352 个权重
  'fc1.bias':   形状 [128] 的张量,        ← 第 1 层的 128 个偏置
  'fc2.weight': 形状 [64, 128] 的张量,
  'fc2.bias':   形状 [64] 的张量,
  'fc3.weight': 形状 [64, 10] 的张量,
  'fc3.bias':   形状 [10] 的张量,
}
```

训练花了 5 分钟，而**保存/加载只要 0.1 秒**——因为要存的只是这些数字，模型结构（几层、多宽）是代码里写死的。

### 两种保存方式：记住只推荐第一种

| 方式 | 代码 | 特点 |
| :--- | :--- | :--- |
| ✅ 保存 state_dict | `torch.save(model.state_dict(), path)` | 只存参数。轻量、灵活、跨环境稳定。**推荐** |
| ⚠️ 保存整个模型 | `torch.save(model, path)` | 连代码结构一起存。换环境、改代码后极易报错。**不推荐** |

本教程从第 5 步起用的都是第一种。

### 训练 vs 推理：两个完全不同的场景

| | 训练（training） | 推理（inference） |
| :--- | :--- | :--- |
| 目的 | 学知识（更新参数） | 用知识（参数冻结） |
| 数据 | 成批（batch） | 常常单张 |
| 要算梯度吗 | 要（backward） | **不要**（no_grad，快且省内存） |
| 模式 | `model.train()` | `model.eval()` |

推理的正确姿势：`model.eval()` + `torch.no_grad()`，两者都要有。

### softmax：把分数翻译成概率

第 4 步留过一个悬念：模型输出的是 10 个「分数」，不是概率。softmax 就是翻译官：

```
概率_i = e^(分数_i) / (e^(分数_1) + e^(分数_2) + ... + e^(分数_10))
```

结果 10 个数全部在 0~1 之间且**加起来等于 1**。分数最高 → 概率最大 → 就是模型的答案，概率值即「置信度」。置信度 99% 的「7」和 30% 的「7」完全是两种可信度。

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 7.1 运行脚本

**前提**：`outputs/model_mlp.pth` 存在（第 5 步已生成）。

```bash
python code/07_save_load.py
```

```python
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
```

**预期输出**：

```
模型加载成功
fc1.weight     形状 [128, 784]
fc1.bias       形状 [128]
fc2.weight     形状 [64, 128]
fc2.bias       形状 [64]
fc3.weight     形状 [64, 10]
fc3.bias       形状 [10]
真实标签: 9 | 模型预测: 9 | 置信度: 99.98%
已保存推理结果图 -> outputs/07_prediction_probs.png
```

打开 `outputs/07_prediction_probs.png`：左边是输入的图片，右边 10 根柱子——模型几乎把全部概率押在了一个数字上。

### 7.2 逐段讲解

**① 加载三步：结构 → 参数 → 模式**

```python
model = MLP()                                            # 先搭好结构
model.load_state_dict(torch.load(path, map_location="cpu"))  # 再装入参数
model.eval()                                             # 最后切到评估模式
```

- `map_location="cpu"`：如果模型是在 GPU 上训练的，这行能把它搬到 CPU 上加载，**加上了就永远不会错**
- 顺序不能反：结构先于参数，就像先造房子再搬家具

**② 推理函数**

```python
logits = model(image_tensor.unsqueeze(0))   # 单张 [1,28,28] -> [1,1,28,28]
```

`unsqueeze(0)` 在最前面补一维——模型是按 batch 设计的，单张图也要假装成「一个只有 1 张图的 batch」。然后 softmax 变概率、argmax 取答案。这个函数就是后续第 8 步交互 Demo 的核心。

**③ 置信度的意义**：打开概率图，如果柱子「一柱擎天」，说明模型很确定；如果几根柱子差不多高，说明模型在犹豫。**输出置信度而不只是答案，是成熟 AI 系统的做法**——让使用方知道何时该怀疑结果。

### 7.3 小实验

1. 把 `test_set[7]` 换成 `test_set[0]`、`test_set[42]` 等不同下标，看看模型对哪些图最自信
2. 把 softmax 去掉（直接看 logits），观察输出的值域变化
3. 故意把加载路径写错，看看报什么错，学会从报错里识别「文件不存在」

## ✅ 完成自检

- [ ] 脚本跑通，state_dict 打印的 6 个张量形状与文档一致
- [ ] 能说出 state_dict 是什么、为什么保存它而不是整个模型
- [ ] 能说出训练和推理在「梯度、batch、模式」三个方面的区别
- [ ] 能默写推理函数的两句关键代码：`unsqueeze(0)` 和 `torch.no_grad()`
- [ ] 能解释为什么概率条形的 10 根柱子加起来等于 1

## 🐛 常见问题

**Q：报错 `Missing key(s) in state_dict: "fc1.weight"...`？**
A：加载的模型结构和保存时不一致（比如你改了隐藏层宽度）。保存和加载必须用**完全相同**的结构代码。

**Q：报错 `Unexpected key(s)`？**
A：反向情况——保存的模型比现在的结构多东西。同样是结构不一致。

**Q：`torch.load` 警告 FutureWarning？**
A：新版 PyTorch 建议 `torch.load(path, weights_only=True)` 更安全（防止恶意文件执行代码）。加不加都不影响本教程运行，了解即可。

**Q：`map_location="cpu"` 必须加吗？**
A：在本教程里模型本来就在 CPU 上，不加也能跑；但加上后代码在任何环境（包括别人 GPU 训练的模型）都不会出错，属于「防御性编程」的好习惯。

## 📌 本步小结

1. state_dict = 模型学到的全部参数数值；保存它轻量又稳定，是业界标准做法
2. 加载三步骤：搭结构 → 装参数 → `eval()`；推理两件套：`eval()` + `no_grad()`
3. softmax 把分数变成和为 1 的概率，argmax 取其最大者即答案

## 🔗 下一步

👉 [第 8 步 · 识别自己的手写数字](md/step-08-识别自己的手写数字.md)：把推理函数包装成一个可以玩的程序——在屏幕上用鼠标写数字，让 AI 猜。
