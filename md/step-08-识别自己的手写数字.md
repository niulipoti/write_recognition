# 第 8 步 · 识别自己的手写数字

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐⭐☆☆ | 约 30 分钟 | [第 7 步](md/step-07-保存与加载模型.md) | [code/08_predict_own_digit.py](../code/08_predict_own_digit.py) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 理解「训练时数据长什么样，推理时数据就必须长什么样」这条铁律
- [ ] 把任意一张图片（照片、截图、自己画的）变成模型能吃的 28×28 张量
- [ ] 运行画板程序：用鼠标写数字，让 AI 猜
- [ ] 向家人朋友展示你的第一个 AI 应用 🎉

## 📚 知识准备

### 数据分布一致性：AI 工程最重要的一条铁律

模型在第 3 步「吃」的是这样的数据：**黑底白字、28×28、归一化后的张量**。推理时喂给它的任何东西，都必须长得一模一样。就像你教一个人认「红底白字的车牌」，他自然认不出「白底红字」的。

常见的不一致：

| 不一致点 | 后果 |
| :--- | :--- |
| 你写的是白底黑字，MNIST 是黑底白字 | 准确率暴跌（本步代码会自动处理：检测底色并反转） |
| 图片尺寸不是 28×28 | 报错或变形（本步代码会自动缩放） |
| 彩色图片 | 形状不对（本步代码会自动转灰度） |
| 忘了归一化 | 输入分布偏移，准确率下降 |

**所以推理前的「预处理管线」必须和训练时一一对应**：灰度 → 缩放 → 转张量 → 归一化。

### 本步的三种用法

| 命令 | 作用 |
| :--- | :--- |
| `python code/08_predict_own_digit.py` | 打开画板，鼠标写数字，按键识别 |
| `python code/08_predict_own_digit.py 图片路径` | 识别一张现成的图片文件 |
| `python code/08_predict_own_digit.py --selftest` | 取一张真实 MNIST 图片，验证整条推理管线（无需显示器交互） |

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 8.1 先自测：用一张真实图片验证整条管线

**前提**：`outputs/model_mlp.pth` 存在（第 5 步生成）。

```bash
python code/08_predict_own_digit.py --selftest
```

**预期输出**（确定性结果：同一个模型文件每次运行都一样；重新训练模型后置信度会在 98%~100% 间微小浮动）：

```
已保存一张真实测试图（真实标签 7）-> outputs/08_selftest_digit.png
识别结果: 7 | 置信度: 99.94%
0: 0.000
1: 0.000
2: 0.000
3: 0.000
4: 0.000
5: 0.000
6: 0.000
7: 0.999
8: 0.000
9: 0.000
管线自检通过：图片文件 -> 识别的整条流程正确
```

自测做的事情：从 MNIST 测试集里取出一张**真实的**手写数字图片（这张图的真实标签是 7），存成 PNG 文件，再让它走完整条「图片文件 → 灰度 → 缩放到 28×28 → 归一化 → 模型预测」的推理管线，最后把预测结果与真实标签核对。

**为什么这样设计自测？** 因为输入与训练数据同分布，模型理应有极高的置信度（通常在 98% 以上）。如果这一步结果不对，就说明预处理管线写错了——这比「随口报个数字」可靠得多，也是工程上验证数据管线的标准做法。

> 💡 想亲眼看到「风格差异」的威力？用画板（8.3）故意写一个**笔直的印刷体 7**，你会看到置信度掉到 60% 左右，第二候选常常是「2」。这不是 bug——**模型只认它训练时见过的风格**，这正是第 3 步讲的「数据分布一致性」，也是真实 AI 项目里最容易被忽视、代价最大的坑。

### 8.2 再试图片文件

随便找一张手写数字照片（手机拍的计算器、账单都行），或直接用自测保存的那张 `outputs/08_selftest_digit.png`：

```bash
python code/08_predict_own_digit.py outputs/08_selftest_digit.png
```

### 8.3 画板模式：亲手写一个

```bash
python code/08_predict_own_digit.py
```

会弹出一个「黑底画板」窗口：

- **按住鼠标左键拖动** = 写字（写大一点，尽量居中，模仿 MNIST 的写法）
- **按 `s`** = 保存你写的字并让模型识别，结果打印在终端、显示在窗口标题
- **按 `c`** = 清空画板重写
- **按 `q`** = 退出

多试几个数字，看看哪些数字 AI 最容易猜错——大概率和你第 6 步混淆矩阵里看到的「难认数字」一致。你写的字被保存到 `outputs/08_my_digit.png`。

### 8.4 完整代码

```python
"""
第 8 步 · 识别你自己的手写数字
用法：
  python code/08_predict_own_digit.py                # 打开画板，用鼠标写数字
  python code/08_predict_own_digit.py 图片路径        # 直接识别一张图片文件
  python code/08_predict_own_digit.py --selftest     # 取一张真实 MNIST 图片验证整条管线
画板按键：鼠标拖动 = 写字 | s = 保存并识别 | c = 清空 | q = 退出
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

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


def load_model():
    """加载第 5 步训练好的模型"""
    model = MLP()
    model.load_state_dict(torch.load(OUT_DIR / "model_mlp.pth", map_location="cpu"))
    model.eval()
    return model


def array_to_input(arr):
    """
    把任意 28x28 的 0~255 uint8 灰度数组变成模型能吃的张量。
    白底黑字、黑底白字都能自动处理，与训练时的预处理严格一致。
    """
    x = arr.astype(np.float32) / 255.0                 # [0,1]
    if x.mean() > 0.5:                                 # 背景偏亮 -> 白底黑字，反转成黑底白字
        x = 1.0 - x
    x = np.clip((x - 0.3) / 0.7, 0.0, 1.0)            # 压掉灰色噪点，增强笔画对比度
    t = transforms.Normalize((0.1307,), (0.3081,))(torch.tensor(x).unsqueeze(0))
    return t.unsqueeze(0)                              # [1,1,28,28]


def image_to_input(pil_image):
    """把任意 PIL 图片 -> 灰度 -> 28x28 -> 模型输入张量"""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(pil_image).convert("L")   # 处理手机照片的旋转信息，转灰度
    img = img.resize((28, 28), Image.Resampling.BILINEAR)   # 缩放到 28x28
    return array_to_input(np.array(img, dtype=np.uint8))


def predict(model, tensor):
    """输入 [1,1,28,28] 张量，输出 (预测类别, 各类别概率)"""
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    return probs.argmax().item(), probs


def selftest(model):
    """
    管线自检：从 MNIST 测试集取一张真实图片，存成 PNG 文件，
    再走完整条「PNG 文件 -> 灰度 -> 28x28 -> 归一化 -> 预测」的推理管线，
    最后与真实标签核对。输入与训练数据同分布，应当得到高置信度的正确结果。
    """
    from torchvision import datasets

    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True)  # 原始 PIL 图片
    index = 0
    pil_image, true_label = test_set[index]
    path = OUT_DIR / "08_selftest_digit.png"
    pil_image.save(path)
    print(f"已保存一张真实测试图（真实标签 {true_label}）-> {path}")

    pred, probs = predict(model, image_to_input(pil_image))
    print(f"识别结果: {pred} | 置信度: {probs[pred]:.2%}")
    for i, p in enumerate(probs):
        print(f"{i}: {p:.3f}")
    if pred == true_label:
        print("管线自检通过：图片文件 -> 识别的整条流程正确")
    else:
        print(f"管线自检未通过：真实标签是 {true_label}，请检查 image_to_input 的预处理")


def run_canvas(model):
    """画板模式：鼠标写数字，按 s 识别"""
    import matplotlib
    try:
        matplotlib.use("TkAgg")                        # 交互式后端（Windows 自带）
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    canvas = np.zeros((28, 28), dtype=np.float32)      # 黑底画布，白色笔迹（与 MNIST 一致）
    drawing = False

    def paint(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < 28 and 0 <= y < 28:
            for dx in range(-2, 3):                    # 画 5x5 的粗笔迹
                for dy in range(-2, 3):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < 28 and 0 <= yy < 28:
                        canvas[yy, xx] = min(1.0, canvas[yy, xx] + 0.8)
        redraw()

    def redraw(title=""):
        ax.clear()
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        fig.canvas.draw_idle()

    def on_press(event):
        nonlocal drawing
        if event.inaxes:
            drawing = True
            paint(event)

    def on_release(event):
        nonlocal drawing
        drawing = False

    def on_motion(event):
        if drawing:                    # 只有按住鼠标拖动时才画
            paint(event)

    def on_key(event):
        if event.key == "s":
            img = Image.fromarray(np.uint8(canvas * 255), mode="L")
            img.save(OUT_DIR / "08_my_digit.png")
            pred, probs = predict(model, array_to_input(np.uint8(canvas * 255)))
            print(f"识别结果: {pred} | 置信度: {probs[pred]:.2%}")
            print("已保存你的字 -> outputs/08_my_digit.png")
            redraw(title=f"prediction: {pred}")
        elif event.key == "c":
            canvas[:] = 0
            redraw()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)  # 拖动时也画（仅按住时）
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw(title="Draw a digit!  s=recognize  c=clear  q=quit")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="识别你自己的手写数字")
    parser.add_argument("image", nargs="?", help="要识别的图片路径（不填则打开画板）")
    parser.add_argument("--selftest", action="store_true", help="取一张真实 MNIST 图片验证整条推理管线")
    args = parser.parse_args()

    model = load_model()

    if args.selftest:
        selftest(model)
    elif args.image:
        from PIL import Image
        pred, probs = predict(model, image_to_input(Image.open(args.image)))
        print(f"识别结果: {pred} | 置信度: {probs[pred]:.2%}")
        print("各类别概率:", {i: round(float(p), 3) for i, p in enumerate(probs)})
    else:
        run_canvas(model)


if __name__ == "__main__":
    main()
```

### 8.5 逐段讲解

**① `array_to_input`：本步的灵魂**

```python
if x.mean() > 0.5:      # 背景偏亮 -> 白底黑字，反转成黑底白字
    x = 1.0 - x
```

用「平均亮度」判断底色：白底图片整体亮（均值 > 0.5），反转后变成 MNIST 风格的黑底白字。这是对齐数据分布的关键一步。

```python
x = np.clip((x - 0.3) / 0.7, 0.0, 1.0)
```

把小于 0.3 的灰色噪声压到 0、大于 0.3 的部分拉伸——让笔迹干净利落，更接近 MNIST 的印刷式笔画。**推理数据的质量决定了识别效果**。

**② 画板实现要点**

- 画布是 28×28 的数组（和 MNIST 同尺寸），显示时 `vmin=0, vmax=1` 保证黑白对比
- `button_press_event` 开始画、`motion_notify_event` 拖动续画、`button_release_event` 停笔
- 笔迹半径 2 像素——因为 MNIST 的字相对较粗，太细的线条模型不认

**③ 为什么推荐用画板而不是拍照片？**

照片里的数字往往背景复杂、角度歪斜、还有阴影，需要更多预处理技巧（阈值分割、旋转矫正等），超出本教程范围。画板输出是「完美条件」，先体验成功，再挑战真实照片（第 10 步的挑战任务之一）。

## ✅ 完成自检

- [ ] `--selftest` 打印「管线自检通过」，置信度 > 95%
- [ ] 图片模式能识别 `outputs/08_selftest_digit.png`
- [ ] 画板模式能弹出窗口，写一个数字按 `s` 后大概率认对（写端正、居中、粗一点成功率更高）
- [ ] 能说出「数据分布一致性」是什么意思，并举一个不一致的例子
- [ ] 能向朋友解释：为什么照片识别比画板难

## 🐛 常见问题

**Q：画板窗口弹不出来？**
A：你的 Python 可能没装 tkinter（部分精简版会缺）。不影响学习：用 `--selftest` 和图片模式即可体验完整管线。修复方法：到 python.org 重装官方版 Python（默认带 tkinter）。

**Q：写得很端正还是认错？**
A：常见原因：① 字太小（尽量撑满画板）；② 笔画太细（写粗点）；③ 连笔潦草（MNIST 基本是正楷）。多写几个，你会发现模型和人类的「口味」差不多。

**Q：识别真实照片效果差？**
A：正常，真实照片有背景、阴影、倾斜。可以试试：裁剪到只剩数字、转黑白、提高对比度后再识别。这也是第 10 步的挑战任务。

**Q：`Image.Resampling.BILINEAR` 报错？**
A：旧版 Pillow 用 `Image.BILINEAR`。升级 Pillow 即可：`pip install -U pillow`。

## 📌 本步小结

1. 推理数据必须与训练数据同分布：灰度、28×28、黑底白字、同样的归一化——一处不一致，效果就崩
2. `array_to_input` 自动处理底色反转和对比度增强，是推理管线的核心
3. 一个可玩的 Demo = 训练好的模型 + 干净的输入管线 + 友好的交互界面

## 🔗 下一步

👉 [第 9 步 · CNN 进阶](md/step-09-CNN进阶.md)：用卷积神经网络把准确率推到 98%+，并理解「为什么 CNN 天生适合图像」。
