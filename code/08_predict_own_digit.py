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
