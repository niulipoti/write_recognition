"""
第 11 步 · 用 FastAPI 把模型变成网页应用（后端）
运行方式：
    python code/11_web_app.py
然后用浏览器打开：http://127.0.0.1:8000

接口一览：
    GET  /              -> 返回前端页面 web/index.html
    GET  /api/info      -> 返回模型信息（名称、参数量、类别数）
    POST /api/predict   -> 传入画布图片的 base64，返回预测数字、置信度、10 类概率
"""
import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
WEB_DIR = ROOT / "web"

MEAN, STD = 0.1307, 0.3081          # 与训练时完全一致的归一化参数


# ---------------- 1. 模型结构（与第 4 步 / 第 9 步完全一致）----------------
class MLP(nn.Module):
    """第 4 步的全连接网络"""
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
        return self.fc3(x)


class CNN(nn.Module):
    """第 9 步的卷积网络"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- 2. 加载模型（优先用准确率更高的 CNN）----------------
def load_model():
    cnn_path = OUT_DIR / "model_cnn.pth"
    mlp_path = OUT_DIR / "model_mlp.pth"
    if cnn_path.exists():                       # 第 9 步训练出来的 CNN
        model, name, path = CNN(), "CNN", cnn_path
    elif mlp_path.exists():                     # 退回到第 5 步的 MLP
        model, name, path = MLP(), "MLP", mlp_path
    else:
        raise FileNotFoundError(
            "找不到模型文件！请先在项目根目录运行：\n"
            "  python code/05_train.py     （生成 outputs/model_mlp.pth）\n"
            "  python code/09_cnn.py       （生成 outputs/model_cnn.pth，推荐）"
        )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, name, path.name


MODEL, MODEL_NAME, MODEL_FILE = load_model()
PARAMS = sum(p.numel() for p in MODEL.parameters())


# ---------------- 3. 推理管线（与第 8 步完全一致）----------------
def image_to_array(pil_image):
    """任意图片 -> 28x28 的 0~1 灰度数组（黑底白字、已压掉灰噪）；这就是模型真正看到的画面"""
    img = ImageOps.exif_transpose(pil_image).convert("L")   # 手机照片方向 + 转灰度
    img = img.resize((28, 28), Image.Resampling.BILINEAR)   # 缩放到 28x28
    x = np.array(img, dtype=np.float32) / 255.0             # [0,1]
    if x.mean() > 0.5:                                      # 白底黑字 -> 反转为黑底白字
        x = 1.0 - x
    return np.clip((x - 0.3) / 0.7, 0.0, 1.0)               # 压噪、增强对比度


def array_to_input(x):
    """28x28 的 0~1 数组 -> [1,1,28,28] 的归一化张量"""
    tensor = transforms.Normalize((MEAN,), (STD,))(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return tensor.unsqueeze(0)


def predict(tensor):
    """输入 [1,1,28,28] 张量，输出 (预测类别, 10 类概率)"""
    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1).squeeze()
    return probs.argmax().item(), probs


def preview_data_url(x):
    """把 28x28 数组放大 10 倍转成 base64 PNG，让前端能显示「模型看到的画面」"""
    img = Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255), mode="L")
    img = img.resize((280, 280), Image.Resampling.NEAREST)   # NEAREST：保留像素感，不糊
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------- 4. FastAPI 应用 ----------------
app = FastAPI(title="手写数字识别 API", description="AI 入门实战教程 · 第 11 步")


class PredictRequest(BaseModel):
    """请求体：前端把 canvas.toDataURL() 的结果原样传过来"""
    image: str


@app.get("/")
def index():
    """返回前端页面"""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/info")
def info():
    """模型信息，前端底部展示"""
    return {
        "model": MODEL_NAME,
        "model_file": MODEL_FILE,
        "params": PARAMS,
        "classes": 10,
        "input_size": "28x28",
    }


@app.post("/api/predict")
def predict_api(req: PredictRequest):
    """接收 base64 图片，返回预测结果和概率分布"""
    # 1. 解析 data URL（形如 data:image/png;base64,xxxx）
    try:
        raw = base64.b64decode(req.image.split(",", 1)[-1])
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="图片解析失败，请确认传的是 PNG/JPEG 的 base64 数据")

    # 2. 空画布检查：全黑（没画）或全白（白纸）都直接提示
    gray = np.array(image.convert("L"))
    if gray.max() < 20 or gray.min() > 235:
        raise HTTPException(status_code=400, detail="画布是空的，请先写一个数字")

    # 3. 走与第 8 步完全相同的推理管线
    x = image_to_array(image)
    pred, probs = predict(array_to_input(x))

    return {
        "prediction": pred,
        "confidence": round(float(probs[pred]), 4),
        "probabilities": [round(float(p), 4) for p in probs],
        "input_preview": preview_data_url(x),      # 给前端看一眼模型眼中的输入
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手写数字识别网页应用")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；用 0.0.0.0 可让同局域网手机访问")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args()

    print(f"已加载 {MODEL_NAME} 模型（{MODEL_FILE}，参数量 {PARAMS:,}）")
    print(f"请在浏览器打开： http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    print("按 Ctrl+C 停止服务")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
