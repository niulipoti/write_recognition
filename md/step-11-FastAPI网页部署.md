# 第 11 步 · 网页部署（FastAPI + 前端）

| 项目 | 难度 | 预计用时 | 前置要求 | 完整代码 |
| :--- | :--- | :--- | :--- | :--- |
| 手写数字识别（MNIST） | ⭐⭐⭐☆☆ | 约 60 分钟 | [第 9 步](md/step-09-CNN进阶.md)（需已训练出模型） | [code/11_web_app.py](../code/11_web_app.py) + [web/index.html](../web/index.html) |

## 🎯 本步目标

学完本步，你将能够：

- [ ] 说清一个 Web 应用里「前端 / 后端 / 接口」各自负责什么
- [ ] 用 **FastAPI** 把训练好的模型包装成 HTTP 接口
- [ ] 用**原生 HTML + Canvas + JavaScript** 写一个能画数字的前端页面
- [ ] 在浏览器里跑通「画数字 → 请求接口 → 显示预测和概率」的完整链路
- [ ] 用 FastAPI 自动生成的 `/docs` 在线调试接口
- [ ] 把地址改成 `0.0.0.0`，让**同一个 WiFi 下的手机**也能访问你的 AI 应用

## 📚 知识准备

### 从「脚本」到「应用」差了什么？

前面 10 步我们已经有了能识别数字的模型，但它只能通过 `python code/08_predict_own_digit.py` 这种命令行方式使用。要让别人也能用，缺的是**一层服务化包装**：把模型挂在网络上，任何人都能通过浏览器访问。

```
第 8 步：  你 → 命令行脚本 → 模型 → 终端里打印结果
第 11 步： 浏览器（前端）→ HTTP 请求 → FastAPI（后端）→ 模型 → JSON → 浏览器渲染
```

### 前端、后端、接口分别是什么？

| 角色 | 通俗理解 | 本项目里是谁 | 负责什么 |
| :--- | :--- | :--- | :--- |
| **前端** | 门面 + 服务员 | `web/index.html` | 画板、按钮、把结果画得好看 |
| **后端** | 后厨 | `code/11_web_app.py` | 拿着模型算，返回预测结果 |
| **接口（API）** | 点菜的菜单 | `/api/predict` | 约定前端怎么问、后端怎么答 |
| **HTTP 请求/响应** | 传菜 | POST + JSON | 前端把图片发给后端，后端把结果发回来 |

关键认知：**前端不需要懂 AI，后端不需要懂界面**，两者只通过接口（约定好的 URL + 数据格式）沟通。这就是为什么同一个模型能同时给网页、手机 App、小程序用。

### 为什么选 FastAPI？

| 框架 | 特点 |
| :--- | :--- |
| **FastAPI** | 现代、快、**自带交互式接口文档**（`/docs`）、用 Python 类型注解自动校验数据。当前最热门的 Python Web 框架之一 |
| Flask | 更老更简单，但需要额外装插件才有接口文档和数据校验 |
| Django | 全家桶，适合大型业务系统，对本项目来说太重 |

### JSON：前后端的通用语言

前后端传输数据用 **JSON**（一种文本格式），可以把它理解成「跨语言的字典」：

```json
{ "prediction": 7, "confidence": 0.9994, "probabilities": [0.0, 0.0, ...] }
```

Python 里的 `dict` 会由 FastAPI 自动转成 JSON 发给浏览器，浏览器再用 `JSON.parse` 变回对象——**双方都不用关心中间是怎么编码的**。

### 图片怎么传给后端？

两种常见做法：

| 方式 | 说明 | 本教程 |
| :--- | :--- | :--- |
| **base64 内嵌 JSON** | 把 PNG 二进制编码成字符串塞进 JSON，一个请求搞定 | ✅ 用它（简单、能看懂全过程） |
| multipart/form-data | 传统文件上传，需要额外解析 | 生产环境常用，见「常见问题」 |

前端一行 `canvas.toDataURL("image/png")` 就能拿到 `data:image/png;base64,iVBOR...` 这样的字符串，后端 base64 解码就还原成图片。

### 本地服务与端口

`http://127.0.0.1:8000` 的含义：

- `127.0.0.1`（即 `localhost`）：**只有本机**能访问
- `0.0.0.0`：监听所有网卡，**同一局域网的其他设备**（手机、平板）都能访问
- `8000`：端口号，一个程序占一个端口；被占用时换一个（如 8001）

## 🛠️ 动手实践

> 💻 **运行前确认**：终端已激活虚拟环境（命令行开头应有 `(.venv)`），当前目录是项目根目录 `手写数字识别教程`。
> 没激活就执行 `.venv\Scripts\Activate.ps1`（macOS/Linux 用 `source .venv/bin/activate`）；不在根目录就先 `cd 手写数字识别教程`。

### 11.1 安装 Web 依赖

这两个包前面 10 步用不到，所以单独放在 `requirements-web.txt` 里按需安装：

```bash
pip install -r requirements-web.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 11.2 本步新增的文件

```
手写数字识别教程/
├── code/11_web_app.py      ← 后端：FastAPI 接口 + 推理管线
├── web/index.html          ← 前端：画板 + 结果展示（HTML/CSS/JS 全在一个文件里）
└── requirements-web.txt    ← Web 依赖
```

### 11.3 启动服务

```bash
python code/11_web_app.py
```

**预期输出**：

```
已加载 CNN 模型（model_cnn.pth，参数量 52,138）
请在浏览器打开： http://127.0.0.1:8000
按 Ctrl+C 停止服务
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

看到 `Application startup complete` 就说明服务起来了。**这个终端窗口要一直开着**——它就是你的服务器，关掉服务就停了。

### 11.4 在浏览器里玩

打开 <http://127.0.0.1:8000>，你会看到一个深色画板：

1. 用鼠标（手机上用手指）**写一个大一点的数字**，尽量居中、笔画粗一点
2. 点「识别」（或直接按键盘 <kbd>s</kbd>）
3. 右侧立刻显示：AI 认为是几、置信度多少、10 个数字各自的概率条
4. 最下面的「**模型眼中的输入**」会把 28×28 的灰度图放大给你看——**这是理解「模型眼里的世界」最直观的方式**
5. 按 <kbd>c</kbd> 清空重写；勾选「松手自动识别」可以边写边猜

试试写一个**笔直的印刷体 7**，再写一个**带弧度的草书 7**，对比置信度差异——你会亲眼看到第 8 步讲的「数据分布一致性」。

### 11.5 看一眼接口（FastAPI 的杀手锏）

服务运行时，浏览器访问 <http://127.0.0.1:8000/docs>，FastAPI 会**自动生成交互式接口文档**：

- 列出 `GET /api/info`、`POST /api/predict` 等所有接口
- 点开任意接口 → 「Try it out」→ 填参数 → 「Execute」，**不写一行代码就能调试**
- 还能直接看到请求/响应的 JSON 结构和数据类型校验规则

再访问 <http://127.0.0.1:8000/api/info>，你会看到模型信息：

```json
{"model":"CNN","model_file":"model_cnn.pth","params":52138,"classes":10,"input_size":"28x28"}
```

> 💡 在真实工作中，这份自动生成的文档就是前端和后端约定接口的「合同」——不用再手写接口文档，改代码时文档自动同步。

### 11.6 后端完整代码

```python
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
```

**逐段讲解**

**① 模型和推理管线：一行都没改，只是换了调用方式**

对比第 8 步你会发现，`MLP`、`CNN` 结构、`image_to_array`、`array_to_input`、`predict` 全是原样搬过来的。**这就是分层设计的好处：模型层不动，只加一层服务层。** 注意这里多做了一步——把预处理后的 28×28 数组也返回给前端（`input_preview`），方便你在页面上看到「模型眼中的输入」。

**② 模型只加载一次**

```python
MODEL, MODEL_NAME, MODEL_FILE = load_model()
```

写在模块顶层，服务启动时加载一次，之后每个请求复用。**千万不要把加载写进接口函数里**——那样每次请求都要读一遍模型文件，慢几十倍。（改了模型文件想生效，需要重启服务。）

**③ 接口就是函数 + 装饰器**

```python
@app.post("/api/predict")
def predict_api(req: PredictRequest):
```

- `@app.get("/")` / `@app.post("/api/predict")`：声明「访问这个路径时调用这个函数」，POST 表示提交数据、GET 表示取数据
- 参数 `req: PredictRequest`：FastAPI 自动把请求体 JSON 解析成对象，**格式不对会自动返回 422 错误**并说明哪里不对——这是 FastAPI 相对 Flask 的最大便利
- 返回值是 `dict`，FastAPI 自动转成 JSON

**④ 用 HTTPException 返回友好错误**

```python
raise HTTPException(status_code=400, detail="画布是空的，请先写一个数字")
```

前端拿到 400 后会读取 `detail` 显示给用户。**这就是「后端管校验、前端管展示」的分工**：用户画了空画布，后端负责判断并给出原因。

**⑤ 为什么用 `--host 0.0.0.0`？**

默认 `127.0.0.1` 只有本机能访问（更安全）。要让手机访问就传 `--host 0.0.0.0`，然后用电脑的局域网 IP（`ipconfig` 里找，形如 `192.168.1.5`）在手机浏览器访问 `http://192.168.1.5:8000`。

### 11.7 前端完整代码

在项目根目录**新建一个 `web` 文件夹**，把下面的内容保存成 `web/index.html`（**一个文件就包含界面、样式和所有逻辑**，不需要任何前端构建工具）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手写数字识别 · AI 入门实战</title>
<style>
  :root {
    --bg: #0f1420;
    --card: #1a2131;
    --card-2: #222b3d;
    --line: #2f3a50;
    --text: #e8edf7;
    --muted: #8c9ab3;
    --accent: #4c8dff;
    --accent-2: #22c55e;
    --warn: #f59e0b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 28px 18px 48px;
    background: radial-gradient(1200px 600px at 50% -10%, #1b2740 0%, var(--bg) 60%);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
    min-height: 100vh;
  }
  header { text-align: center; margin-bottom: 26px; }
  h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: .5px; }
  header p { margin: 0; color: var(--muted); font-size: 14px; }
  .wrap {
    max-width: 980px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 20px;
    align-items: start;
  }
  @media (max-width: 820px) { .wrap { grid-template-columns: 1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 18px;
  }
  .card h2 {
    margin: 0 0 14px;
    font-size: 15px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: .5px;
  }
  #board {
    width: 280px;
    height: 280px;
    display: block;
    margin: 0 auto 14px;
    background: #000;
    border-radius: 10px;
    border: 1px solid var(--line);
    cursor: crosshair;
    touch-action: none;            /* 手机上禁止滚动，才能顺畅画字 */
  }
  .row { display: flex; gap: 10px; }
  button {
    flex: 1;
    padding: 11px 14px;
    font-size: 15px;
    font-family: inherit;
    color: #fff;
    background: var(--accent);
    border: 0;
    border-radius: 9px;
    cursor: pointer;
    transition: filter .15s, opacity .15s;
  }
  button:hover { filter: brightness(1.12); }
  button:disabled { opacity: .55; cursor: not-allowed; }
  button.ghost { background: var(--card-2); border: 1px solid var(--line); color: var(--text); }
  .hint { margin-top: 12px; color: var(--muted); font-size: 12.5px; line-height: 1.7; }
  .hint kbd {
    background: var(--card-2); border: 1px solid var(--line); border-radius: 4px;
    padding: 1px 6px; font-family: ui-monospace, Consolas, monospace; font-size: 12px;
  }
  .toggle { display: flex; align-items: center; gap: 8px; margin-top: 12px; color: var(--muted); font-size: 13px; }
  .toggle input { width: 16px; height: 16px; accent-color: var(--accent); }

  .result-top { display: flex; align-items: center; gap: 20px; margin-bottom: 18px; }
  .big-digit {
    width: 104px; height: 104px; flex: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 58px; font-weight: 700;
    background: var(--card-2); border: 1px solid var(--line); border-radius: 14px;
    color: var(--muted);
  }
  .big-digit.done { color: var(--accent-2); border-color: #2c5c3c; }
  .meta { font-size: 14px; color: var(--muted); line-height: 1.9; }
  .meta b { color: var(--text); font-size: 17px; }

  .bars { display: grid; gap: 6px; margin-bottom: 18px; }
  .bar { display: grid; grid-template-columns: 20px 1fr 54px; align-items: center; gap: 9px; font-size: 13px; }
  .bar .label { color: var(--muted); text-align: center; font-family: ui-monospace, Consolas, monospace; }
  .bar .track { height: 14px; background: var(--card-2); border-radius: 999px; overflow: hidden; }
  .bar .fill { height: 100%; width: 0; background: #3b5c94; border-radius: 999px; transition: width .28s ease; }
  .bar.top .fill { background: var(--accent-2); }
  .bar .pct { color: var(--muted); text-align: right; font-family: ui-monospace, Consolas, monospace; }

  .preview-box { display: flex; gap: 14px; align-items: center; border-top: 1px dashed var(--line); padding-top: 16px; }
  .preview-box img {
    width: 104px; height: 104px; flex: none;
    image-rendering: pixelated;                 /* 保留 28x28 的像素颗粒感 */
    background: #000; border: 1px solid var(--line); border-radius: 10px;
  }
  .preview-box .txt { color: var(--muted); font-size: 12.5px; line-height: 1.8; }
  .status { margin-top: 14px; font-size: 13px; color: var(--muted); min-height: 19px; }
  .status.err { color: var(--warn); }
  .status.ok { color: var(--accent-2); }
  footer { max-width: 980px; margin: 22px auto 0; text-align: center; color: var(--muted); font-size: 12.5px; }
  code { font-family: ui-monospace, Consolas, monospace; }
</style>
</head>
<body>

<header>
  <h1>手写数字识别</h1>
  <p>在下面的画板上写一个 0~9 的数字，让 AI 猜猜看</p>
</header>

<div class="wrap">
  <!-- 左：画板 -->
  <section class="card">
    <h2>① 写一个数字</h2>
    <canvas id="board" width="280" height="280"></canvas>
    <div class="row">
      <button id="btn-predict">识别</button>
      <button id="btn-clear" class="ghost">清空</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="auto"> 松手自动识别
    </label>
    <div class="hint">
      建议写<strong>粗一点、居中</strong>，像 MNIST 那样（黑底白字）。<br>
      快捷键：<kbd>s</kbd> 识别 · <kbd>c</kbd> 清空
    </div>
  </section>

  <!-- 右：结果 -->
  <section class="card">
    <h2>② 模型的回答</h2>
    <div class="result-top">
      <div class="big-digit" id="big-digit">?</div>
      <div class="meta" id="meta">
        还没有识别结果<br>
        写下数字后点「识别」
      </div>
    </div>

    <div class="bars" id="bars"></div>

    <div class="preview-box">
      <img id="preview" alt="模型看到的画面" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7">
      <div class="txt">
        <strong style="color:var(--text)">模型眼中的输入</strong><br>
        你画的原图会被转成 <code>28×28</code> 灰度图、反转底色、压掉灰噪，<br>
        再归一化后送进网络——这就是 AI「看到」的样子。
      </div>
    </div>

    <div class="status" id="status"></div>
  </section>
</div>

<footer id="footer">正在读取模型信息…</footer>

<script>
/* ---------- 画板：支持鼠标与触摸 ---------- */
const SIZE = 280;
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
let drawing = false;
let hasInk = false;          // 是否画过东西
let last = null;             // 上一个点

function clearBoard() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  hasInk = false;
  last = null;
}

function toBoard(e) {        // 事件坐标 -> 画布坐标
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * SIZE / r.width,
    y: (e.clientY - r.top) * SIZE / r.height
  };
}

function stroke(a, b) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 20;        // 粗笔迹，更接近 MNIST 的笔画
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  hasInk = true;
  last = toBoard(e);
  stroke(last, { x: last.x + 0.01, y: last.y + 0.01 });   // 单击也留一个点
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = toBoard(e);
  stroke(last, p);
  last = p;
});

["pointerup", "pointercancel"].forEach((ev) =>
  canvas.addEventListener(ev, () => {
    drawing = false;
    if (document.getElementById("auto").checked && hasInk) recognize();
  })
);

/* ---------- 概率条：初始化 10 行 ---------- */
const bars = document.getElementById("bars");
const barEls = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar";
  row.innerHTML = `<span class="label">${d}</span>
                   <span class="track"><span class="fill"></span></span>
                   <span class="pct">—</span>`;
  bars.appendChild(row);
  barEls.push({
    row,
    fill: row.querySelector(".fill"),
    pct: row.querySelector(".pct")
  });
}

const elBig = document.getElementById("big-digit");
const elMeta = document.getElementById("meta");
const elPreview = document.getElementById("preview");
const elStatus = document.getElementById("status");
const btnPredict = document.getElementById("btn-predict");

function setStatus(text, kind) {
  elStatus.textContent = text || "";
  elStatus.className = "status" + (kind ? " " + kind : "");
}

function renderResult(data) {
  elBig.textContent = data.prediction;
  elBig.classList.add("done");
  elMeta.innerHTML =
    `AI 认为是：<b>${data.prediction}</b><br>置信度：<b>${(data.confidence * 100).toFixed(2)}%</b>`;
  barEls.forEach((bar, d) => {
    const p = data.probabilities[d];
    bar.fill.style.width = (p * 100).toFixed(1) + "%";
    bar.pct.textContent = (p * 100).toFixed(1) + "%";
    bar.row.classList.toggle("top", d === data.prediction);
  });
  if (data.input_preview) elPreview.src = data.input_preview;
}

/* ---------- 识别：把画布转成 base64 发给后端 ---------- */
async function recognize() {
  if (!hasInk) {
    setStatus("画布是空的，先写一个数字吧 ✍️", "err");
    return;
  }
  btnPredict.disabled = true;
  setStatus("识别中…");
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") })
    });
    const data = await res.json();
    if (!res.ok) {                       // 后端返回 400 时会带 detail 说明
      setStatus(data.detail || "识别失败，请重试", "err");
      return;
    }
    renderResult(data);
    setStatus("识别完成 ✅", "ok");
  } catch (err) {
    setStatus("请求失败：" + err.message + "（服务还在运行吗？）", "err");
  } finally {
    btnPredict.disabled = false;
  }
}

/* ---------- 按钮、快捷键、模型信息 ---------- */
btnPredict.addEventListener("click", recognize);
document.getElementById("btn-clear").addEventListener("click", () => {
  clearBoard();
  elBig.textContent = "?";
  elBig.classList.remove("done");
  elMeta.innerHTML = "还没有识别结果<br>写下数字后点「识别」";
  barEls.forEach((b) => { b.fill.style.width = "0"; b.pct.textContent = "—"; b.row.classList.remove("top"); });
  setStatus("");
});

window.addEventListener("keydown", (e) => {
  if (e.key === "s") recognize();
  if (e.key === "c") document.getElementById("btn-clear").click();
});

fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    document.getElementById("footer").innerHTML =
      `当前模型：<code>${info.model}</code>（${info.model_file}）· 参数量 <code>${info.params.toLocaleString()}</code> · 输入 ${info.input_size} · ${info.classes} 分类`;
  })
  .catch(() => { document.getElementById("footer").textContent = "模型信息读取失败"; });

clearBoard();
</script>
</body>
</html>
```

**逐段讲解**

**① 画板：用 Canvas 当作「纸」**

`<canvas width="280" height="280">` 是一块位图画布，用 `getContext("2d")` 拿到画笔后就能画线。关键是 `pointerdown / pointermove / pointerup` 三个事件：

- 按下：记下起点，`setPointerCapture` 锁定指针（手指滑出画布也能继续画）
- 移动：从上一个点连一条线到当前点
- 抬起：结束这一笔

`pointer` 事件**同时兼容鼠标、触摸屏和手写笔**，一套代码通吃，所以手机也能用。CSS 里的 `touch-action: none` 是必须的，否则手机上拖动页面会跟着滚动。

**② 坐标换算：为什么不能直接用 `e.clientX`？**

```javascript
x: (e.clientX - r.left) * SIZE / r.width
```

`clientX` 是相对浏览器窗口的坐标，而画布可能被缩放或位移。用 `getBoundingClientRect()` 拿到画布真实位置，再按比例换算成画布内部坐标——**这样无论画布显示成多大，笔画位置都准确**。

**③ 把画布变成图片：一行搞定**

```javascript
canvas.toDataURL("image/png")
```

直接返回 `data:image/png;base64,...` 字符串。前端不需要懂图片编码，后端用 base64 解码 + PIL 打开即可。**这就是前面「base64 传图」的落地方式**。

**④ 发请求与渲染结果**

```javascript
const res = await fetch("/api/predict", { method: "POST", ... });
const data = await res.json();
```

`fetch` 是浏览器内置的请求工具，`async/await` 让异步代码看起来像同步——等请求回来再渲染。注意 `/api/predict` 是**相对路径**：页面和后端同一个域名端口，所以不用写完整 URL（这叫同源，也就不会遇到跨域问题）。

**⑤ 概率条与像素预览**

- 概率条：先建好 10 行 DOM，拿到结果后只改宽度和文字（**不要每次都重建 DOM**，那样会闪）
- 像素预览：`image-rendering: pixelated` 让放大后的 28×28 图保持颗粒感，而不是被浏览器糊成一片——这样你才能真正看清模型吃进去的东西

### 11.8 小实验

1. 把启动命令换成 `python code/11_web_app.py --port 8001`，页面也要跟着改成 `http://127.0.0.1:8001`
2. 打开 `web/index.html`，把 `ctx.lineWidth = 20` 改成 `6`（细笔迹），重新识别，观察准确率变化——**体会「推理数据质量」的影响**
3. 在页面上加一个「保存这张图」按钮：提示是 `canvas.toDataURL("image/png")` 配合一个 `<a download>`
4. 用 `python code/11_web_app.py --host 0.0.0.0` 启动，用同一个 WiFi 的手机访问电脑 IP，在手机上写数字
5. 打开 `/docs`，用「Try it out」直接调 `/api/info` 和 `/api/predict`，感受接口调试

## ✅ 完成自检

- [ ] 终端打印「已加载 CNN 模型」并且 `Application startup complete`
- [ ] 浏览器打开 `http://127.0.0.1:8000` 能看到画板和右侧结果区
- [ ] 写一个数字点「识别」，能显示预测数字 + 置信度 + 10 条概率条
- [ ] 概率条中最长的那一根与预测数字一致
- [ ] 「模型眼中的输入」能看到 28×28 的像素化灰度图
- [ ] 空画布点「识别」会提示「画布是空的」
- [ ] 访问 `/docs` 能看到自动生成的接口文档，并能用 Try it out 调通
- [ ] 访问 `/api/info` 返回模型名、参数量等信息
- [ ] 能用 `Ctrl+C` 停止服务，重启后页面仍正常
- [ ] 能说出前端、后端、接口各自负责什么

## 🐛 常见问题

**Q：启动报 `ModuleNotFoundError: No module named 'fastapi'`？**
A：还没装 Web 依赖。运行 `pip install -r requirements-web.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`（注意和前面 10 步的 `requirements.txt` 是两个文件）。

**Q：报 `error while attempting to bind on address ('127.0.0.1', 8000): 只允许使用每个套接字地址`？**
A：端口被占用（常见于上一次服务没关掉）。换个端口：`python code/11_web_app.py --port 8001`。查占用：`netstat -ano | findstr :8000`。

**Q：启动就报 `找不到模型文件！`？**
A：先训练模型：`python code/05_train.py`，或者更好——`python code/09_cnn.py`（CNN 更准）。模型文件会生成在 `outputs/` 下。

**Q：页面打开是 `{"detail":"Not Found"}` 或空白？**
A：说明 `web/index.html` 不在位。确认目录结构是「`web/` 与 `code/` 同级」，文件名叫 `index.html`（小写）。

**Q：页面能打开，但中文全是乱码（问号、方块）？**
A：`index.html` 的保存编码不对，必须存成 **UTF-8**。用 VS Code 打开后点右下角的编码（显示 `GBK` 或 `ANSI`），选「通过编码保存」→ `UTF-8`。记事本另存为时也要把编码选成 UTF-8。

**Q：点了「识别」没反应，控制台报错 `Failed to fetch`？**
A：后端没在运行，或者你改过端口却仍访问旧地址。确认运行服务的终端还开着、地址端口一致。

**Q：为什么不用 `python -m uvicorn code.11_web_app:app`？**
A：因为文件名以数字开头，Python 无法用 `import 11_web_app` 语法导入（模块名不能以数字开头）。所以本教程把 `uvicorn.run()` 写在脚本里，用 `python code/11_web_app.py` 启动。想用命令行方式，把文件改名为 `app.py` 再执行 `uvicorn app:app --reload`。

**Q：改了模型文件（重新训练）后，网页结果没变化？**
A：模型在服务启动时加载一次，**改完必须重启服务**（`Ctrl+C` 再启动）。这是刻意的设计——避免每个请求都读一遍几百 KB 的模型文件。

**Q：真的要用这种 base64 方式传图片吗？**
A：教学用途足够，但生产环境更常用文件上传：

```python
from fastapi import UploadFile, File

@app.post("/api/predict-upload")
async def predict_upload(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    ...
```

**Q：`--host 0.0.0.0` 后手机打不开？**
A：① 手机和电脑要在同一个 WiFi；② 用电脑的局域网 IP（`ipconfig` / `ipconfig` 找 IPv4 地址，形如 `192.168.x.x`），不是 `127.0.0.1`；③ Windows 防火墙可能拦截，首次运行 Python 时选「允许访问」。

**Q：这样部署到公网安全吗？**
A：**uvicorn 自带的服务器只适合本地开发/演示**，不适合直接暴露公网。要上线需要：加反向代理（Nginx）、HTTPS、鉴权、限流，并且用 `gunicorn`/`uvicorn workers` 起多进程。本步目标是「理解服务化」，不是「生产运维」。

## 📌 本步小结

1. 服务化 = 在模型外面包一层 HTTP 接口：前端负责界面、后端负责计算、JSON 负责沟通，三者通过接口（URL + 数据格式）解耦
2. FastAPI 用「函数 + 装饰器 + 类型注解」定义接口，自带数据校验、错误处理和交互式文档 `/docs`
3. 模型和推理管线一行都不用改——**分层设计让你的 AI 能力可以被网页、App、小程序反复复用**

## 🔗 下一步

你已经把「模型 → 接口 → 界面」的完整产品链路走通了。接下来可以：

- 回到 [README](../README.md) 查看全部 11 步的总览与学习路线
- 去 [第 10 步](md/step-10-总结与学习路线.md) 的挑战任务继续升级（Fashion-MNIST、Kaggle 竞赛、更多网络结构）
- 把本步的 `/api/predict` 接进你自己的项目：批量识别表格里的手写数字、做一个小程序、或者换成识别字母与汉字（EMNIST / 中文手写数据集）
