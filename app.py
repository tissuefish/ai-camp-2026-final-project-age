"""年龄估计网页演示（本地运行，不需要服务器/账号）

运行：
    python app.py
然后浏览器打开：http://127.0.0.1:5000
上传一张人脸照片 → 显示预测年龄。

依赖已装：flask, torch, pillow, numpy
"""

import sys
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

import train  # 复用 train.py 里的模型结构和配置

MODEL_PATH = Path(__file__).resolve().parent / "models" / "age_cnn.pt"

app = Flask(__name__)
model = None


def has_face(img):
    """判断图片里是否有人脸（用于"未识别出人脸就拒绝"的人工边界）。

    注意：Haar 对"脸占满整张图"的裁剪正脸会漏检，所以对小图（<400px）不拦截，
    视为已是正脸裁剪；对大图先按原尺寸、再缩小各检测一次。未安装 cv2 时不拦截。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return True

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def detect(im):
        gray = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
        # minNeighbors=5：实测 nb=3 会把图案/纹理误检成人脸（1290x2796 无人脸图
        # nb=3 误检 1 个、nb=5 为 0），用 5 既能拒绝无人脸图，也不会漏掉真实正脸
        return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # 注意：detectMultiScale 检测到人脸时返回 numpy 数组，直接用 if 判断会抛
    # "ambiguous truth value"，必须用 len() 判断数量
    if len(detect(img)):
        return True
    if max(img.size) >= 800:  # 大图缩小后再试，Haar 对适中大小的人脸更灵敏
        scale = 640 / max(img.size)
        small = img.resize((int(img.width * scale), int(img.height * scale)))
        if len(detect(small)):
            return True
    return max(img.size) < 400  # 很小的图视为已经是裁剪正脸，不拦截


def load_model():
    """加载训练好的模型；没训练过则提示先训练。"""
    global model
    if not MODEL_PATH.exists():
        return False
    m = train.SmallCNN()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()
    model = m
    return True


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    """接收一张照片，返回预测年龄；处理不了的情况给出拒绝信息（人工边界）。"""
    if model is None:
        return jsonify({"error": "模型未就绪，请先运行 python train.py"}), 400

    file = request.files.get("photo")
    if file is None or file.filename == "":
        return jsonify({"error": "没有收到照片，请选择一张照片"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "无法读取这张图片，请换一张清晰的正脸照片"}), 400

    # 人工边界：图片太小或明显不是照片时，拒绝而不是乱猜
    if min(img.size) < 32:
        return jsonify({"error": "图片太小，无法可靠判断，请上传更大的清晰照片"}), 400

    # 人工边界：没有检测到人脸时拒绝，而不是乱猜
    if not has_face(img):
        return jsonify({"error": "未识别出人脸，请上传一张含清晰正脸的照片"}), 400

    # 实测（2026-08-25）：对真实自拍，不同裁剪方式预测差异巨大（5/6/9/25/50），
    # 人脸裁剪反而把结果从"整张图 25 岁"拉低到"9 岁"。因此直接用整张图预测更稳。
    img = img.resize((train.IMG_SIZE, train.IMG_SIZE))
    x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        age = float(model(x).item())

    age = int(max(1, min(116, round(age))))
    return jsonify({
        "age": age,
        "note": "演示工具，仅用于教学，不用于真实年龄判断或任何决定",
    })


if __name__ == "__main__":
    if not load_model():
        print("未找到 models/age_cnn.pt，请先运行：python train.py")
        sys.exit(1)
    print("模型已加载。浏览器打开：http://127.0.0.1:5000")
    print("局域网访问：同一 WiFi 的设备打开 http://<本机局域网IP>:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)  # 0.0.0.0 允许局域网设备访问
