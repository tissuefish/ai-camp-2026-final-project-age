"""UTKFace 年龄估计训练脚本

作用：
    在同一测试集上比较 基线（平均年龄） 和 候选（小 CNN 回归），
    保存真实指标和每条测试样本的预测，供失败分析使用。

运行：
    python train.py

产出：
    results/metrics.json            基线/候选在同一测试集上的 MAE 等指标
    results/test_predictions.csv    每条测试样本：真实年龄 / 基线预测 / CNN 预测 / 误差
    models/age_cnn.pt               训练好的模型（被 .gitignore 排除，不提交）
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------- 配置 ----------------
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "utkface" / "UTKFace"
OUT_DIR = Path(__file__).resolve().parent / "results"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "age_cnn.pt"

IMG_SIZE = 64          # 输入图片缩放到 64x64
BATCH_SIZE = 64
SEED = 42


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="UTKFace 年龄估计：基线 vs 小 CNN")
    p.add_argument("--train-n", type=int, default=4000, help="训练样本数（默认；数据不足时用全部非测试样本）")
    p.add_argument("--test-n", type=int, default=1000, help="测试样本数（固定，和基线同一批）")
    p.add_argument("--epochs", type=int, default=2, help="训练轮次")
    p.add_argument("--race", type=int, default=None, help="只保留该人种（0白 1黑 2亚洲 3印度 4其他），默认全部")
    p.add_argument("--tag", default="default", help="结果标签，用于区分不同实验")
    return p.parse_args()

# 文件名格式：[年龄]_[性别]_[人种]_[日期].jpg[.chip].jpg
FILENAME_PATTERN = re.compile(r"^(\d+)_([01])_([0-4])_\d+\.jpg(\.chip)?\.jpg$")


# ---------------- 数据 ----------------
def load_samples(race=None):
    """读取所有能解析的文件名，返回 [(图片路径, 年龄)]，坏文件名自动跳过。
    排序保证在任何机器上结果可复现。race 非空时只保留该人种。"""
    samples = []
    for f in sorted(DATA_DIR.glob("*.jpg")):
        m = FILENAME_PATTERN.match(f.name)
        if m:
            age, gender, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if race is not None and r != race:
                continue
            samples.append((str(f), age))
    return samples


class FaceAgeDataset(Dataset):
    """把 (路径, 年龄) 列表变成 torch 数据集：返回 (3x64x64 归一化张量, 年龄)。"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, age = self.samples[i]
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return x, torch.tensor(age, dtype=torch.float32)


# ---------------- 候选模型：小 CNN（回归） ----------------
class SmallCNN(nn.Module):
    """3 层卷积提取面部特征 + 全连接输出 1 个年龄数值。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 8) ** 2, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.reg(self.features(x)).squeeze(-1)


# ---------------- 主流程 ----------------
def main():
    args = parse_args()
    TRAIN_N, TEST_N, EPOCHS = args.train_n, args.test_n, args.epochs
    random.seed(SEED)
    torch.manual_seed(SEED)

    samples = load_samples(race=args.race)
    print(f"总样本：{len(samples)}（跳过 5 个坏文件名）")

    # 固定划分：先按种子 42 打乱；测试集固定取最后 TEST_N 个（跨实验相同），
    # 训练集取前 TRAIN_N 个；若过滤后数据不足，则用全部非测试样本训练。
    random.shuffle(samples)
    test_samples = samples[-TEST_N:]
    if args.train_n and args.train_n < len(samples) - TEST_N:
        train_samples = samples[:args.train_n]
    else:
        train_samples = samples[:-TEST_N]
    print(f"[实验 tag={args.tag}] 训练集：{len(train_samples)}   测试集：{len(test_samples)}（固定）")

    train_loader = DataLoader(FaceAgeDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FaceAgeDataset(test_samples), batch_size=BATCH_SIZE)

    # ---- 基线：对每个人预测训练集平均年龄 ----
    mean_age = float(np.mean([s[1] for s in train_samples]))
    true_ages = np.array([s[1] for s in test_samples], dtype=float)
    base_preds = np.full(len(test_samples), mean_age)
    base_mae = float(np.mean(np.abs(true_ages - base_preds)))
    print(f"\n[基线] 平均年龄={mean_age:.2f} 岁   测试 MAE = {base_mae:.3f} 岁")

    # ---- 候选：小 CNN 回归 ----
    model = SmallCNN()
    criterion = nn.L1Loss()          # MAE 对应的损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        print(f"[候选] epoch {epoch}/{EPOCHS}   训练 MAE = {total_loss / n:.3f}")

    model.eval()
    cnn_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            cnn_preds.extend(model(xb).tolist())
    cnn_preds = np.clip(np.array(cnn_preds), 1, 116)
    cnn_mae = float(np.mean(np.abs(true_ages - cnn_preds)))
    print(f"[候选] 测试 MAE = {cnn_mae:.3f} 岁")

    # ---- 保存结果 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)                     # 最新模型，网页演示用
    torch.save(model.state_dict(), MODEL_DIR / f"age_cnn_{args.tag}.pt")  # 带标签副本

    metrics = {
        "tag": args.tag,
        "train_n": len(train_samples), "test_n": len(test_samples),
        "img_size": IMG_SIZE, "epochs": EPOCHS, "seed": SEED,
        "mean_age_train": round(mean_age, 2),
        "baseline_mae": round(base_mae, 3),
        "cnn_mae": round(cnn_mae, 3),
        "improvement": round(base_mae - cnn_mae, 3),
    }
    with open(OUT_DIR / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存每条测试样本预测，按误差排序，方便找成功/失败案例
    rows = []
    for i, (path, true_age) in enumerate(test_samples):
        rows.append({
            "image": Path(path).name,
            "true_age": int(true_age),
            "baseline_pred": round(float(base_preds[i]), 1),
            "cnn_pred": round(float(cnn_preds[i]), 1),
            "cnn_error": round(abs(true_ages[i] - cnn_preds[i]), 1),
        })
    rows.sort(key=lambda r: -r["cnn_error"])
    import csv
    with open(OUT_DIR / f"test_predictions_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n已保存：results/metrics_{args.tag}.json, results/test_predictions_{args.tag}.csv, models/age_cnn_{args.tag}.pt")
    print(f"\n=== 结果汇总（tag={args.tag}）===")
    print(f"基线 MAE = {base_mae:.3f} 岁 | 候选 CNN MAE = {cnn_mae:.3f} 岁 | 提升 = {base_mae - cnn_mae:+.3f} 岁")
    print("\n误差最大的 3 个测试样本（失败案例候选）：")
    for r in rows[:3]:
        print(f"  {r['image']}  真实={r['true_age']}  预测={r['cnn_pred']}  误差={r['cnn_error']}")


if __name__ == "__main__":
    main()
