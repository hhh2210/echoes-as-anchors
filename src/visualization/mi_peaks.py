#!/usr/bin/env python3
"""
与 plot_attention_layers.py 一致的出版风格绘图：
- 绘制 AIME24 与 MATH 500 两张折线图
- 使用文件内置数据，无需 CLI 参数
"""

from __future__ import annotations

import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    print("[WARN] matplotlib not available; skip plot.")
    raise SystemExit(0)

# 与 plot_attention_layers.py 保持一致的出版风格 rcParams
plt.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# =====================
# 数据（AIME24）
# =====================
aime_x_blue = [1024, 2048, 3072, 4096, 6144, 8192, 12288]
aime_y_blue = [23.5, 27.2, 30.2, 31.2, 33.2, 34.2, 35.2]

aime_x_orange = [1024, 2048, 3072, 4096, 6144, 8192, 12288]
aime_y_orange = [22.5, 22.5, 33.5, 33.5, 33.0, 30.0, 33.2]

aime_x_star = [1024, 2048, 3072, 4096, 6144, 8192, 12288]
aime_y_star = [23.3, 30.0, 30.0, 36.7, 40.0, 43.3, 43.3]


# =====================
# 数据（MATH 500）
# =====================
math_x = [512, 1024, 2048, 3072, 4096, 5120]
math_y_blue = [59.8, 65.6, 72.0, 75.3, 77.3, 77.9]
math_y_orange = [57.1, 63.2, 70.0, 73.2, 74.9, 75.8]

repeat_x = [512, 772, 1024, 2048, 3072, 4096, 5120]
repeat_y = [71.0, 72.6, 76.0, 77.4, 78.8, 78.6, 78.6]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _plot_lines(series: dict, xlabel: str, ylabel: str, title: str, out_path: str, figsize=(3.4, 2.1), x_ticks=None) -> None:
    plt.figure(figsize=figsize)
    for label, (xs, ys, style) in series.items():
        style = dict(style or {})
        plt.plot(xs, ys, label=label, **style)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # 添加横坐标灰色网格线
    if x_ticks:
        # 统一与 y 轴字号（遵循 rcParams），并将横坐标刻度旋转 45°
        plt.xticks(x_ticks, rotation=45)
    plt.grid(True, axis='x', color='lightgray', linestyle='--', alpha=0.7, linewidth=1)
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def plot_aime24(out_path: str) -> None:
    # 形状与图例对齐：TTTS=圆圈，Original=X，Ours=五角星
    series = {
        "TTTS": (aime_x_blue, aime_y_blue, {"linestyle": "-", "marker": "o", "color": "#1f77b4"}),
        "Original": (aime_x_orange, aime_y_orange, {"linestyle": "-", "marker": "x", "color": "#ff7f0e"}),
        "Ours": (aime_x_star, aime_y_star, {"linestyle": "-", "marker": "*", "color": "#f1c40f"}),
    }
    _plot_lines(series, xlabel="Context length", ylabel="Accuracy (%)", title="AIME24", out_path=out_path, x_ticks=aime_x_blue)


def plot_math500(out_path: str) -> None:
    # 形状与图例对齐：TTTS=圆圈，Original=X，Ours=五角星
    series = {
        "TTTS": (math_x, math_y_blue, {"linestyle": "-", "marker": "o", "color": "#1f77b4"}),
        "Original": (math_x, math_y_orange, {"linestyle": "-", "marker": "x", "color": "#ff7f0e"}),
        "Ours": (repeat_x, repeat_y, {"linestyle": "-", "marker": "*", "color": "#f1c40f"}),
    }
    _plot_lines(series, xlabel="Context length", ylabel="Accuracy (%)", title="MATH 500", out_path=out_path, x_ticks=math_x)


if __name__ == "__main__":
    # It's recommended to set this path via an argument or config file.
    out_dir = "visualizations/"
    plot_aime24(os.path.join(out_dir, "AIME24_reproduced.pdf"))
    plot_math500(os.path.join(out_dir, "MATH500.pdf"))
