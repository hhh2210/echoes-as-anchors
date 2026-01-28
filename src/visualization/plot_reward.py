import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_reward_curve(
    log_csv: str, save_path: str | None = None, window_size: int = 20
) -> None:
    df = pd.read_csv(log_csv)
    if df.empty:
        print(f"No data found in {log_csv}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["avg_total"].rolling(window=window_size, min_periods=1).mean(), label="total_reward", linewidth=2)
    plt.plot(df["iteration"], df["avg_correct"].rolling(window=window_size, min_periods=1).mean(), label="R_correct", linewidth=1)
    plt.plot(df["iteration"], df["avg_format"].rolling(window=window_size, min_periods=1).mean(), label="R_format", linewidth=1)
    plt.plot(df["iteration"], df["avg_repeat"].rolling(window=window_size, min_periods=1).mean(), label="R_repeat", linewidth=1)

    plt.xlabel("Generation Iteration")
    plt.ylabel("Average Reward")
    plt.title(f"Reward Progress (Smoothed, window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(log_csv), "reward_curve_smoothed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Reward curve saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward log")
    parser.add_argument("log_csv", help="Path to reward_log.csv")
    parser.add_argument("--save", dest="save_path", help="Output image path")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Smoothing window size for moving average.",
    )
    args = parser.parse_args()

    plot_reward_curve(args.log_csv, args.save_path, args.window)


if __name__ == "__main__":
    main()

