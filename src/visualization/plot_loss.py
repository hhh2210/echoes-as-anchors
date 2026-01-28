import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curve(
    log_csv: str, save_path: str | None = None, window_size: int = 20
) -> None:
    df = pd.read_csv(log_csv)
    if df.empty:
        print(f"No data found in {log_csv}")
        return

    plt.figure(figsize=(10, 6))
    # Plot raw data with transparency
    plt.plot(df["step"], df["loss"], label="training_loss (raw)", linewidth=1, alpha=0.4)

    # Plot smoothed data
    smoothed_loss = df["loss"].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df["step"], smoothed_loss, label=f"training_loss (smoothed)", linewidth=2)


    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve (Smoothed, window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(log_csv), "loss_curve_smoothed.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Loss curve saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss log")
    parser.add_argument("log_csv", help="Path to loss_log.csv")
    parser.add_argument("--save", dest="save_path", help="Output image path")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Smoothing window size for moving average.",
    )
    args = parser.parse_args()

    plot_loss_curve(args.log_csv, args.save_path, args.window)


if __name__ == "__main__":
    main()

