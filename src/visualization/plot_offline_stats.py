#!/usr/bin/env python3
"""
Plot offline stats into publication-ready PDFs.

Example:

python -m src.visualization.plot_offline_stats \
  --results_dir /path/to/analysis_results_YYYYMMDD_HHMMSS \
  --fmt pgf
Inputs (under --results_dir):
- comparison_table.csv (Echo Likelihood Gap main table)
- removed_prefix_length_distribution.csv (group/bin/count)
- length_stratified_summary.csv (group/bin/metrics)
- Optional minor trends:
  - deltaL_deciles_vs_acc_overall.csv, removed_tokens_bin_vs_acc_overall.csv, zx_vs_acc_overall.csv
  - Or fallback compute from per_sample_extended.csv or extended_metrics_*.jsonl

Outputs (saved to --results_dir):
- removed_prefix_length_distribution.pdf
- length_stratified_summary.pdf
- latex_tables/echo_gap_table.tex
- (optional) deciles_vs_acc_overall.pdf, removed_bins_vs_acc_overall.pdf, zx_vs_acc_overall.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

# --- Matplotlib/LaTeX configuration (paper-aligned) ---
# We configure sizes relative to LaTeX body font (default 10pt)

def configure_latex_fonts(base_pt: int = 10, font_family: str = "Times") -> None:
    import matplotlib
    matplotlib.use("pgf")
    sizes = {
        "title": base_pt,       # e.g., 10
        "label": base_pt - 1,   # e.g., 9
        "tick":  base_pt - 2,   # e.g., 8
        "legend": base_pt - 2,  # e.g., 8
    }
    matplotlib.rcParams.update({
        "backend": "pgf",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": [font_family],
        "pgf.preamble": "\n".join([
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{times}",
        ]),
        "axes.titlesize": sizes["title"],
        "axes.labelsize": sizes["label"],
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
        "legend.title_fontsize": sizes["legend"],
    })

# Initialize defaults (can be overridden by CLI args later in main())
configure_latex_fonts()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402


def _read_csv_maybe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _read_jsonl_rows(p: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _compute_overall_if_missing(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall tables from per-sample if precomputed overall CSVs are missing."""
    # Try to import helpers
    try:
        from train_repeat.src.evaluation.offline_quick_stats import (
            deciles_vs_acc,
            removed_len_vs_acc,
            zx_vs_acc,
        )
    except ModuleNotFoundError:
        # Allow running this file directly via absolute path by injecting project root
        import sys as _sys
        from pathlib import Path as _Path
        _project_root = _Path(__file__).resolve().parents[3]  # .../RL
        if str(_project_root) not in _sys.path:
            _sys.path.insert(0, str(_project_root))
        from train_repeat.src.evaluation.offline_quick_stats import (
            deciles_vs_acc,
            removed_len_vs_acc,
            zx_vs_acc,
        )

    # Load per-sample
    csv_path = results_dir / "per_sample_extended.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        cj = results_dir / "extended_metrics_correct.jsonl"
        wj = results_dir / "extended_metrics_wrong.jsonl"
        if not (cj.exists() and wj.exists()):
            raise FileNotFoundError(
                "missing overall CSVs and per-sample sources (per_sample_extended.csv or extended_metrics_*.jsonl)"
            )
        rows = _read_jsonl_rows(cj) + _read_jsonl_rows(wj)
        df = pd.DataFrame(rows)

    dec_all = deciles_vs_acc(df, group_col=None)
    rt_all = removed_len_vs_acc(df, group_col=None)
    zx_all = zx_vs_acc(df, group_col=None)
    return dec_all, rt_all, zx_all


def _plot_deciles(df: pd.DataFrame, out_path: Path, width_in: float, aspect: float = 0.55) -> None:
    plt.figure(figsize=(width_in, width_in * aspect))
    x = df["decile"].astype(int)
    y = df["accuracy"].astype(float)
    ylo = df["ci95_lower"].astype(float)
    yhi = df["ci95_upper"].astype(float)
    plt.plot(x, y, marker="o", lw=1.6, color="#1f77b4")
    plt.fill_between(x, ylo, yhi, color="#1f77b4", alpha=0.18, linewidth=0)
    plt.xticks(range(1, int(x.max()) + 1))
    plt.ylim(0.0, 1.0)
    plt.xlabel(r"$\Delta \mathcal{L}$ decile (overall)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_bars(df: pd.DataFrame, xcol: str, xlabel: str, out_path: Path, width_in: float, aspect: float = 0.55) -> None:
    plt.figure(figsize=(width_in, width_in * aspect))
    order = list(df[xcol].unique())
    if xcol == "removed_tokens_bin":
        order = ["0", "1-5", "6-10", "11-20", "21+"]
    df2 = df.set_index(xcol).reindex(order).reset_index()
    xs = range(len(df2))
    y = df2["accuracy"].astype(float)
    ylo = df2["ci95_lower"].astype(float)
    yhi = df2["ci95_upper"].astype(float)
    plt.bar(xs, y, color="#ff7f0e", alpha=0.85)
    err = [y - ylo, yhi - y]
    plt.errorbar(xs, y, yerr=err, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    plt.xticks(list(xs), order)
    plt.ylim(0.0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def auto_copy_to_latex_project(source_path: Path, target_dir: Path = None) -> bool:
    """
    Automatically copy the generated figure to the LaTeX project directory.
    
    Args:
        source_path: Path to the generated figure file
        target_dir: Target directory (defaults to LaTeX project)
    
    Returns:
        bool: True if copy was successful, False otherwise
    """
    if target_dir is None:
        # Default target: the LaTeX project directory (override via --latex_dir)
        target_dir = Path("/path/to/your/latex_project")
    
    if not source_path.exists():
        print(f"Warning: Source file {source_path} does not exist")
        return False
    
    if not target_dir.exists():
        print(f"Warning: Target directory {target_dir} does not exist")
        return False
    
    target_path = target_dir / source_path.name
    
    try:
        shutil.copy2(source_path, target_path)
        print(f"✓ Auto-copied figure: {source_path} → {target_path}")
        return True
    except Exception as e:
        print(f"Error copying figure: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot offline stats into PDFs")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--fmt", default="pgf", choices=["pgf", "pdf", "svg", "png"], help="Output format")
    # New: figure/typography controls
    ap.add_argument("--width_in", type=float, default=5.5, help="Figure width in inches (match LaTeX \\linewidth for single column)")
    ap.add_argument("--height_ratio", type=float, default=0.35, help="Height/width ratio for the combined figure")
    ap.add_argument("--base_pt", type=int, default=10, help="LaTeX body font size in pt (10/11/12)")
    ap.add_argument("--font_family", default="Times", help="Serif font family to match LaTeX (Times/Palatino/…) ")
    # Auto-copy configuration
    ap.add_argument("--auto_copy", action="store_true", default=True, help="Automatically copy figure1.pgf to LaTeX project")
    ap.add_argument("--latex_dir", default="/path/to/your/latex_project", help="Target LaTeX project directory")
    args = ap.parse_args()

    # Apply typography now that CLI is parsed
    configure_latex_fonts(base_pt=args.base_pt, font_family=args.font_family)

    rd = Path(args.results_dir).expanduser().resolve()
    # 1) Always produce two mechanism-centric figures (stronger for paper)
    #    a) removed_prefix_length_distribution.csv → stacked/side-by-side bars per group/bin
    #    b) length_stratified_summary.csv → per-bin metric bars (delta_bar_per_token, suffix gap)
    # --- Combined Figure Generation ---
    rdist = _read_csv_maybe(rd / "removed_prefix_length_distribution.csv")
    lstrat = _read_csv_maybe(rd / "length_stratified_summary.csv")

    if rdist is not None and not rdist.empty and lstrat is not None and not lstrat.empty:
        fig_w = args.width_in
        fig_h = args.width_in * args.height_ratio
        # Create a simple 1x2 subplot layout without separate legend area
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        ax0, ax1 = axes
        # Add spacing between subplots and increase bottom margin for rotated x-ticks
        plt.subplots_adjust(wspace=0.3, bottom=0.22)

        colors = ["#1f77b4", "#ff7f0e"]  # Default blue and orange
        hatches = ["//", "xx"]
        edgecolor = "black"
        order_bins = ["0", "1-5", "6-10", "11-20", "21+"]
        # Derive a slightly smaller x-tick label size to reduce overlap when rotated
        try:
            _xtick_base = plt.rcParams.get("xtick.labelsize", 8)
            if isinstance(_xtick_base, (int, float)):
                xtick_smaller = max(1.0, float(_xtick_base) - 1.0)
            else:
                xtick_smaller = 7.0
        except Exception:
            xtick_smaller = 7.0

        # --- (a) Distribution ---
        ax = axes[0]
        rdist["count"] = pd.to_numeric(rdist["count"], errors="coerce").fillna(0).astype(int)
        rdist["bin"] = pd.Categorical(rdist["bin"], categories=order_bins, ordered=True)
        pivot = rdist.pivot_table(index="bin", columns="group", values="count", aggfunc="sum", fill_value=0, observed=False)
        pivot = pivot.loc[order_bins]
        props = pivot / pivot.sum(axis=0).replace(0, 1)

        groups = props.columns
        n_groups = len(groups)
        xs = np.arange(len(order_bins))
        width = 0.8 / n_groups

        for i, group in enumerate(groups):
            offset = (i - (n_groups - 1) / 2) * width
            vals = pd.to_numeric(props[group], errors="coerce").fillna(0.0).values
            ax.bar(
                xs + offset,
                vals,
                width=width,
                label=group,
                color=colors[i],
                hatch=hatches[i],
                edgecolor=edgecolor,
                alpha=0.8,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(order_bins, rotation=45, ha="right", fontsize=xtick_smaller)
        ax.set_ylabel("Proportion within group")
        # Remove panel title/label; rely on figure caption
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        # Add legend inside the first subplot on the left
        ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
                  handlelength=1.2, columnspacing=1.0, handletextpad=0.5)

        # --- (b) Per-token Gap & (c) Suffix-only Gap ---
        lstrat["bin"] = pd.Categorical(lstrat["bin"], categories=order_bins, ordered=True)
        groups = sorted(lstrat["group"].dropna().unique())
        bar_width = 0.38
        xs = np.arange(len(order_bins))

        # Plot (b)
        ax = axes[1]
        metric, ylabel = ("delta_bar_per_token", r"$\Delta \mathcal{L}$ per token")
        if metric in lstrat.columns:
            for i, g in enumerate(groups):
                sub = lstrat[lstrat["group"] == g].set_index("bin").reindex(order_bins)
                vals = pd.to_numeric(sub[metric], errors="coerce").fillna(0.0).values
                bar_pos = [x + (i - (len(groups) - 1) / 2) * bar_width for x in xs]
                ax.bar(
                    bar_pos,
                    vals,
                    width=bar_width,
                    label=g,
                    color=colors[i],
                    hatch=hatches[i],
                    edgecolor=edgecolor,
                    alpha=0.8,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(order_bins, rotation=45, ha="right", fontsize=xtick_smaller)
            ax.set_ylabel(ylabel)
            # Remove panel label; rely on figure caption
            ax.grid(axis="y", linestyle=":", alpha=0.7)
            # Add legend inside the second subplot on the left
            ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False,
                      handlelength=1.2, columnspacing=1.0, handletextpad=0.5)

        # Plot (c)
        # ax = axes[2]
        # metric, ylabel = ("delta_suffix_mean", "Suffix-only gap")
        # if metric in lstrat.columns:
        #     for i, g in enumerate(groups):
        #         sub = lstrat[lstrat["group"] == g].set_index("bin").reindex(order_bins)
        #         vals = pd.to_numeric(sub[metric], errors="coerce").fillna(0.0).values
        #         bar_pos = [x + (i - (len(groups) - 1) / 2) * bar_width for x in xs]
        #         ax.bar(
        #             bar_pos,
        #             vals,
        #             width=bar_width,
        #             label=g,
        #             color=colors[i],
        #             hatch=hatches[i],
        #             edgecolor=edgecolor,
        #             alpha=0.8,
        #         )
        #
        #     ax.set_xticks(xs)
        #     ax.set_xticklabels(order_bins, rotation=45, ha="right")
        #     ax.set_ylabel(ylabel)
        #     # Remove panel label; rely on figure caption
        #     ax.grid(axis="y", linestyle=":", alpha=0.7)

        # Figure-wide x label (place inside figure bottom to avoid overlap/occlusion)
        fig.text(0.5, 0.02, "Removed echo tokens (count bin)", ha="center", va="center")

        out_path = rd / f"figure_1_combined.{args.fmt}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print("Saved combined figure:", out_path)
        
        # Auto-copy to LaTeX project directory
        if args.auto_copy:
            auto_copy_to_latex_project(out_path, Path(args.latex_dir))
    else:
        print("Skipping combined figure generation: one or more required CSV files are missing.")

    # 2) Echo Gap main table → LaTeX (booktabs-ready, English headers)
    main_tbl = _read_csv_maybe(rd / "comparison_table.csv")
    if main_tbl is not None and not main_tbl.empty:
        # Expected Chinese headers in CSV → map to English
        col_map = {
            "组别": "Group",
            "样本数": "N",
            "平均Δ对数概率": "Mean $\\overline{\\Delta\\mathcal{L}}$",
            "标准差": "Std $\\sigma(\\Delta\\mathcal{L})$",
            "负值比例": "Neg. ratio (\%)",
        }
        need = ["组别", "样本数", "平均Δ对数概率", "标准差", "负值比例"]
        if all(c in main_tbl.columns for c in need):
            df_en = main_tbl[need].copy()
            # Map group names
            df_en["组别"] = df_en["组别"].map({"正确答案组": "Correct", "错误答案组": "Wrong"}).fillna(df_en["组别"])
            # Parse percentage strings like "0.12%" to numeric with 2 decimals
            def _pct_to_num(x):
                try:
                    s = str(x).strip().replace("%", "")
                    return float(s)
                except Exception:
                    return x
            df_en["负值比例"] = df_en["负值比例"].map(_pct_to_num)

            tex_dir = rd / "latex_tables"
            tex_dir.mkdir(parents=True, exist_ok=True)
            # Build LaTeX lines
            header_en = [col_map[c] for c in need]
            lines = [
                "\\begin{table}[t]",
                "  \\centering",
                "  \\caption{Echo Likelihood Gap on GSM8K (per-token).}",
                "  \\label{tab:echo_gap}",
                "  \\begin{tabular}{lrrrr}",
                "    \\toprule",
                "    " + " & ".join(header_en) + " \\ ",
                "    \\midrule",
            ]
            for _, r in df_en.iterrows():
                group = str(r["组别"])  # Correct/Wrong
                n = int(r["样本数"]) if pd.notna(r["样本数"]) else r["样本数"]
                mean = float(r["平均Δ对数概率"]) if pd.notna(r["平均Δ对数概率"]) else r["平均Δ对数概率"]
                std = float(r["标准差"]) if pd.notna(r["标准差"]) else r["标准差"]
                neg = r["负值比例"]
                if isinstance(neg, float):
                    neg_str = f"{neg:.2f}"
                else:
                    neg_str = str(neg)
                lines.append(
                    "    "
                    + " & ".join([
                        group,
                        f"{n}",
                        f"{mean:.4f}",
                        f"{std:.4f}",
                        neg_str,
                    ])
                    + " \\ "
                )
            lines += [
                "    \\bottomrule",
                "  \\end{tabular}",
                "\\end{table}",
            ]
            (tex_dir / "echo_gap_table.tex").write_text("\n".join(lines), encoding="utf-8")
            print("Saved:", tex_dir / "echo_gap_table.tex")

    # 3) Optional trends (kept for appendix)
    # Try to read overall CSVs
    dec = _read_csv_maybe(rd / "deltaL_deciles_vs_acc_overall.csv")
    rt = _read_csv_maybe(rd / "removed_tokens_bin_vs_acc_overall.csv")
    zx = _read_csv_maybe(rd / "zx_vs_acc_overall.csv")

    if dec is None or rt is None or zx is None:
        dec, rt, zx = _compute_overall_if_missing(rd)

    # Plot
    small_w = args.width_in * 0.64  # side figures ~2/3 of column width
    _plot_deciles(dec, rd / f"deciles_vs_acc_overall.{args.fmt}", width_in=small_w)
    _plot_bars(rt, "removed_tokens_bin", "Removed tokens (bin)", rd / f"removed_bins_vs_acc_overall.{args.fmt}", width_in=small_w)
    _plot_bars(zx, "Zx", "Acceptance Zx (0/1)", rd / f"zx_vs_acc_overall.{args.fmt}", width_in=small_w)
    print("Saved (appendix trends):")
    print(" -", rd / f"deciles_vs_acc_overall.{args.fmt}")
    print(" -", rd / f"removed_bins_vs_acc_overall.{args.fmt}")
    print(" -", rd / f"zx_vs_acc_overall.{args.fmt}")


if __name__ == "__main__":
    main()
