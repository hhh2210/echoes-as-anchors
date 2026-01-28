import json
import matplotlib.pyplot as plt
import argparse
import os
import shutil
from pathlib import Path
"""
python train_repeat/src/visualization/plot_repeat_summary.py --json_path train_repeat/results/direct_openai_mlp/repeat_summary.json --output_path train_repeat/results/direct_openai_mlp/repeat_summary.pgf
"""

# It's recommended to set this path via an argument or config file.
LATEX_PROJECT_DIR = "/path/to/your/latex_project"


def _auto_copy_to_latex_project(source_path: Path, target_dir: Path | None = None) -> bool:
    if target_dir is None:
        target_dir = Path(LATEX_PROJECT_DIR)
    if not source_path.exists():
        print(f"Warning: Source file {source_path} does not exist")
        return False
    if not target_dir.exists():
        print(f"Warning: Target directory {target_dir} does not exist")
        return False
    target_path = target_dir / source_path.name
    try:
        shutil.copy2(source_path, target_path)
        print(f"\u2713 Auto-copied figure: {source_path} \u2192 {target_path}")
        return True
    except Exception as e:
        print(f"Error copying figure: {e}")
        return False


def _compute_iclr_halfwidth_figsize() -> tuple[float, float]:
    """Return a figsize (width, height) in inches for ICLR half-column.

    Uses environment overrides when provided:
    - TARGET_FIG_WIDTH_IN: directly specify width in inches
    - ICLR_TEXTWIDTH_IN: full text width (default 5.5in)
    - SUBFIG_FRACTION: fraction of text width used (default 0.48)
    - FIG_ASPECT: height/width ratio (default 0.75)
    """
    # direct width override
    target_width_env = os.environ.get("TARGET_FIG_WIDTH_IN")
    if target_width_env is not None:
        try:
            width_in = float(target_width_env)
        except ValueError:
            width_in = 2.64
    else:
        textwidth_in = float(os.environ.get("ICLR_TEXTWIDTH_IN", "5.5"))
        subfig_fraction = float(os.environ.get("SUBFIG_FRACTION", "0.48"))
        width_in = textwidth_in * subfig_fraction
    aspect = float(os.environ.get("FIG_ASPECT", "0.75"))
    height_in = width_in * aspect
    return width_in, height_in


def plot_repeat_summary(json_path, output_path, fmt: str = "pdf"):
    """
    Generates a bar chart with diagonal stripes from a repeat summary JSON file.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path to save the output plot image.
    """
    # Set style (10pt to match ICLR body font). Keep simple to avoid surprises.
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = data.get('stats', [])
    if not stats:
        print("No stats found in the JSON file.")
        return

    models = [s['model'] for s in stats]
    repeat_frequencies = [s['repeat_frequency'] for s in stats]

    # Figure size aligned to ICLR half-column to avoid LaTeX scaling
    fig_width_in, fig_height_in = _compute_iclr_halfwidth_figsize()
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # Modern color palette with gradients
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    edge_colors = ["#2980b9", "#c0392b", "#27ae60", "#d68910", "#8e44ad"]
    
    xs = range(len(models))
    bar_colors = [colors[i % len(colors)] for i in xs]
    bar_edge_colors = [edge_colors[i % len(edge_colors)] for i in xs]

    bars = []
    for i, (x, y) in enumerate(zip(xs, repeat_frequencies)):
        b = ax.bar(models[i], y, 
                   color=bar_colors[i], 
                   edgecolor=bar_edge_colors[i], 
                   linewidth=2.5,
                   alpha=0.8,
                   width=0.6)
        bars.extend(b)
        
        # Add subtle gradient effect using overlaid bar
        b2 = ax.bar(models[i], y * 0.95, 
                    color=bar_colors[i], 
                    edgecolor='none',
                    alpha=0.3,
                    width=0.6)

    ax.set_ylabel('Frequency')
    ax.set_title('EOP Frequency by Model', pad=6)
    ax.set_ylim(0, 1.0)
    
    # Style the axes
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(axis='both', which='major', width=1.0, length=4)
    
    # Add horizontal grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.2f}'
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Model names styling
    plt.xticks(rotation=15, ha="right")
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Normalize extension to requested fmt
    out_path_obj = Path(output_path)
    if out_path_obj.suffix.lower() != f".{fmt}":
        out_path_obj = out_path_obj.with_suffix(f".{fmt}")
    plt.savefig(out_path_obj, dpi=300)
    print(f"Plot saved to {out_path_obj}")
    plt.close(fig)

    # Auto copy
    _auto_copy_to_latex_project(out_path_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot repeat summary from a JSON file.")
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the repeat_summary.json file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output plot."
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg", "pgf"],
        help="Output format (default: pdf)"
    )
    args = parser.parse_args()

    plot_repeat_summary(args.json_path, args.output_path, fmt=args.fmt)
