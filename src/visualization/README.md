# Visualization Module

## Overview

This module provides a comprehensive set of visualization tools for analyzing and presenting training processes, experimental results, and model performance. It helps understand the impact of repetition behavior (Echo/Repeat) on model reasoning ability through intuitive charts, supporting publication-quality figure generation.

## File Structure

```
visualization/
├── plot_results.py           # Experiment results comparison visualization
├── plot_loss.py              # Training loss curve plotting
├── plot_reward.py            # Reward distribution visualization
├── plot_offline_stats.py     # Offline statistical analysis charts (generates paper main figures)
├── plot_repeat_summary.py    # Repetition frequency statistics charts
├── attention_case_viz.py     # Attention mechanism case study visualization
├── plot_attention_layers.py  # Layer-wise attention trajectory analysis
├── mi_peaks.py              # Mutual information peak analysis
└── __init__.py              # Module initialization
```

## Core Features

### 1. Experiment Results Visualization (`plot_results.py`)

**Function**: Plot model performance comparison under different w_repeat parameters.

**Main Features**:
- Support multi-experiment comparison
- Dual-metric display (flexible-extract & strict-match)
- Automatic experiment directory scanning
- Generate publication-ready charts

**Core Functions**:
- `load_exp_results()`: Load experiment data
- `plot_accuracy_comparison()`: Plot accuracy comparison
- `generate_summary_table()`: Generate statistics table

**Usage Examples**:
```bash
# Compare all w_repeat experiments
python plot_results.py \
    --exp_dirs experiments/w_repeat_* \
    --output results/comparison.png

# Specify specific experiments
python plot_results.py \
    --exp_dirs experiments/w_repeat_0.0 experiments/w_repeat_-0.05 \
    --title "Impact of Repeat Penalty on GSM8K Performance" \
    --output results/repeat_impact.png
```

**Output Example**:
- Dual Y-axis chart showing two evaluation metrics
- X-axis: Training steps
- Y-axis: Accuracy (%)
- Different colored curves represent different w_repeat values

### 2. Training Loss Visualization (`plot_loss.py`)

**Function**: Track and visualize loss changes during training.

**Main Features**:
- Real-time update support
- Multiple loss type display
- Moving average smoothing
- Batch experiment comparison

**Supported Loss Types**:
- Total Loss
- KL Divergence
- Policy Loss
- Value Loss (if applicable)

**Usage Examples**:
```bash
# Single experiment loss curve
python plot_loss.py \
    --log_file experiments/grpo_repeat/loss_log.jsonl \
    --output loss_curve.png

# Multi-experiment comparison
python plot_loss.py \
    --log_files exp1/loss_log.jsonl exp2/loss_log.jsonl \
    --labels "Baseline" "With Repeat Penalty" \
    --smooth_window 10 \
    --output loss_comparison.png
```

**Configurable Parameters**:
- `--smooth_window`: Moving average window size
- `--log_scale`: Use logarithmic scale
- `--start_step`: Starting step (skip initial instability)

### 3. Reward Distribution Visualization (`plot_reward.py`)

**Function**: Analyze and visualize reward distribution during training.

**Main Features**:
- Reward component decomposition display
- Distribution histogram
- Time series trend analysis
- Correlation analysis

**Reward Components**:
- Total Reward
- Correctness Reward
- Repeat Reward
- Combined Reward

**Usage Examples**:
```bash
# Reward distribution analysis
python plot_reward.py \
    --reward_log experiments/grpo_repeat/reward_log.jsonl \
    --output_dir results/reward_analysis/

# Time series trend chart
python plot_reward.py \
    --reward_log reward_log.jsonl \
    --plot_type timeline \
    --components all \
    --output reward_timeline.png

# Distribution histogram
python plot_reward.py \
    --reward_log reward_log.jsonl \
    --plot_type histogram \
    --bins 50 \
    --output reward_distribution.png
```

## Advanced Features

### Combined Visualization

Create comprehensive analysis charts with multiple subplots:

```python
from visualization import create_combined_plot

create_combined_plot(
    exp_dirs=["exp1/", "exp2/", "exp3/"],
    metrics=["accuracy", "loss", "reward"],
    output="comprehensive_analysis.png"
)
```

### Interactive Charts

Generate interactive HTML charts:

```bash
# Generate interactive charts using plotly
python plot_results.py \
    --exp_dirs experiments/* \
    --interactive \
    --output results/interactive.html
```

### Statistical Analysis

Generate detailed statistical reports:

```bash
# Generate statistical summary
python plot_results.py \
    --exp_dirs experiments/* \
    --generate_stats \
    --output_csv results/statistics.csv
```

## Output Formats

### Image Output
- PNG (default, 300 DPI)
- SVG (vector graphics)
- PDF (publication quality)

### Data Output
- CSV (statistical tables)
- JSON (raw data)
- HTML (interactive reports)

## Style Configuration

### Custom Color Schemes

```python
# Configure in script
COLOR_SCHEME = {
    "baseline": "#1f77b4",
    "repeat_penalty": "#ff7f0e",
    "repeat_reward": "#2ca02c"
}
```

### Chart Styles

```python
# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# Or use custom style
PLOT_STYLE = {
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
}
```

## Batch Processing

### Generate All Charts

```bash
# One-click generation of all visualizations
./generate_all_plots.sh experiments/
```

### Automatic Report Generation

```python
from visualization import generate_experiment_report

generate_experiment_report(
    exp_dir="experiments/grpo_repeat/",
    output_dir="reports/",
    include_plots=True,
    include_stats=True,
    format="html"  # or "pdf", "markdown"
)
```

## Advanced Visualization Features

### 4. Offline Statistical Analysis Charts (`plot_offline_stats.py`)

**Function**: Generate publication-grade statistical analysis charts, including Echo Likelihood Gap analysis.

**Main Features**:
- Generate LaTeX-compatible PGF format charts
- Length-stratified analysis
- Automatic copy to LaTeX project directory
- Support multiple output formats (pgf, pdf, svg, png)

**Core Outputs**:
- `figure_1_combined.pgf`: Paper main figure, containing:
  - (a) Distribution of removed tokens
  - (b) Echo Likelihood Gap per token
- LaTeX tables (Echo Gap statistics)

**Usage Example**:
```bash
# Generate paper main figure (PGF format)
python src/visualization/plot_offline_stats.py \
  --results_dir analysis_results_20250809_141519_extra_v2 \
  --fmt pgf \
  --auto_copy \
  --latex_dir /path/to/latex/project
```

**Configuration Parameters**:
- `--width_in`: Chart width (inches, matches LaTeX column width)
- `--height_ratio`: Height-to-width ratio
- `--base_pt`: LaTeX font size (10/11/12 pt)
- `--font_family`: Font family (Times/Palatino)

### 5. Repetition Frequency Statistics (`plot_repeat_summary.py`)

**Function**: Visualize EOP (End of Preamble) frequency for different models.

**Main Features**:
- Modern bar chart design
- Gradient color effects
- Automatic value labels
- Support multiple output formats

**Usage Example**:
```bash
python src/visualization/plot_repeat_summary.py \
  --json_path results/direct_openai_mlp/repeat_summary.json \
  --output_path results/repeat_summary.pgf \
  --fmt pgf
```

### 6. Mutual Information Peak Analysis (`mi_peaks.py`)

**Function**: Analyze and visualize mutual information peaks between tokens.

**Main Features**:
- Peak detection algorithm
- Distribution analysis
- Correlation analysis with model performance

## Attention Visualization

### 7. Single Sample Attention Visualization (`attention_case_viz.py`)

**Function**: Visualize attention mechanism for specified sample, layer, and head.

Example:
```bash
python src/visualization/attention_case_viz.py \
  --model /path/to/DeepSeek-R1-Distill-Llama-8B/ \
  --logp_results_json analysis_results/correct_logp_results.json \
  --idx 0 \
  --layer -1 --head 0 \
  --query_answer_index -1 \
  --max_seq_len 768 \
  --heatmap_region answer \
  --output_dir analysis_results/attn_cases \
  --save_png
```

Notes:
- Generates HTML (token-by-token coloring) and PNG heatmap (default answer region only)
- Title shows aggregated metrics consistent with paper: answer→question / answer→answer-prefix(K=32)

### 8. Layer-wise Attention Trajectory Analysis (`plot_attention_layers.py`)

**Function**: Analyze attention pattern changes across layers.

**Export per-layer metrics**:
```bash
python src/evaluation/attention_from_converted.py \
  --correct_converted correct_converted.jsonl \
  --wrong_converted wrong_converted.jsonl \
  --model /path/to/model/ \
  --answer_prefix_tokens 32 \
  --per_layer_trajectories \
  --output_dir analysis_results/attn_layers
```

**Plot trajectory**:
```bash
python src/visualization/plot_attention_layers.py \
  --metrics_dir analysis_results/attn_layers \
  --output fig_layers.pdf \
  --focus_layers 16 24
```

**Output Results**:
- Two curve plots: Answer→Question & Answer→Answer-prefix (mean±standard error)
- Terminal prints focus layer segment (e.g., 16-24) group difference Δ(C-W) percentage

## Best Practices

### Data Preprocessing
- Remove outliers
- Handle missing data
- Standardize comparison baselines

### Chart Design Principles
- Use clear labels and legends
- Choose appropriate chart types
- Maintain consistent color schemes
- Generate LaTeX-compatible chart formats

### Performance Optimization
- Use sampling for large datasets
- Cache preprocessing results
- Parallel processing for multiple charts

## Paper Figure Generation Workflow

### Generate Figure 1 (Echo Likelihood Gap Analysis)
```bash
# 1. Run offline analysis
python src/evaluation/offline_quick_stats.py \
  --results_dir analysis_results_20250809_141519_extra_v2

# 2. Generate PGF format charts
python src/visualization/plot_offline_stats.py \
  --results_dir analysis_results_20250809_141519_extra_v2 \
  --fmt pgf \
  --auto_copy

# 3. Reference in LaTeX
# \input{figure_1_combined.pgf}
```

### Generate Attention Analysis Figures
```bash
# 1. Calculate attention metrics
python src/evaluation/attention_from_converted.py \
  --correct_converted correct_converted.jsonl \
  --wrong_converted wrong_converted.jsonl \
  --model /path/to/your/model/ \
  --per_layer_trajectories

# 2. Plot layer-wise trajectories
python src/visualization/plot_attention_layers.py \
  --metrics_dir attn_layers \
  --output fig_layers.pdf
```

## Dependencies

```python
matplotlib >= 3.5.0
numpy >= 1.21.0
pandas >= 1.3.0
seaborn >= 0.11.0  # Optional, for advanced styles
plotly >= 5.0.0    # Optional, for interactive charts
scipy >= 1.7.0     # For statistical analysis
```

## FAQ

**Q: Chart displays garbled Chinese characters?**
```python
# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

**Q: How to export high-quality images?**
```python
# Set DPI and format
fig.savefig('output.png', dpi=300, bbox_inches='tight')
```

**Q: Too many data points causing crowded charts?**
```python
# Use data sampling or aggregation
data = data[::10]  # Take 1 out of every 10 points
# Or use moving average
data_smooth = data.rolling(window=10).mean()
```

**Q: How to ensure chart compatibility with LaTeX documents?**
```python
# Use PGF backend
import matplotlib
matplotlib.use('pgf')
# Configure LaTeX fonts
matplotlib.rcParams.update({
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.family': 'serif',
    'font.serif': ['Times'],
})
```

## Changelog

- 2025-01: Added `plot_offline_stats.py`, supports generating publication-grade Echo Likelihood Gap analysis figures
- 2025-01: Added `plot_repeat_summary.py`, visualizes EOP frequency statistics
- 2025-01: Added attention visualization modules (`attention_case_viz.py`, `plot_attention_layers.py`)
- 2025-01: Support automatic copying of charts to LaTeX project directory
- 2025-01: Added PGF format output support, fully compatible with LaTeX documents