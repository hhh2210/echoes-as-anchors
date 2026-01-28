#!/usr/bin/env python3
"""
Minimal post-processing on existing CSVs to produce paper-ready stats.

Inputs (under --results_dir):
- per_sample_extended.csv (required; columns include: group, is_correct, delta_bar_per_token, removed_tokens, trim_token_count, ...)

Outputs (to --results_dir):
- deltaL_deciles_vs_acc.csv
- removed_tokens_bin_vs_acc.csv
- zx_vs_acc.csv
- spearman_correlation.txt

Design goals:
- Keep <150 lines
- Reuse existing helpers where possible (binning removed_tokens)
- No heavy dependencies; pure pandas
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import json

try:
    # Reuse the same binning as other modules if available
    from train_repeat.src.evaluation.compare_trimmed_accuracy import _bin_removed_tokens  # type: ignore
except Exception:
    def _bin_removed_tokens(n: int) -> str:  # fallback
        if n is None:
            return "NA"
        try:
            n = int(n)
        except Exception:
            return "NA"
        if n <= 0:
            return "0"
        if n <= 5:
            return "1-5"
        if n <= 10:
            return "6-10"
        if n <= 20:
            return "11-20"
        return "21+"


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).astype(int).astype(bool)
    lower = s.astype(str).str.lower()
    return lower.isin(["1", "true", "yes", "y"]) & ~lower.isin(["0", "false", "no", "n", "nan", "none"]) 


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def deciles_vs_acc(df: pd.DataFrame, delta_col: str = "delta_bar_per_token", group_col: Optional[str] = "group", bins: int = 10) -> pd.DataFrame:
    sub = df[[delta_col, "is_correct"] + ([group_col] if group_col in df.columns and group_col else [])].dropna(subset=[delta_col]).copy()
    sub["is_correct"] = _to_bool(sub["is_correct"]).astype(int)
    qs = [float(sub[delta_col].quantile(i / bins)) for i in range(1, bins + 1)]

    def assign(v: float) -> int:
        for i, q in enumerate(qs, start=1):
            if v <= q:
                return i
        return bins

    sub["decile"] = sub[delta_col].astype(float).map(assign)
    keys = ["decile"] + ([group_col] if group_col and group_col in sub.columns else [])
    rows: List[Dict[str, object]] = []
    for k, g in sub.groupby(keys):
        if not isinstance(k, tuple):
            k = (k,)
        decile = int(k[0])
        grp = k[1] if len(k) > 1 else "all"
        n = int(len(g))
        c = int(g["is_correct"].sum())
        acc = c / n if n else float("nan")
        lo, hi = _wilson_ci(c, n)
        lo_b = qs[decile - 2] if decile > 1 else float("-inf")
        hi_b = qs[decile - 1]
        rows.append({"group": grp, "decile": decile, "lower_bound": lo_b, "upper_bound": hi_b, "n_samples": n, "num_correct": c, "accuracy": acc, "ci95_lower": lo, "ci95_upper": hi})
    return pd.DataFrame(rows).sort_values(["group", "decile"]).reset_index(drop=True)


def removed_len_vs_acc(df: pd.DataFrame, removed_col: str = "removed_tokens", group_col: Optional[str] = "group") -> pd.DataFrame:
    sub = df[[removed_col, "is_correct"] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    sub["is_correct"] = _to_bool(sub["is_correct"]).astype(int)
    sub[removed_col] = pd.to_numeric(sub[removed_col], errors="coerce").fillna(0).astype(int)
    sub["bin"] = sub[removed_col].map(_bin_removed_tokens)
    keys = ["bin"] + ([group_col] if group_col and group_col in sub.columns else [])
    rows: List[Dict[str, object]] = []
    for k, g in sub.groupby(keys):
        if not isinstance(k, tuple):
            k = (k,)
        b = str(k[0])
        grp = k[1] if len(k) > 1 else "all"
        n = int(len(g))
        c = int(g["is_correct"].sum())
        acc = c / n if n else float("nan")
        lo, hi = _wilson_ci(c, n)
        rows.append({"group": grp, "removed_tokens_bin": b, "n_samples": n, "num_correct": c, "accuracy": acc, "ci95_lower": lo, "ci95_upper": hi})
    return pd.DataFrame(rows).sort_values(["group", "removed_tokens_bin"]).reset_index(drop=True)


def zx_vs_acc(df: pd.DataFrame, removed_col: str = "removed_tokens", group_col: Optional[str] = "group") -> pd.DataFrame:
    sub = df[[removed_col, "is_correct"] + ([group_col] if group_col and group_col in df.columns else [])].copy()
    sub["is_correct"] = _to_bool(sub["is_correct"]).astype(int)
    sub[removed_col] = pd.to_numeric(sub[removed_col], errors="coerce").fillna(0).astype(int)
    sub["Zx"] = (sub[removed_col] > 0).astype(int)
    keys = ["Zx"] + ([group_col] if group_col and group_col in sub.columns else [])
    rows: List[Dict[str, object]] = []
    for k, g in sub.groupby(keys):
        if not isinstance(k, tuple):
            k = (k,)
        z = int(k[0])
        grp = k[1] if len(k) > 1 else "all"
        n = int(len(g))
        c = int(g["is_correct"].sum())
        acc = c / n if n else float("nan")
        lo, hi = _wilson_ci(c, n)
        rows.append({"group": grp, "Zx": z, "n_samples": n, "num_correct": c, "accuracy": acc, "ci95_lower": lo, "ci95_upper": hi})
    return pd.DataFrame(rows).sort_values(["group", "Zx"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick offline stats (ΔL/removed/Zx vs accuracy)")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--decile_bins", type=int, default=10)
    args = ap.parse_args()

    rd = Path(args.results_dir).expanduser().resolve()
    csv_path = rd / "per_sample_extended.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        correct_jsonl = rd / "extended_metrics_correct.jsonl"
        wrong_jsonl = rd / "extended_metrics_wrong.jsonl"
        if not (correct_jsonl.exists() and wrong_jsonl.exists()):
            raise FileNotFoundError("per_sample_extended.csv 或 extended_metrics_*.jsonl 均未找到")

        def _read_jsonl(p: Path) -> List[Dict[str, object]]:
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

        rows = _read_jsonl(correct_jsonl) + _read_jsonl(wrong_jsonl)
        if not rows:
            raise RuntimeError("extended_metrics_*.jsonl 为空或解析失败")
        df = pd.DataFrame(rows)

    # Per-group deciles (kept for completeness)
    dec = deciles_vs_acc(df, bins=args.decile_bins)
    dec.to_csv(rd / "deltaL_deciles_vs_acc.csv", index=False)
    # Overall deciles (aggregate across groups) for trend visualization
    dec_all = deciles_vs_acc(df, bins=args.decile_bins, group_col=None)
    dec_all.to_csv(rd / "deltaL_deciles_vs_acc_overall.csv", index=False)

    rl = removed_len_vs_acc(df)
    rl.to_csv(rd / "removed_tokens_bin_vs_acc.csv", index=False)
    rl_all = removed_len_vs_acc(df, group_col=None)
    rl_all.to_csv(rd / "removed_tokens_bin_vs_acc_overall.csv", index=False)

    zx = zx_vs_acc(df)
    zx.to_csv(rd / "zx_vs_acc.csv", index=False)
    zx_all = zx_vs_acc(df, group_col=None)
    zx_all.to_csv(rd / "zx_vs_acc_overall.csv", index=False)

    # Spearman via pandas
    s = pd.DataFrame({"x": pd.to_numeric(df.get("delta_bar_per_token"), errors="coerce"), "y": _to_bool(df.get("is_correct", False)).astype(int) }).dropna()
    rho = float(s["x"].corr(s["y"], method="spearman")) if not s.empty else float("nan")
    (rd / "spearman_correlation.txt").write_text(f"Spearman(delta_bar_per_token, is_correct) = {rho:.6f}\n", encoding="utf-8")

    print("Wrote:")
    print(" -", rd / "deltaL_deciles_vs_acc.csv")
    print(" -", rd / "deltaL_deciles_vs_acc_overall.csv")
    print(" -", rd / "removed_tokens_bin_vs_acc.csv")
    print(" -", rd / "removed_tokens_bin_vs_acc_overall.csv")
    print(" -", rd / "zx_vs_acc.csv")
    print(" -", rd / "zx_vs_acc_overall.csv")
    print(" -", rd / "spearman_correlation.txt")


if __name__ == "__main__":
    main()


