#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["rr", "pf", "mr", "fmr_rl"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "FMR-RL"}


def label(mode):
    return LABELS.get(mode, mode)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def discover_bw_dirs(run_dir: Path):
    return sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bw")],
        key=lambda p: int(p.name.replace("bw", "")),
    )


def get_all_rntis(bw_dir: Path):
    rr = pd.read_csv(bw_dir / "rr" / "slot_log_rr.csv")
    return sorted(int(x) for x in rr["rnti"].unique())


def load_mode_matrix(bw_dir: Path, mode: str, all_rntis: list[int]) -> pd.DataFrame:
    path = bw_dir / mode / f"slot_log_{mode}.csv"
    df = pd.read_csv(path)

    required = {"time_s", "beam_id", "rnti", "dl_mcs", "buf_req", "alloc_rbg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    rows = []

    for idx, (_, g) in enumerate(df.groupby(["time_s", "beam_id"], sort=False)):
        slot = (
            g.groupby("rnti")
            .agg(
                dl_mcs=("dl_mcs", "max"),
                buf_req=("buf_req", "max"),
                alloc_rbg=("alloc_rbg", "sum"),
            )
            .reindex(all_rntis, fill_value=0)
        )
        slot["slot_idx"] = idx
        rows.append(slot.reset_index())

    out = pd.concat(rows, ignore_index=True)
    return out


def build_correlations(run_dir: Path) -> pd.DataFrame:
    records = []

    for bw_dir in discover_bw_dirs(run_dir):
        bw = bw_dir.name
        all_rntis = get_all_rntis(bw_dir)

        for mode in MODES:
            path = bw_dir / mode / f"slot_log_{mode}.csv"
            if not path.exists():
                continue

            df = load_mode_matrix(bw_dir, mode, all_rntis)

            for rnti, g in df.groupby("rnti"):
                mcs = g["dl_mcs"].to_numpy()
                alloc = g["alloc_rbg"].to_numpy()
                buf = g["buf_req"].to_numpy()

                # proxy de throughput instantâneo por UE.
                # Como não temos throughput por slot no slot_log,
                # usamos MCS * RBG como proxy de capacidade entregue.
                thr_proxy = mcs * alloc

                records.append({
                    "bandwidth": bw,
                    "mode": mode,
                    "rnti": int(rnti),
                    "corr_mcs_alloc": safe_corr(mcs, alloc),
                    "corr_mcs_thr_proxy": safe_corr(mcs, thr_proxy),
                    "corr_buf_alloc": safe_corr(buf, alloc),
                    "corr_buf_thr_proxy": safe_corr(buf, thr_proxy),
                    "mean_mcs": float(np.mean(mcs)),
                    "mean_alloc": float(np.mean(alloc)),
                    "mean_buf": float(np.mean(buf)),
                    "mean_thr_proxy": float(np.mean(thr_proxy)),
                })

    return pd.DataFrame(records)


def plot_box(df: pd.DataFrame, bw: str, col: str, title: str, out: Path):
    sub = df[df["bandwidth"] == bw].copy()

    values = []
    labels = []

    for mode in MODES:
        x = sub[sub["mode"] == mode][col].dropna().values
        if len(x):
            values.append(x)
            labels.append(label(mode))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_summary_bar(df: pd.DataFrame, bw: str, cols: list[str], title: str, out: Path):
    sub = df[df["bandwidth"] == bw].copy()

    rows = []
    for mode in MODES:
        g = sub[sub["mode"] == mode]
        if g.empty:
            continue
        row = {"mode": mode}
        for c in cols:
            row[c] = g[c].mean()
        rows.append(row)

    s = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(s))
    width = 0.8 / len(cols)

    for i, c in enumerate(cols):
        offset = (i - (len(cols) - 1) / 2) * width
        ax.bar(x + offset, s[c], width=width, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in s["mode"]])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_scatter(df: pd.DataFrame, bw: str, x_col: str, y_col: str, title: str, out: Path):
    sub = df[df["bandwidth"] == bw].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode in MODES:
        g = sub[sub["mode"] == mode]
        if g.empty:
            continue
        ax.scatter(g[x_col], g[y_col], s=70, label=label(mode), alpha=0.8)

        for _, row in g.iterrows():
            ax.annotate(
                str(int(row["rnti"])),
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 plot_correlation_diagnostics.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "plots_correlation_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_correlations(run_dir)
    if df.empty:
        raise RuntimeError("Nenhum dado encontrado.")

    csv_path = out_dir / "correlation_summary_per_ue.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] saved: {csv_path}")

    for bw in sorted(df["bandwidth"].unique()):
        plot_box(
            df,
            bw,
            "corr_mcs_thr_proxy",
            f"1 - Pearson: MCS vs throughput proxy — {bw}",
            out_dir / f"1_box_corr_mcs_throughput_proxy_{bw}.png",
        )

        plot_box(
            df,
            bw,
            "corr_mcs_alloc",
            f"2 - Pearson: MCS vs RBG allocation — {bw}",
            out_dir / f"2_box_corr_mcs_alloc_{bw}.png",
        )

        plot_box(
            df,
            bw,
            "corr_buf_alloc",
            f"3 - Pearson: backlog vs RBG allocation — {bw}",
            out_dir / f"3_box_corr_backlog_alloc_{bw}.png",
        )

        plot_box(
            df,
            bw,
            "corr_buf_thr_proxy",
            f"4 - Pearson: backlog vs throughput proxy — {bw}",
            out_dir / f"4_box_corr_backlog_throughput_proxy_{bw}.png",
        )

        plot_summary_bar(
            df,
            bw,
            ["corr_mcs_alloc", "corr_buf_alloc"],
            f"5 - Canal vs demanda na decisão de alocação — {bw}",
            out_dir / f"5_bar_corr_channel_vs_demand_alloc_{bw}.png",
        )

        plot_summary_bar(
            df,
            bw,
            ["corr_mcs_thr_proxy", "corr_buf_thr_proxy"],
            f"6 - Canal vs demanda no throughput entregue — {bw}",
            out_dir / f"6_bar_corr_channel_vs_demand_thr_proxy_{bw}.png",
        )

        plot_scatter(
            df,
            bw,
            "mean_mcs",
            "mean_thr_proxy",
            f"7 - Média MCS vs throughput proxy médio por UE — {bw}",
            out_dir / f"7_scatter_mean_mcs_vs_thr_proxy_{bw}.png",
        )

        plot_scatter(
            df,
            bw,
            "mean_buf",
            "mean_alloc",
            f"8 - Backlog médio vs alocação média por UE — {bw}",
            out_dir / f"8_scatter_mean_backlog_vs_alloc_{bw}.png",
        )

    print(f"\n[DONE] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()