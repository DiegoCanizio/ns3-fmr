#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["rr", "pf", "mr", "fmr_rl"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "FMR-RL"}


def label(m):
    return LABELS.get(m, m)


def order_modes(df):
    df = df.copy()
    df["mode_order"] = df["mode"].map({m: i for i, m in enumerate(MODES)})
    return df.sort_values("mode_order").drop(columns="mode_order")


def bar_scheduler(df, col, title, ylabel, out):
    df = order_modes(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([label(m) for m in df["mode"]], df[col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def grouped_per_ue(df, col, title, ylabel, out):
    pivot = (
        df.pivot(index="rnti", columns="mode", values=col)
        .reindex(columns=[m for m in MODES if m in df["mode"].unique()])
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))

    for i, mode in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[mode].values, width=width, label=label(mode))

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("UE (RNTI)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 plot_backlog_debug.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    debug_dir = run_dir / "backlog_debug"
    out_dir = run_dir / "plots_backlog_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(debug_dir / "backlog_summary.csv")

    for bw, sub in summary.groupby("bandwidth"):
        sub = order_modes(sub)

        bar_scheduler(
            sub,
            "pct_buf_gt0_alloc_eq0",
            f"1 - Backlog sem atendimento — {bw}",
            "% casos com buf_req > 0 e alloc = 0",
            out_dir / f"1_backlog_sem_atendimento_{bw}.png",
        )

        bar_scheduler(
            sub,
            "mean_served_ues_per_slot",
            f"2 - UEs atendidos por slot — {bw}",
            "Média de UEs com alloc > 0",
            out_dir / f"2_ues_atendidos_por_slot_{bw}.png",
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width/2, sub["mean_backlogged_ues_per_slot"], width, label="UEs com backlog")
        ax.bar(x + width/2, sub["mean_served_ues_per_slot"], width, label="UEs atendidos")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Média por slot")
        ax.set_title(f"3 - UEs com backlog vs UEs atendidos — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        out = out_dir / f"3_backlog_vs_atendidos_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, sub["mean_jain_buf_req_per_slot"], width, label="Jain do buffer")
        ax.bar(x + width/2, sub["mean_jain_alloc_per_slot"], width, label="Jain da alocação")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Jain médio por slot")
        ax.set_title(f"4 - Jain do backlog vs Jain da alocação — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        out = out_dir / f"4_jain_buffer_vs_alocacao_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")

        bar_scheduler(
            sub,
            "sum_buf_req_mean_per_slot",
            f"5 - Backlog total médio por slot — {bw}",
            "Soma média de buf_req por slot",
            out_dir / f"5_backlog_total_medio_slot_{bw}.png",
        )

        per_ue_path = debug_dir / f"backlog_per_ue_{bw}.csv"
        if per_ue_path.exists():
            ue = pd.read_csv(per_ue_path)

            grouped_per_ue(
                ue,
                "pct_backlog_not_served",
                f"6 - Backlog sem atendimento por UE — {bw}",
                "% slots com buf_req > 0 e alloc = 0",
                out_dir / f"6_backlog_sem_atendimento_por_ue_{bw}.png",
            )

            grouped_per_ue(
                ue,
                "pct_slots_served",
                f"7 - Percentual de slots atendidos por UE — {bw}",
                "% slots com alloc > 0",
                out_dir / f"7_slots_atendidos_por_ue_{bw}.png",
            )

            grouped_per_ue(
                ue,
                "mean_buf_req",
                f"8 - Backlog médio por UE — {bw}",
                "Mean buf_req",
                out_dir / f"8_backlog_medio_por_ue_{bw}.png",
            )

            grouped_per_ue(
                ue,
                "mean_alloc_rbg",
                f"9 - Alocação média por UE — {bw}",
                "Mean alloc_rbg",
                out_dir / f"9_alocacao_media_por_ue_{bw}.png",
            )

    print(f"\n[DONE] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()