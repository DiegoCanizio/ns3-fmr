#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = {
    "rr": "RR",
    "pf": "PF",
    "mr": "MR",
    "fmr_rl": "FMR-RL",
}

MODES = ["rr", "pf", "mr", "fmr_rl"]


def label_mode(m):
    return LABELS.get(m, m)


def plot_bar(df, x_col, y_col, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [label_mode(m) for m in df[x_col]]
    ax.bar(labels, df[y_col].values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def plot_grouped_per_ue(df, value_col, title, ylabel, out_path):
    pivot = (
        df.pivot(index="rnti", columns="mode", values=value_col)
        .reindex(columns=[m for m in MODES if m in df["mode"].unique()])
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(pivot.index))
    n_modes = len(pivot.columns)
    width = 0.8 / max(1, n_modes)

    for i, mode in enumerate(pivot.columns):
        offset = (i - (n_modes - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[mode].values,
            width=width,
            label=label_mode(mode),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in pivot.index])
    ax.set_xlabel("UE (RNTI)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 plot_backlog_debug.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    debug_dir = run_dir / "backlog_debug"

    summary_path = debug_dir / "backlog_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)
    out_dir = run_dir / "plots_backlog_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw, sub in df.groupby("bandwidth"):
        sub = sub.copy()
        sub["mode_order"] = sub["mode"].map({m: i for i, m in enumerate(MODES)})
        sub = sub.sort_values("mode_order")

        plot_bar(
            sub,
            "mode",
            "mean_buf_req",
            f"1 - Backlog médio por scheduler — {bw}",
            "Mean buf_req",
            out_dir / f"1_backlog_medio_scheduler_{bw}.png",
        )

        plot_bar(
            sub,
            "mode",
            "sum_buf_req_mean_per_slot",
            f"2 - Backlog total médio por slot — {bw}",
            "Mean total buf_req per slot",
            out_dir / f"2_backlog_total_medio_slot_{bw}.png",
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [label_mode(m) for m in sub["mode"]]
        x = np.arange(len(labels))
        width = 0.36

        ax.bar(
            x - width / 2,
            sub["mean_backlogged_ues_per_slot"],
            width,
            label="UEs com buffer",
        )
        ax.bar(
            x + width / 2,
            sub["mean_served_ues_per_slot"],
            width,
            label="UEs atendidos",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean UEs per slot")
        ax.set_title(f"3 - UEs com buffer vs UEs atendidos — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()

        out = out_dir / f"3_backlogged_vs_served_ues_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        width = 0.36

        ax.bar(
            x - width / 2,
            sub["mean_jain_buf_req_per_slot"],
            width,
            label="Jain do buffer",
        )
        ax.bar(
            x + width / 2,
            sub["mean_jain_alloc_per_slot"],
            width,
            label="Jain da alocação",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean slot-level Jain")
        ax.set_title(f"4 - Jain do buffer vs Jain da alocação — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()

        out = out_dir / f"4_jain_buffer_vs_alocacao_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")

        plot_bar(
            sub,
            "mode",
            "pct_buf_gt0_alloc_eq0",
            f"5 - Backlog sem atendimento — {bw}",
            "% casos com buf_req > 0 e alloc = 0",
            out_dir / f"5_backlog_sem_atendimento_{bw}.png",
        )

        plot_bar(
            sub,
            "mode",
            "pct_buf_eq0_alloc_eq0",
            f"6 - Slots ociosos sem alocação — {bw}",
            "% casos com buf_req = 0 e alloc = 0",
            out_dir / f"6_ocioso_sem_alocacao_{bw}.png",
        )

        per_ue_path = debug_dir / f"backlog_per_ue_{bw}.csv"
        if per_ue_path.exists():
            df_ue = pd.read_csv(per_ue_path)

            plot_grouped_per_ue(
                df_ue,
                "mean_buf_req",
                f"7 - Backlog médio por UE — {bw}",
                "Mean buf_req",
                out_dir / f"7_backlog_medio_por_ue_{bw}.png",
            )

            plot_grouped_per_ue(
                df_ue,
                "mean_alloc_rbg",
                f"8 - Alocação média por UE — {bw}",
                "Mean alloc_rbg",
                out_dir / f"8_alocacao_media_por_ue_{bw}.png",
            )

            plot_grouped_per_ue(
                df_ue,
                "pct_slots_need_service",
                f"9 - Percentual de slots com demanda por UE — {bw}",
                "% slots com buf_req > 0",
                out_dir / f"9_slots_com_demanda_por_ue_{bw}.png",
            )

            plot_grouped_per_ue(
                df_ue,
                "pct_slots_served",
                f"10 - Percentual de slots atendidos por UE — {bw}",
                "% slots com alloc > 0",
                out_dir / f"10_slots_atendidos_por_ue_{bw}.png",
            )

            plot_grouped_per_ue(
                df_ue,
                "pct_backlog_not_served",
                f"11 - Demanda sem atendimento por UE — {bw}",
                "% slots com buf_req > 0 e alloc = 0",
                out_dir / f"11_demanda_sem_atendimento_por_ue_{bw}.png",
            )

    print(f"\n[DONE] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()