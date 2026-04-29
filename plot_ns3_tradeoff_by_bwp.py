#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


MODES = ["rr", "pf", "mr", "fmr_rl"]


def jain_index(values):
    x = pd.Series(values, dtype="float64")
    s = x.sum()
    ss = (x ** 2).sum()
    n = len(x)
    if n == 0 or ss == 0:
        return 0.0
    return float((s * s) / (n * ss))

def load_jain_rbg_from_slot_log(path: Path, all_rntis: list[int]) -> float:
    if not path.exists():
        print(f"[WARN] missing slot log: {path}")
        return float("nan")

    df = pd.read_csv(path)

    required = {"time_s", "beam_id", "rnti", "alloc_rbg"}
    if not required.issubset(df.columns):
        print(f"[WARN] invalid slot log columns: {path}")
        return float("nan")

    jains = []

    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = (
            g.groupby("rnti")["alloc_rbg"]
            .sum()
            .reindex(all_rntis, fill_value=0)
            .to_numpy(dtype=float)
        )

        jains.append(jain_index(alloc))

    return float(pd.Series(jains).mean()) if jains else float("nan")

def discover_bw_dirs(run_dir: Path):
    out = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bw")]
    return sorted(out, key=lambda p: int(p.name.replace("bw", "")))


def jain(x):
    x = np.array(x)
    if np.sum(x) == 0:
        return 0.0
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))


def load_points(run_dir):
    run_dir = Path(run_dir)

    rows = []
    for bw_dir in sorted(run_dir.glob("bw*")):
        bandwidth = bw_dir.name
        rr_slot_path = bw_dir / "rr" / "slot_log_rr.csv"
        df_rr_slot = pd.read_csv(rr_slot_path)
        all_rntis = sorted(df_rr_slot["rnti"].unique())
        print(f"[INFO] {bw_dir.name} all_rntis = {all_rntis}")

    for mode_dir in bw_dir.iterdir():
        mode = mode_dir.name

        slot_path = mode_dir / f"slot_log_{mode}.csv"
        slot_path = mode_dir / f"slot_log_{mode}.csv"

        jain_rbg = load_jain_rbg_from_slot_log(slot_path, all_rntis)

        if pd.isna(jain_rbg):
            continue
        flow_path = mode_dir / f"flow_summary_{mode}.csv"

        if not flow_path.exists():
            print(f"[WARN] missing flow summary: {flow_path}")
            continue

        df_flow = pd.read_csv(flow_path)
        thr = df_flow["throughput_mbps"]

        rows.append({
            "bandwidth": bandwidth,
            "mode": mode,
            "aggregate_throughput_mbps": thr.sum(),
            "mean_flow_throughput_mbps": thr.mean(),
            "min_flow_throughput_mbps": thr.min(),
            "p5_flow_throughput_mbps": thr.quantile(0.05),
            "jain_rbg": jain_rbg,
        })

    return pd.DataFrame(rows)

def plot_tradeoff(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "rr": "RR",
        "pf": "PF",
        "mr": "MR",
        "fmr_rl": "FMR-RL",
    }

    markers = {
        "rr": "o",
        "pf": "s",
        "mr": "^",
        "fmr_rl": "D",
    }

    for bw, sub in df.groupby("bandwidth"):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        for _, row in sub.iterrows():
            mode = row["mode"]
            ax.scatter(
                row["jain_rbg"],
                row["aggregate_throughput_mbps"],
                marker=markers.get(mode, "o"),
                s=120,
                label=labels.get(mode, mode),
            )

            ax.annotate(
                labels.get(mode, mode),
                (
                    row["jain_rbg"],
                    row["aggregate_throughput_mbps"],
                ),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=11,
            )

        ax.set_xlabel("Jain index over RBG Alloc", fontsize=13)
        ax.set_ylabel("Aggregate throughput (Mbps)", fontsize=13)
        ax.set_title(f"Throughput-fairness trade-off — {bw}", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=11)

        ax.legend(fontsize=11, frameon=True)
        plt.tight_layout()

        out_png = out_dir / f"tradeoff_{bw}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[OK] saved: {out_png}")

    summary_csv = out_dir / "tradeoff_points_by_bwp.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[OK] saved: {summary_csv}")


def plot_mean_rbg_per_ue(run_dir, plots_dir):
    run_dir = Path(run_dir)

    for bw_dir in sorted(run_dir.glob("bw*")):
        bandwidth = bw_dir.name

        plt.figure(figsize=(10, 5))

        for mode_dir in sorted(bw_dir.iterdir()):
            mode = mode_dir.name

            slot_path = mode_dir / f"slot_log_{mode}.csv"

            if not slot_path.exists():
                print(f"[WARN] missing slot log: {slot_path}")
                continue

            df = pd.read_csv(slot_path)

            # média de RBG por UE
            ue_mean = (
                df.groupby("rnti")["alloc_rbg"]
                .mean()
                .sort_index()
            )

            plt.plot(
                ue_mean.index,
                ue_mean.values,
                marker="o",
                label=mode
            )

        plt.xlabel("UE (RNTI)")
        plt.ylabel("Mean allocated RBG")
        plt.title(f"Mean RBG allocation per UE ({bandwidth})")
        plt.grid(True)
        plt.legend()

        out = plots_dir / f"mean_rbg_per_ue_{bandwidth}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] saved: {out}")

def plot_grouped_mean_rbg_per_ue(run_dir, plots_dir):
    run_dir = Path(run_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    mode_order = ["rr", "pf", "mr", "fmr_rl"]

    for bw_dir in sorted(run_dir.glob("bw*")):
        bandwidth = bw_dir.name
        records = []

        for mode in mode_order:
            slot_path = bw_dir / mode / f"slot_log_{mode}.csv"

            if not slot_path.exists():
                print(f"[WARN] missing slot log: {slot_path}")
                continue

            df = pd.read_csv(slot_path)

            if "rnti" not in df.columns or "alloc_rbg" not in df.columns:
                print(f"[WARN] invalid slot log columns: {slot_path}")
                continue

            ue_mean = (
                df.groupby("rnti")["alloc_rbg"]
                .mean()
                .reset_index()
                .rename(columns={"alloc_rbg": "mean_alloc_rbg"})
            )

            ue_mean["mode"] = mode
            records.append(ue_mean)

        if not records:
            continue

        df_all = pd.concat(records, ignore_index=True)

        pivot = (
            df_all.pivot(index="rnti", columns="mode", values="mean_alloc_rbg")
            .reindex(columns=[m for m in mode_order if m in df_all["mode"].unique()])
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
                label=mode.upper() if mode != "fmr_rl" else "FMR-RL",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(rnti) for rnti in pivot.index])
        ax.set_xlabel("UE (RNTI)")
        ax.set_ylabel("Mean allocated RBGs")
        ax.set_title(f"Mean RBG allocation per UE — {bandwidth}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True)

        plt.tight_layout()

        out_png = plots_dir / f"grouped_mean_rbg_per_ue_{bandwidth}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        out_csv = plots_dir / f"grouped_mean_rbg_per_ue_{bandwidth}.csv"
        pivot.to_csv(out_csv)

        print(f"[OK] saved: {out_png}")
        print(f"[OK] saved: {out_csv}")


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 plot_ns3_tradeoff_by_bwp.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    df = load_points(run_dir)
    if df.empty:
        raise RuntimeError("Nenhum flow_summary encontrado.")

    plots_dir = run_dir / "plots"
    plot_tradeoff(df, plots_dir)
    plot_mean_rbg_per_ue(run_dir, plots_dir)
    plot_grouped_mean_rbg_per_ue(run_dir, plots_dir)

    print("\n=== POINTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
