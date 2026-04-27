#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


MODES = ["rr", "pf", "mr", "fmr_rl"]


def jain_index(values):
    x = pd.Series(values, dtype="float64")
    s = x.sum()
    ss = (x ** 2).sum()
    n = len(x)
    if n == 0 or ss == 0:
        return 0.0
    return float((s * s) / (n * ss))

def load_jain_rbg_from_slot_log(path: Path) -> float:
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
        alloc = g["alloc_rbg"].to_numpy(dtype=float)

        # ignora linhas sem alocação útil
        if alloc.size == 0:
            continue

        jains.append(jain_index(alloc))

    if not jains:
        return float("nan")

    return float(pd.Series(jains).mean())

def discover_bw_dirs(run_dir: Path):
    out = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bw")]
    return sorted(out, key=lambda p: int(p.name.replace("bw", "")))


def load_points(run_dir: Path) -> pd.DataFrame:
    rows = []

    for bw_dir in discover_bw_dirs(run_dir):
        for mode in MODES:
            slot_path = bw_dir / mode / f"slot_log_{mode}.csv"
            jain_rbg = load_jain_rbg_from_slot_log(slot_path)
            if not slot_path.exists():
                print(f"[WARN] missing: {path}")
                continue

            df = pd.read_csv(path)
            thr = df["throughput_mbps"]

            slot_path = bw_dir / mode / f"slot_log_{mode}.csv"
            jain_rbg = load_jain_rbg_from_slot_log(slot_path)

            rows.append({
                "bandwidth": bw_dir.name,
                "mode": mode,
                "aggregate_throughput_mbps": float(thr.sum()),
                "mean_flow_throughput_mbps": float(thr.mean()),
                "min_flow_throughput_mbps": float(thr.min()),
                "p5_flow_throughput_mbps": float(thr.quantile(0.05)),
                "jain_rbg": jain_rbg,
                "mean_delay_ms": float(df["mean_delay_ms"].mean()),
                "mean_loss_ratio": float(df["loss_ratio"].mean()),
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
                row["jain_RBG"],
                row["aggregate_throughput_mbps"],
                marker=markers.get(mode, "o"),
                s=120,
                label=labels.get(mode, mode),
            )

            ax.annotate(
                labels.get(mode, mode),
                (
                    row["jain_RBG"],
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

    print("\n=== POINTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()