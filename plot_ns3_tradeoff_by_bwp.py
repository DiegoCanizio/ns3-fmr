#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODES = ["rr", "pf", "mr", "fmr_rl"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "FMR-RL"}
MARKERS = {"rr": "o", "pf": "s", "mr": "^", "fmr_rl": "D"}


def jain_index(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    ss = float((x ** 2).sum())
    if s <= 0.0 or ss <= 0.0:
        return 0.0
    return float((s * s) / (len(x) * ss))


def discover_bw_dirs(run_dir: Path):
    out = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bw")]
    return sorted(out, key=lambda p: int(p.name.replace("bw", "")))


def infer_total_rbg_from_fmr(bw_dir: Path) -> float:
    fmr_path = bw_dir / "fmr_rl" / "slot_log_fmr_rl.csv"
    if not fmr_path.exists():
        raise FileNotFoundError(f"Missing FMR slot log to infer total RBG: {fmr_path}")

    df = pd.read_csv(fmr_path)
    g = df.groupby(["time_s", "beam_id"])["alloc_rbg"].sum()

    if g.empty:
        raise RuntimeError(f"Empty FMR slot log: {fmr_path}")

    total = float(round(g.median()))
    print(f"[INFO] {bw_dir.name} inferred total_rbg = {total}")
    return total


def get_all_rntis_from_rr(bw_dir: Path) -> list[int]:
    rr_path = bw_dir / "rr" / "slot_log_rr.csv"
    if not rr_path.exists():
        raise FileNotFoundError(f"Missing RR slot log used as RNTI reference: {rr_path}")

    df = pd.read_csv(rr_path)
    if "rnti" not in df.columns:
        raise ValueError(f"RR slot log has no rnti column: {rr_path}")

    return sorted(int(x) for x in df["rnti"].unique())


def load_slot_log_normalized(path: Path, total_rbg: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"time_s", "beam_id", "rnti", "alloc_rbg"}
    if not required.issubset(df.columns):
        raise ValueError(f"Invalid slot log columns: {path}")

    df = df.copy()
    slot_sum = df.groupby(["time_s", "beam_id"])["alloc_rbg"].transform("sum").astype(float)

    df["alloc_rbg_norm"] = np.where(
        slot_sum > 0.0,
        df["alloc_rbg"].astype(float) * float(total_rbg) / slot_sum,
        0.0,
    )

    return df


def load_jain_rbg_from_slot_log(path: Path, all_rntis: list[int], total_rbg: float) -> float:
    if not path.exists():
        print(f"[WARN] missing slot log: {path}")
        return float("nan")

    try:
        df = load_slot_log_normalized(path, total_rbg)
    except Exception as e:
        print(f"[WARN] invalid slot log {path}: {e}")
        return float("nan")

    jains = []

    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = (
            g.groupby("rnti")["alloc_rbg_norm"]
            .sum()
            .reindex(all_rntis, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        jains.append(jain_index(alloc))

    return float(np.mean(jains)) if jains else float("nan")


def load_mean_rbg_per_ue(slot_path: Path, all_rntis: list[int], total_rbg: float) -> pd.Series:
    df = load_slot_log_normalized(slot_path, total_rbg)

    rows = []

    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = (
            g.groupby("rnti")["alloc_rbg_norm"]
            .sum()
            .reindex(all_rntis, fill_value=0.0)
        )
        rows.append(alloc)

    if not rows:
        return pd.Series(0.0, index=all_rntis)

    mat = pd.DataFrame(rows)
    return mat.mean(axis=0).reindex(all_rntis, fill_value=0.0)


def validate_slot_sums(run_dir: Path):
    print("\n=== SLOT SUM VALIDATION AFTER NORMALIZATION ===")

    for bw_dir in discover_bw_dirs(run_dir):
        total_rbg = infer_total_rbg_from_fmr(bw_dir)

        for mode in MODES:
            slot_path = bw_dir / mode / f"slot_log_{mode}.csv"
            if not slot_path.exists():
                continue

            df = load_slot_log_normalized(slot_path, total_rbg)
            g_raw = df.groupby(["time_s", "beam_id"])["alloc_rbg"].sum()
            g_norm = df.groupby(["time_s", "beam_id"])["alloc_rbg_norm"].sum()

            print(
                f"{bw_dir.name:>5} {mode:>6} | "
                f"raw mean={g_raw.mean():8.3f} raw max={g_raw.max():8.3f} | "
                f"norm mean={g_norm.mean():6.3f} norm min={g_norm.min():6.3f} norm max={g_norm.max():6.3f}"
            )


def load_points(run_dir: Path) -> pd.DataFrame:
    rows = []

    for bw_dir in discover_bw_dirs(run_dir):
        bandwidth = bw_dir.name
        total_rbg = infer_total_rbg_from_fmr(bw_dir)
        all_rntis = get_all_rntis_from_rr(bw_dir)

        print(f"[INFO] {bandwidth} all_rntis = {all_rntis}")

        for mode in MODES:
            mode_dir = bw_dir / mode
            slot_path = mode_dir / f"slot_log_{mode}.csv"
            flow_path = mode_dir / f"flow_summary_{mode}.csv"

            jain_rbg = load_jain_rbg_from_slot_log(slot_path, all_rntis, total_rbg)
            if pd.isna(jain_rbg):
                continue

            if not flow_path.exists():
                print(f"[WARN] missing flow summary: {flow_path}")
                continue

            df_flow = pd.read_csv(flow_path)
            if "throughput_mbps" not in df_flow.columns:
                print(f"[WARN] invalid flow summary: {flow_path}")
                continue

            thr = df_flow["throughput_mbps"]

            rows.append({
                "bandwidth": bandwidth,
                "mode": mode,
                "aggregate_throughput_mbps": float(thr.sum()),
                "mean_flow_throughput_mbps": float(thr.mean()),
                "min_flow_throughput_mbps": float(thr.min()),
                "p5_flow_throughput_mbps": float(thr.quantile(0.05)),
                "jain_rbg": float(jain_rbg),
            })

    return pd.DataFrame(rows)


def plot_tradeoff(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw, sub in df.groupby("bandwidth"):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))

        for _, row in sub.iterrows():
            mode = row["mode"]

            ax.scatter(
                row["jain_rbg"],
                row["aggregate_throughput_mbps"],
                marker=MARKERS.get(mode, "o"),
                s=120,
                label=LABELS.get(mode, mode),
            )

            ax.annotate(
                LABELS.get(mode, mode),
                (row["jain_rbg"], row["aggregate_throughput_mbps"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=11,
            )

        ax.set_xlabel("Mean slot-level Jain index over normalized RBG allocation", fontsize=13)
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


def plot_grouped_mean_rbg_per_ue(run_dir: Path, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    for bw_dir in discover_bw_dirs(run_dir):
        bandwidth = bw_dir.name
        total_rbg = infer_total_rbg_from_fmr(bw_dir)
        all_rntis = get_all_rntis_from_rr(bw_dir)

        data = {}

        for mode in MODES:
            slot_path = bw_dir / mode / f"slot_log_{mode}.csv"

            if not slot_path.exists():
                print(f"[WARN] missing slot log: {slot_path}")
                continue

            data[mode] = load_mean_rbg_per_ue(slot_path, all_rntis, total_rbg)

        if not data:
            continue

        pivot = pd.DataFrame(data).reindex(index=all_rntis)

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
                label=LABELS.get(mode, mode),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(rnti) for rnti in pivot.index])
        ax.set_xlabel("UE (RNTI)")
        ax.set_ylabel("Mean normalized RBGs per slot")
        ax.set_title(f"Mean normalized RBG allocation per UE — {bandwidth}")
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

    validate_slot_sums(run_dir)

    df = load_points(run_dir)
    if df.empty:
        raise RuntimeError("Nenhum dado encontrado.")

    plots_dir = run_dir / "plots"
    plot_tradeoff(df, plots_dir)
    plot_grouped_mean_rbg_per_ue(run_dir, plots_dir)

    print("\n=== POINTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()