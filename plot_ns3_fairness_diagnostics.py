#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODES = ["rr", "pf", "mr", "fmr_rl"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "FMR-RL"}


def jain_index(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    s = x.sum()
    ss = np.sum(x * x)
    if s <= 0 or ss <= 0:
        return 0.0
    return float((s * s) / (len(x) * ss))


def discover_bw_dirs(run_dir: Path):
    return sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bw")],
        key=lambda p: int(p.name.replace("bw", "")),
    )


def infer_total_rbg(bw_dir: Path) -> float:
    fmr_path = bw_dir / "fmr_rl" / "slot_log_fmr_rl.csv"
    df = pd.read_csv(fmr_path)
    g = df.groupby(["time_s", "beam_id"])["alloc_rbg"].sum()
    return float(round(g.median()))


def get_all_rntis(bw_dir: Path) -> list[int]:
    rr_path = bw_dir / "rr" / "slot_log_rr.csv"
    df = pd.read_csv(rr_path)
    return sorted(int(x) for x in df["rnti"].unique())


def load_slot_log(path: Path, total_rbg: float) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"time_s", "beam_id", "rnti", "alloc_rbg"}
    if not required.issubset(df.columns):
        raise ValueError(f"Invalid columns in {path}")

    df = df.copy()
    slot_sum = df.groupby(["time_s", "beam_id"])["alloc_rbg"].transform("sum").astype(float)

    df["alloc_norm"] = np.where(
        slot_sum > 0,
        df["alloc_rbg"].astype(float) * total_rbg / slot_sum,
        0.0,
    )

    return df


def slot_matrix(df: pd.DataFrame, all_rntis: list[int]) -> pd.DataFrame:
    rows = []

    for key, g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = (
            g.groupby("rnti")["alloc_norm"]
            .sum()
            .reindex(all_rntis, fill_value=0.0)
        )
        alloc.name = key
        rows.append(alloc)

    if not rows:
        return pd.DataFrame(columns=all_rntis)

    return pd.DataFrame(rows).reset_index(drop=True)


def consecutive_zero_runs(values: np.ndarray) -> list[int]:
    runs = []
    current = 0

    for v in values:
        if v <= 0:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0

    if current > 0:
        runs.append(current)

    return runs


def load_all(run_dir: Path, bw_dir: Path):
    total_rbg = infer_total_rbg(bw_dir)
    all_rntis = get_all_rntis(bw_dir)

    data = {}

    for mode in MODES:
        path = bw_dir / mode / f"slot_log_{mode}.csv"
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue

        df = load_slot_log(path, total_rbg)
        mat = slot_matrix(df, all_rntis)

        data[mode] = {
            "df": df,
            "mat": mat,
            "jain": mat.apply(jain_index, axis=1),
            "active": (mat > 0).sum(axis=1),
            "zero_fraction_per_slot": (mat <= 0).mean(axis=1),
            "zero_fraction_per_ue": (mat <= 0).mean(axis=0),
        }

    return data, total_rbg, all_rntis


def plot_1_jain_timeseries(data, bw: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5))

    for mode, d in data.items():
        y = d["jain"].reset_index(drop=True)
        ax.plot(y.index, y.values, label=LABELS.get(mode, mode), linewidth=1.4)

    ax.set_xlabel("Slot index")
    ax.set_ylabel("Slot-level Jain index")
    ax.set_title(f"1 - Jain instantâneo por slot — {bw}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    out = out_dir / f"1_jain_instantaneo_por_slot_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_2_jain_boxplot(data, bw: str, out_dir: Path):
    labels = []
    values = []

    for mode in MODES:
        if mode in data:
            labels.append(LABELS.get(mode, mode))
            values.append(data[mode]["jain"].dropna().values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)

    ax.set_ylabel("Slot-level Jain index")
    ax.set_title(f"2 - Boxplot do Jain instantâneo — {bw}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    out = out_dir / f"2_boxplot_jain_instantaneo_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_3_active_ues_histogram(data, bw: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.arange(0.5, 10.5, 1.0)

    for mode, d in data.items():
        ax.hist(
            d["active"].values,
            bins=bins,
            alpha=0.45,
            label=LABELS.get(mode, mode),
            density=True,
        )

    ax.set_xlabel("Número de UEs ativos no slot")
    ax.set_ylabel("Densidade")
    ax.set_title(f"3 - Distribuição de UEs ativos por slot — {bw}")
    ax.set_xticks(range(1, 10))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    out = out_dir / f"3_histograma_ues_ativos_por_slot_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_4_heatmaps(data, bw: str, all_rntis: list[int], out_dir: Path):
    for mode, d in data.items():
        mat = d["mat"].copy()

        fig, ax = plt.subplots(figsize=(12, 4.8))

        im = ax.imshow(
            mat.T.values,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )

        ax.set_xlabel("Slot index")
        ax.set_ylabel("UE (RNTI)")
        ax.set_yticks(np.arange(len(all_rntis)))
        ax.set_yticklabels([str(x) for x in all_rntis])
        ax.set_title(f"4 - Heatmap de alocação RBG — {LABELS.get(mode, mode)} — {bw}")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalized RBG allocation")

        out = out_dir / f"4_heatmap_rbg_{mode}_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")


def plot_5_zero_fraction_per_ue(data, bw: str, all_rntis: list[int], out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(all_rntis))
    n_modes = len([m for m in MODES if m in data])
    width = 0.8 / max(1, n_modes)

    present_modes = [m for m in MODES if m in data]

    for i, mode in enumerate(present_modes):
        y = data[mode]["zero_fraction_per_ue"].reindex(all_rntis).values * 100.0
        offset = (i - (n_modes - 1) / 2) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            label=LABELS.get(mode, mode),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in all_rntis])
    ax.set_xlabel("UE (RNTI)")
    ax.set_ylabel("% de slots com alloc = 0")
    ax.set_title(f"5 - Starvation instantâneo por UE — {bw}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    out = out_dir / f"5_starvation_instantaneo_por_ue_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_6_zero_fraction_per_slot(data, bw: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5))

    for mode, d in data.items():
        y = d["zero_fraction_per_slot"].reset_index(drop=True) * 100.0
        ax.plot(y.index, y.values, linewidth=1.3, label=LABELS.get(mode, mode))

    ax.set_xlabel("Slot index")
    ax.set_ylabel("% de UEs sem alocação no slot")
    ax.set_title(f"6 - Starvation instantâneo por slot — {bw}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    out = out_dir / f"6_starvation_instantaneo_por_slot_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_7_cdf_zero_runs(data, bw: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, d in data.items():
        mat = d["mat"]
        runs = []

        for col in mat.columns:
            runs.extend(consecutive_zero_runs((mat[col].values <= 0).astype(int)))

        if not runs:
            runs = [0]

        x = np.sort(np.asarray(runs, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)

        ax.plot(x, y, linewidth=1.6, label=LABELS.get(mode, mode))

    ax.set_xlabel("Duração consecutiva sem alocação (slots)")
    ax.set_ylabel("CDF")
    ax.set_title(f"7 - CDF de períodos consecutivos sem serviço — {bw}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    out = out_dir / f"7_cdf_periodos_sem_servico_{bw}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out}")


def plot_8_summary_bars(data, bw: str, out_dir: Path):
    rows = []

    for mode, d in data.items():
        rows.append({
            "mode": LABELS.get(mode, mode),
            "mean_jain": d["jain"].mean(),
            "mean_active_ues": d["active"].mean(),
            "mean_zero_ue_percent": d["zero_fraction_per_slot"].mean() * 100.0,
        })

    df = pd.DataFrame(rows)

    metrics = [
        ("mean_jain", "Mean slot-level Jain"),
        ("mean_active_ues", "Mean active UEs per slot"),
        ("mean_zero_ue_percent", "Mean % UEs without allocation"),
    ]

    for idx, (col, title) in enumerate(metrics, start=8):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df["mode"], df[col])

        ax.set_title(f"{idx} - {title} — {bw}")
        ax.set_ylabel(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        out = out_dir / f"{idx}_summary_{col}_{bw}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved: {out}")

    out_csv = out_dir / f"8_summary_fairness_starvation_{bw}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv}")


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 plot_ns3_fairness_diagnostics.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "plots_fairness_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bw_dir in discover_bw_dirs(run_dir):
        bw = bw_dir.name
        print(f"\n[INFO] Processing {bw}")

        data, total_rbg, all_rntis = load_all(run_dir, bw_dir)

        print(f"[INFO] total_rbg={total_rbg}")
        print(f"[INFO] all_rntis={all_rntis}")

        plot_1_jain_timeseries(data, bw, out_dir)
        plot_2_jain_boxplot(data, bw, out_dir)
        plot_3_active_ues_histogram(data, bw, out_dir)
        plot_4_heatmaps(data, bw, all_rntis, out_dir)
        plot_5_zero_fraction_per_ue(data, bw, all_rntis, out_dir)
        plot_6_zero_fraction_per_slot(data, bw, out_dir)
        plot_7_cdf_zero_runs(data, bw, out_dir)
        plot_8_summary_bars(data, bw, out_dir)

    print(f"\n[DONE] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()