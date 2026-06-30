from pathlib import Path
import os
import math
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = Path(os.environ.get(
    "RUN_DIR",
    "compare_runs/expA_fr1_100seeds_30s_baselines_fixedmob"
))

SCENARIO = os.environ.get("SCENARIO", "dynamic_continuous_30s")
OUT_DIR = Path(os.environ.get("OUT_DIR", str(RUN_DIR / "preview_shaded_partial")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = os.environ.get("MODES", "rr,pf,mr").split(",")
BWS = [int(x) for x in os.environ.get(
    "BWS",
    "10,15,20,25,30,35,40,45,50,60,70,80,90,100"
).split(",")]

STABLE_SECONDS = int(os.environ.get("STABLE_SECONDS", "30"))
CONF_LEVEL = 0.99


def ci_half_width(values, level=0.99):
    vals = np.array([v for v in values if pd.notna(v)], dtype=float)
    n = len(vals)

    if n <= 1:
        return np.nan

    sd = vals.std(ddof=1)

    try:
        from scipy.stats import t
        crit = t.ppf((1 + level) / 2, n - 1)
    except Exception:
        crit = 2.576

    return crit * sd / math.sqrt(n)


def is_complete_and_stable(mode_dir: Path, mode: str) -> bool:
    files = [
        mode_dir / f"flow_summary_{mode}.csv",
        mode_dir / f"slot_log_{mode}.csv",
    ]

    if not all(p.exists() and p.stat().st_size > 0 for p in files):
        return False

    newest = max(p.stat().st_mtime for p in files)
    return (time.time() - newest) >= STABLE_SECONDS


def read_metric_from_flow(path: Path):
    df = pd.read_csv(path)

    required = {"throughput_mbps", "mean_delay_ms", "tx_packets", "rx_packets"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Colunas ausentes em {path}: {missing}")

    tx = pd.to_numeric(df["tx_packets"], errors="coerce").sum()
    rx = pd.to_numeric(df["rx_packets"], errors="coerce").sum()

    loss_ratio = np.nan
    if tx > 0:
        loss_ratio = 1.0 - (rx / tx)

    return {
        "aggregate_throughput_mbps": pd.to_numeric(df["throughput_mbps"], errors="coerce").sum(),
        "mean_flow_throughput_mbps": pd.to_numeric(df["throughput_mbps"], errors="coerce").mean(),
        "mean_delay_ms": pd.to_numeric(df["mean_delay_ms"], errors="coerce").mean(),
        "loss_ratio": loss_ratio,
        "n_flows": len(df),
        "tx_packets": tx,
        "rx_packets": rx,
    }


rows = []

for seed_dir in sorted(RUN_DIR.glob("seed_*")):
    if not seed_dir.is_dir():
        continue

    seed = seed_dir.name

    for bw in BWS:
        for mode in MODES:
            mode_dir = seed_dir / SCENARIO / f"bw{bw}" / mode

            if not is_complete_and_stable(mode_dir, mode):
                continue

            flow_path = mode_dir / f"flow_summary_{mode}.csv"

            try:
                metrics = read_metric_from_flow(flow_path)
            except Exception as e:
                print(f"[WARN] Ignorando {flow_path}: {e}")
                continue

            rows.append({
                "seed": seed,
                "bandwidth_mhz": bw,
                "mode": mode,
                **metrics,
            })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("Nenhuma simulação completa encontrada.")

df.to_csv(OUT_DIR / "partial_per_seed_metrics.csv", index=False)

summary_rows = []

metrics_to_summarize = [
    "aggregate_throughput_mbps",
    "mean_flow_throughput_mbps",
    "mean_delay_ms",
    "loss_ratio",
]

for (mode, bw), sub in df.groupby(["mode", "bandwidth_mhz"], sort=True):
    row = {
        "mode": mode,
        "bandwidth_mhz": bw,
        "n_seeds": sub["seed"].nunique(),
        "n_rows": len(sub),
    }

    for metric in metrics_to_summarize:
        vals = pd.to_numeric(sub[metric], errors="coerce")
        mean = vals.mean()
        half = ci_half_width(vals, CONF_LEVEL)

        row[f"{metric}_mean"] = mean
        row[f"{metric}_ic99_half"] = half
        row[f"{metric}_ic99_low"] = mean - half if pd.notna(half) else np.nan
        row[f"{metric}_ic99_high"] = mean + half if pd.notna(half) else np.nan

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "partial_summary_ic99.csv", index=False)


def plot_shaded(metric, ylabel, filename):
    plt.figure(figsize=(10, 5.5))

    for mode in MODES:
        sub = summary[summary["mode"] == mode].sort_values("bandwidth_mhz")

        if sub.empty:
            continue

        x = sub["bandwidth_mhz"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy()
        lo = sub[f"{metric}_ic99_low"].to_numpy()
        hi = sub[f"{metric}_ic99_high"].to_numpy()

        line, = plt.plot(x, y, marker="o", label=mode.upper())
        color = line.get_color()

        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            plt.fill_between(x[ok], lo[ok], hi[ok], alpha=0.18, color=color)

        for bx, by, n in zip(x, y, sub["n_seeds"]):
            plt.annotate(str(int(n)), (bx, by), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    plt.xlabel("Largura de banda (MHz)")
    plt.ylabel(ylabel)
    plt.title(f"Prévia parcial dos baselines com IC99 sombreado\nRun: {RUN_DIR.name}")
    plt.xticks(BWS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


plot_shaded(
    "aggregate_throughput_mbps",
    "Vazão agregada (Mbps)",
    "preview_shaded_aggregate_throughput_ic99.png"
)

plot_shaded(
    "mean_flow_throughput_mbps",
    "Vazão média por fluxo (Mbps)",
    "preview_shaded_mean_flow_throughput_ic99.png"
)

plot_shaded(
    "mean_delay_ms",
    "Delay médio por fluxo (ms)",
    "preview_shaded_delay_ic99.png"
)

plot_shaded(
    "loss_ratio",
    "Taxa de perda",
    "preview_shaded_loss_ratio_ic99.png"
)

print("[OK] Prévia gerada em:", OUT_DIR)
print()
print("Resumo de seeds por modo e banda:")
pivot = summary.pivot_table(
    index="bandwidth_mhz",
    columns="mode",
    values="n_seeds",
    aggfunc="max",
    fill_value=0,
)
print(pivot.to_string())
