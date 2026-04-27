#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


MODES = ["rr", "pf", "mr", "fmr_rl"]


def load_mode_csv(run_dir: Path, bw_dir: Path, mode: str) -> pd.DataFrame | None:
    path = bw_dir / mode / f"flow_summary_{mode}.csv"
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return None

    df = pd.read_csv(path)
    df["mode"] = mode
    df["bandwidth"] = bw_dir.name
    df["run_id"] = run_dir.name
    return df


def summarize_mode(df: pd.DataFrame) -> dict:
    thr = df["throughput_mbps"]
    delay = df["mean_delay_ms"]
    jitter = df["mean_jitter_ms"]
    loss = df["loss_ratio"]

    return {
        "run_id": df["run_id"].iloc[0],
        "bandwidth": df["bandwidth"].iloc[0],
        "mode": df["mode"].iloc[0],
        "n_flows": int(len(df)),
        "aggregate_throughput_mbps": float(thr.sum()),
        "mean_flow_throughput_mbps": float(thr.mean()),
        "min_flow_throughput_mbps": float(thr.min()),
        "p5_flow_throughput_mbps": float(thr.quantile(0.05)),
        "max_flow_throughput_mbps": float(thr.max()),
        "std_flow_throughput_mbps": float(thr.std(ddof=0)),
        "mean_delay_ms": float(delay.mean()),
        "max_delay_ms": float(delay.max()),
        "mean_jitter_ms": float(jitter.mean()),
        "mean_loss_ratio": float(loss.mean()),
        "max_loss_ratio": float(loss.max()),
    }


def build_wide(df_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (run_id, bandwidth), sub in df_summary.groupby(["run_id", "bandwidth"], sort=False):
        row = {"run_id": run_id, "bandwidth": bandwidth}

        for _, r in sub.iterrows():
            mode = r["mode"]
            for col in sub.columns:
                if col in ["run_id", "bandwidth", "mode"]:
                    continue
                row[f"{mode}_{col}"] = r[col]

        rows.append(row)

    return pd.DataFrame(rows)


def discover_bw_dirs(run_dir: Path) -> list[Path]:
    bw_dirs = [
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("bw")
    ]

    def bw_key(p: Path) -> int:
        try:
            return int(p.name.replace("bw", ""))
        except Exception:
            return 10**9

    return sorted(bw_dirs, key=bw_key)


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 compare_flow_summaries.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir não encontrada: {run_dir}")

    bw_dirs = discover_bw_dirs(run_dir)
    if not bw_dirs:
        raise RuntimeError(f"Nenhuma pasta bwXX encontrada em: {run_dir}")

    dfs = []

    for bw_dir in bw_dirs:
        for mode in MODES:
            df = load_mode_csv(run_dir, bw_dir, mode)
            if df is not None:
                dfs.append(df)

    if not dfs:
        raise RuntimeError("Nenhum flow_summary foi encontrado.")

    df_all = pd.concat(dfs, ignore_index=True)

    df_summary = pd.DataFrame(
        [
            summarize_mode(df)
            for _, df in df_all.groupby(["run_id", "bandwidth", "mode"], sort=False)
        ]
    )

    bw_order = {f"bw{x}": i for i, x in enumerate([10, 20, 50, 100])}
    mode_order = {m: i for i, m in enumerate(MODES)}

    df_summary["bw_order"] = df_summary["bandwidth"].map(bw_order).fillna(999)
    df_summary["mode_order"] = df_summary["mode"].map(mode_order).fillna(999)

    df_summary = (
        df_summary
        .sort_values(["bw_order", "mode_order", "bandwidth", "mode"])
        .drop(columns=["bw_order", "mode_order"])
        .reset_index(drop=True)
    )

    df_wide = build_wide(df_summary)

    out_long = run_dir / "flow_comparison_long.csv"
    out_wide = run_dir / "flow_comparison_wide.csv"

    df_summary.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)

    print("\n=== TABELA COMPARATIVA (LONG) ===")
    print(df_summary.to_string(index=False))

    print("\n=== TABELA COMPARATIVA (WIDE) ===")
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(df_wide.to_string(index=False))

    print(f"\n[OK] salvo em: {out_long}")
    print(f"[OK] salvo em: {out_wide}")


if __name__ == "__main__":
    main()