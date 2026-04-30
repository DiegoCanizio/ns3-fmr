#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ["rr", "pf", "mr", "fmr_rl"]


def jain_index(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    ss = float(np.sum(x * x))
    if s <= 0.0 or ss <= 0.0:
        return 0.0
    return float((s * s) / (len(x) * ss))


def load_slot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time_s", "beam_id", "rnti", "buf_req", "alloc_rbg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def summarize_mode(df: pd.DataFrame, all_rntis: list[int]) -> dict:
    rows = []

    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        slot = (
            g.groupby("rnti")
            .agg(buf_req=("buf_req", "max"), alloc_rbg=("alloc_rbg", "sum"))
            .reindex(all_rntis, fill_value=0)
        )
        rows.append(slot)

    if not rows:
        return {}

    full = pd.concat(rows, keys=range(len(rows)), names=["slot_idx", "rnti"]).reset_index()

    need_service = full["buf_req"] > 0
    served = full["alloc_rbg"] > 0

    backlog_not_served = need_service & (~served)
    idle_not_served = (~need_service) & (~served)

    buf_by_slot = full.groupby("slot_idx")["buf_req"].sum()
    active_backlog_by_slot = full.groupby("slot_idx")["buf_req"].apply(lambda x: (x > 0).sum())
    served_by_slot = full.groupby("slot_idx")["alloc_rbg"].apply(lambda x: (x > 0).sum())

    jain_buf_by_slot = full.groupby("slot_idx")["buf_req"].apply(jain_index)
    jain_alloc_by_slot = full.groupby("slot_idx")["alloc_rbg"].apply(jain_index)

    return {
        "n_slots": int(full["slot_idx"].nunique()),
        "n_rntis": int(len(all_rntis)),
        "mean_buf_req": float(full["buf_req"].mean()),
        "median_buf_req": float(full["buf_req"].median()),
        "max_buf_req": float(full["buf_req"].max()),
        "sum_buf_req_mean_per_slot": float(buf_by_slot.mean()),
        "sum_buf_req_max_per_slot": float(buf_by_slot.max()),
        "mean_backlogged_ues_per_slot": float(active_backlog_by_slot.mean()),
        "mean_served_ues_per_slot": float(served_by_slot.mean()),
        "pct_buf_gt0_alloc_eq0": float(backlog_not_served.mean() * 100.0),
        "pct_buf_eq0_alloc_eq0": float(idle_not_served.mean() * 100.0),
        "pct_buf_gt0_alloc_gt0": float((need_service & served).mean() * 100.0),
        "mean_jain_buf_req_per_slot": float(jain_buf_by_slot.mean()),
        "mean_jain_alloc_per_slot": float(jain_alloc_by_slot.mean()),
    }


def summarize_per_ue(df: pd.DataFrame, all_rntis: list[int]) -> pd.DataFrame:
    rows = []

    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        slot = (
            g.groupby("rnti")
            .agg(buf_req=("buf_req", "max"), alloc_rbg=("alloc_rbg", "sum"))
            .reindex(all_rntis, fill_value=0)
        )
        rows.append(slot)

    full = pd.concat(rows, keys=range(len(rows)), names=["slot_idx", "rnti"]).reset_index()

    full["need_service"] = full["buf_req"] > 0
    full["served"] = full["alloc_rbg"] > 0
    full["backlog_not_served"] = full["need_service"] & (~full["served"])

    out = (
        full.groupby("rnti")
        .agg(
            mean_buf_req=("buf_req", "mean"),
            max_buf_req=("buf_req", "max"),
            mean_alloc_rbg=("alloc_rbg", "mean"),
            pct_slots_need_service=("need_service", lambda x: x.mean() * 100.0),
            pct_slots_served=("served", lambda x: x.mean() * 100.0),
            pct_backlog_not_served=("backlog_not_served", lambda x: x.mean() * 100.0),
        )
        .reset_index()
    )

    return out


def main():
    if len(sys.argv) != 2:
        print("Uso:")
        print("  python3 debug_backlog_stats.py ~/ns3-fmr/compare_runs/<run_id>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "backlog_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []

    for bw_dir in sorted(run_dir.glob("bw*")):
        bw = bw_dir.name

        rr_path = bw_dir / "rr" / "slot_log_rr.csv"
        if not rr_path.exists():
            print(f"[WARN] missing RR reference: {rr_path}")
            continue

        rr_df = load_slot(rr_path)
        all_rntis = sorted(int(x) for x in rr_df["rnti"].unique())

        print(f"\n[INFO] {bw} all_rntis={all_rntis}")

        per_ue_tables = []

        for mode in MODES:
            path = bw_dir / mode / f"slot_log_{mode}.csv"
            if not path.exists():
                print(f"[WARN] missing {mode}: {path}")
                continue

            df = load_slot(path)

            summary = summarize_mode(df, all_rntis)
            summary["bandwidth"] = bw
            summary["mode"] = mode
            all_summary.append(summary)

            per_ue = summarize_per_ue(df, all_rntis)
            per_ue.insert(0, "mode", mode)
            per_ue.insert(0, "bandwidth", bw)
            per_ue_tables.append(per_ue)

        if per_ue_tables:
            df_per_ue = pd.concat(per_ue_tables, ignore_index=True)
            out_per_ue = out_dir / f"backlog_per_ue_{bw}.csv"
            df_per_ue.to_csv(out_per_ue, index=False)
            print(f"[OK] saved: {out_per_ue}")

    df_summary = pd.DataFrame(all_summary)

    if df_summary.empty:
        raise RuntimeError("Nenhum dado processado.")

    cols = [
        "bandwidth",
        "mode",
        "n_slots",
        "n_rntis",
        "mean_buf_req",
        "median_buf_req",
        "max_buf_req",
        "sum_buf_req_mean_per_slot",
        "sum_buf_req_max_per_slot",
        "mean_backlogged_ues_per_slot",
        "mean_served_ues_per_slot",
        "pct_buf_gt0_alloc_eq0",
        "pct_buf_eq0_alloc_eq0",
        "pct_buf_gt0_alloc_gt0",
        "mean_jain_buf_req_per_slot",
        "mean_jain_alloc_per_slot",
    ]

    df_summary = df_summary[cols]
    out_summary = out_dir / "backlog_summary.csv"
    df_summary.to_csv(out_summary, index=False)

    print("\n=== BACKLOG SUMMARY ===")
    print(df_summary.to_string(index=False))

    print(f"\n[OK] saved: {out_summary}")


if __name__ == "__main__":
    main()