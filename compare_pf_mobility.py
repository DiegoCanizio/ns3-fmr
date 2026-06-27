from pathlib import Path
import math
import re
import pandas as pd
import numpy as np

CMP = Path("compare_runs/expC_pf_70mhz_mob_on_vs_off_same_load")
OUT = CMP / "comparison_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv_auto(path: Path):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def find_col(df, preferred=(), contains_all=(), contains_any=(), exclude=()):
    norm_map = {c: norm(c) for c in df.columns}

    for p in preferred:
        pn = norm(p)
        for c, cn in norm_map.items():
            if cn == pn:
                return c

    for c, cn in norm_map.items():
        if exclude and any(e in cn for e in exclude):
            continue
        if contains_all and not all(x in cn for x in contains_all):
            continue
        if contains_any and not any(x in cn for x in contains_any):
            continue
        return c

    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def parse_summary_txt(path: Path):
    out = {}
    if not path.exists():
        return out

    txt = path.read_text(errors="ignore")

    patterns = {
        "summary_thr_mbps": [
            r"aggregate\s+throughput.*?([0-9]+(?:\.[0-9]+)?)\s*mbps",
            r"dl\s+aggregate\s+throughput.*?([0-9]+(?:\.[0-9]+)?)\s*mbps",
            r"throughput.*?([0-9]+(?:\.[0-9]+)?)\s*mbps",
        ],
        "summary_delay_ms": [
            r"mean\s+delay.*?([0-9]+(?:\.[0-9]+)?)\s*ms",
            r"delay.*?([0-9]+(?:\.[0-9]+)?)\s*ms",
        ],
        "summary_loss_pct": [
            r"loss.*?([0-9]+(?:\.[0-9]+)?)\s*%",
            r"packet\s+loss.*?([0-9]+(?:\.[0-9]+)?)",
        ],
    }

    low = txt.lower()
    for k, regs in patterns.items():
        for rgx in regs:
            m = re.search(rgx, low)
            if m:
                out[k] = float(m.group(1))
                break

    return out

def metrics_from_run(run_dir: Path):
    row = {}

    flow = run_dir / "flow_summary_pf.csv"
    if flow.exists():
        df = read_csv_auto(flow)

        thr_col = find_col(
            df,
            preferred=("throughput_mbps", "rx_throughput_mbps", "throughputMbps", "throughput"),
            contains_any=("throughput", "thr"),
            exclude=("pct", "percent")
        )

        if thr_col is not None:
            vals = to_num(df[thr_col])
            row["flow_thr_mbps_sum"] = float(vals.sum(skipna=True))
            row["flow_thr_mbps_mean"] = float(vals.mean(skipna=True))

        delay_col = find_col(
            df,
            preferred=("mean_delay_ms", "delay_ms", "avg_delay_ms"),
            contains_all=("delay",),
            contains_any=("ms",)
        )

        if delay_col is not None:
            vals = to_num(df[delay_col])
            row["flow_delay_ms_mean"] = float(vals.mean(skipna=True))

        lost_col = find_col(df, preferred=("lostPackets", "lost_packets"), contains_any=("lost",))
        tx_col = find_col(df, preferred=("txPackets", "tx_packets"), contains_all=("tx", "packet"))
        rx_col = find_col(df, preferred=("rxPackets", "rx_packets"), contains_all=("rx", "packet"))

        if lost_col is not None:
            lost = to_num(df[lost_col]).sum(skipna=True)
            row["lost_packets_sum"] = float(lost)

            if tx_col is not None:
                tx = to_num(df[tx_col]).sum(skipna=True)
                if tx > 0:
                    row["loss_ratio_tx"] = float(lost / tx)

            if rx_col is not None:
                rx = to_num(df[rx_col]).sum(skipna=True)
                if rx + lost > 0:
                    row["loss_ratio_rx_lost"] = float(lost / (rx + lost))

    slot = run_dir / "slot_log_pf.csv"
    if slot.exists():
        df = read_csv_auto(slot)
        thr_cols = []
        for c in df.columns:
            cn = norm(c)
            if ("thr" in cn or "throughput" in cn) and "pct" not in cn and "percent" not in cn:
                vals = to_num(df[c])
                if vals.notna().sum() > 0:
                    thr_cols.append(c)

        if thr_cols:
            mat = df[thr_cols].apply(to_num)
            row["slot_thr_mean_all_cols"] = float(mat.mean(axis=1).mean(skipna=True))
            row["slot_thr_sum_all_cols_mean"] = float(mat.sum(axis=1).mean(skipna=True))
            row["slot_zero_thr_fraction"] = float((mat.sum(axis=1) <= 0).mean())

    row.update(parse_summary_txt(run_dir / "summary.txt"))
    return row

def ci_mean(values, level=0.99):
    vals = np.array([v for v in values if pd.notna(v)], dtype=float)
    n = len(vals)
    if n <= 1:
        return np.nan, np.nan, np.nan, n

    mean = vals.mean()
    sd = vals.std(ddof=1)

    try:
        from scipy.stats import t
        crit = t.ppf((1 + level) / 2, n - 1)
    except Exception:
        crit = 2.576 if abs(level - 0.99) < 1e-9 else 1.96

    half = crit * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n

def main():
    on_dirs = {p.name: p for p in (CMP / "mob_on").iterdir() if p.is_dir()}
    off_dirs = {p.name: p for p in (CMP / "mob_off").iterdir() if p.is_dir()}

    paired = sorted(set(on_dirs) & set(off_dirs))
    missing_on = sorted(set(off_dirs) - set(on_dirs))
    missing_off = sorted(set(on_dirs) - set(off_dirs))

    (OUT / "paired_seeds.txt").write_text("\n".join(paired) + "\n")
    (OUT / "missing_in_mob_on.txt").write_text("\n".join(missing_on) + ("\n" if missing_on else ""))
    (OUT / "missing_in_mob_off.txt").write_text("\n".join(missing_off) + ("\n" if missing_off else ""))

    rows = []
    for seed in paired:
        for cond, dct in [("mob_on", on_dirs), ("mob_off", off_dirs)]:
            m = metrics_from_run(dct[seed])
            m["seed"] = seed
            m["condition"] = cond
            rows.append(m)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)

    delta_rows = []
    metrics = [c for c in per_seed.columns if c not in ("seed", "condition")]
    wide = per_seed.pivot(index="seed", columns="condition", values=metrics)

    for metric in metrics:
        if (metric, "mob_on") not in wide.columns or (metric, "mob_off") not in wide.columns:
            continue

        delta = wide[(metric, "mob_on")] - wide[(metric, "mob_off")]
        rel = 100.0 * delta / wide[(metric, "mob_off")].replace(0, np.nan)

        for seed in wide.index:
            delta_rows.append({
                "seed": seed,
                "metric": metric,
                "mob_on": wide.loc[seed, (metric, "mob_on")],
                "mob_off": wide.loc[seed, (metric, "mob_off")],
                "delta_on_minus_off": delta.loc[seed],
                "relative_delta_pct": rel.loc[seed],
            })

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(OUT / "paired_deltas.csv", index=False)

    summary_rows = []
    for metric in metrics:
        sub = deltas[deltas["metric"] == metric]
        if sub.empty:
            continue

        mean_on, lo_on, hi_on, n_on = ci_mean(sub["mob_on"], 0.99)
        mean_off, lo_off, hi_off, n_off = ci_mean(sub["mob_off"], 0.99)
        mean_delta, lo_delta, hi_delta, n_delta = ci_mean(sub["delta_on_minus_off"], 0.99)
        mean_rel, lo_rel, hi_rel, n_rel = ci_mean(sub["relative_delta_pct"], 0.99)

        summary_rows.append({
            "metric": metric,
            "n_paired": n_delta,
            "mob_on_mean": mean_on,
            "mob_on_ic99_low": lo_on,
            "mob_on_ic99_high": hi_on,
            "mob_off_mean": mean_off,
            "mob_off_ic99_low": lo_off,
            "mob_off_ic99_high": hi_off,
            "delta_on_minus_off_mean": mean_delta,
            "delta_ic99_low": lo_delta,
            "delta_ic99_high": hi_delta,
            "relative_delta_pct_mean": mean_rel,
            "relative_delta_pct_ic99_low": lo_rel,
            "relative_delta_pct_ic99_high": hi_rel,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_mobility_effect_ic99.csv", index=False)

    print(f"[OK] Seeds pareadas: {len(paired)}")
    print(f"[OK] Faltando em mob_on: {len(missing_on)}")
    print(f"[OK] Faltando em mob_off: {len(missing_off)}")
    print(f"[OK] Arquivos gerados em: {OUT}")
    print()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
