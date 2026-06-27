from pathlib import Path
import csv
import math
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Ajuste aqui se o run-id for outro
# ==========================================================
RUN_DIR = Path("~/ns3-fmr/compare_runs/expA_fr1_100seeds_30s_fmr").expanduser()
SCENARIO = "dynamic_continuous_30s"

BANDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
MODE = "fmr_rl"

OUT_DIR = RUN_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# t crítico aproximado para IC95%
# ==========================================================
def tcrit95(n):
    if n <= 1:
        return 0.0

    df = n - 1

    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }

    if df in table:
        return table[df]

    if df <= 40:
        return 2.021

    if df <= 60:
        return 2.000

    if df <= 100:
        return 1.984

    return 1.960

# ==========================================================
# Coleta vazão agregada por seed e banda
# ==========================================================
data = defaultdict(list)

for bw in BANDS:
    pattern = f"seed_*/{SCENARIO}/bw{bw}/{MODE}/flow_summary_{MODE}.csv"

    for p in RUN_DIR.glob(pattern):
        if not p.exists() or p.stat().st_size == 0:
            continue

        try:
            seed_name = p.relative_to(RUN_DIR).parts[0]
            seed = int(seed_name.replace("seed_", ""))

            agg_thr = 0.0

            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agg_thr += float(row["throughput_mbps"])

            data[bw].append((seed, agg_thr))

        except Exception as e:
            print(f"[WARN] Ignorando arquivo com problema: {p} | {e}")

# ==========================================================
# Estatísticas
# ==========================================================
rows = []

for bw in BANDS:
    values = [v for _, v in data[bw]]
    n = len(values)

    if n == 0:
        rows.append({
            "mode": MODE,
            "bandwidth_mhz": bw,
            "n": 0,
            "mean": "",
            "median": "",
            "std": "",
            "ic95": "",
            "ic_rel": "",
            "ci_low": "",
            "ci_high": "",
        })
        continue

    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0

    tcrit = tcrit95(n)
    ic95 = tcrit * se
    ci_low = mean - ic95
    ci_high = mean + ic95
    ic_rel = (ic95 / mean) * 100 if mean != 0 else 0.0

    rows.append({
        "mode": MODE,
        "bandwidth_mhz": bw,
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "ic95": ic95,
        "ic_rel": ic_rel,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })

# ==========================================================
# Salvar CSV
# ==========================================================
csv_path = OUT_DIR / "preview_ci_fmr_by_bw.csv"

with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "mode",
        "bandwidth_mhz",
        "n",
        "mean",
        "median",
        "std",
        "ic95",
        "ic_rel",
        "ci_low",
        "ci_high",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] CSV salvo em: {csv_path}")

# ==========================================================
# Gráfico 1: média ± IC95%
# ==========================================================
plt.figure(figsize=(12, 6))

xs = []
ys = []
yerr = []

for r in rows:
    if r["n"] == 0:
        continue

    xs.append(r["bandwidth_mhz"])
    ys.append(r["mean"])
    yerr.append(r["ic95"])

if xs:
    plt.errorbar(
        xs,
        ys,
        yerr=yerr,
        marker="o",
        capsize=4,
        linewidth=1.8,
        label="IA-FMR"
    )

plt.xlabel("Largura de banda (MHz)")
plt.ylabel("Vazão agregada média (Mbps)")
plt.title("Prévia do IA-FMR: vazão agregada média com IC95%")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path = OUT_DIR / "preview_throughput_ci_fmr.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"[OK] Gráfico salvo em: {fig_path}")

# ==========================================================
# Gráfico 2: IC relativo
# ==========================================================
plt.figure(figsize=(12, 6))

xs = []
ys = []

for r in rows:
    if r["n"] == 0:
        continue

    xs.append(r["bandwidth_mhz"])
    ys.append(r["ic_rel"])

if xs:
    plt.plot(
        xs,
        ys,
        marker="o",
        linewidth=1.8,
        label="IA-FMR"
    )

plt.axhline(20, linestyle="--", linewidth=1.2)
plt.xlabel("Largura de banda (MHz)")
plt.ylabel("IC relativo (%)")
plt.title("Prévia do IA-FMR: IC relativo da vazão agregada")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path2 = OUT_DIR / "preview_ic_relative_fmr.png"
plt.savefig(fig_path2, dpi=300)
plt.close()

print(f"[OK] Gráfico salvo em: {fig_path2}")

# ==========================================================
# Resumo no terminal
# ==========================================================
print()
print("=" * 90)
print("RESUMO DO IA-FMR POR BANDA")
print("=" * 90)

for r in rows:
    if r["n"] == 0:
        print(f"IA-FMR | BW={r['bandwidth_mhz']:>3} MHz | n=0")
    else:
        print(
            f"IA-FMR | "
            f"BW={r['bandwidth_mhz']:>3} MHz | "
            f"n={r['n']:>3} | "
            f"média={r['mean']:.4f} Mbps | "
            f"IC95=±{r['ic95']:.4f} | "
            f"ICrel={r['ic_rel']:.2f}% | "
            f"mediana={r['median']:.4f}"
        )
