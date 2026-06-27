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
RUN_DIR = Path("~/ns3-fmr/compare_runs/expA_fr1_100seeds_30s_baselines").expanduser()
SCENARIO = "dynamic_continuous_30s"

BANDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
MODES = ["rr", "pf", "mr"]

OUT_DIR = RUN_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Coleta vazão agregada por seed, banda e escalonador
# ==========================================================
data = defaultdict(list)

for mode in MODES:
    for bw in BANDS:
        pattern = f"seed_*/{SCENARIO}/bw{bw}/{mode}/flow_summary_{mode}.csv"

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

                data[(mode, bw)].append((seed, agg_thr))

            except Exception as e:
                print(f"[WARN] Ignorando arquivo com problema: {p} | {e}")

# ==========================================================
# Estatísticas
# ==========================================================
rows = []

for mode in MODES:
    for bw in BANDS:
        values = [v for _, v in data[(mode, bw)]]
        n = len(values)

        if n == 0:
            rows.append({
                "mode": mode,
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

        # Aproximação adequada para n geralmente acima de 30.
        # Para n menor, ainda serve como prévia.
        tcrit = 1.96 if n >= 30 else 2.10

        ic95 = tcrit * se
        ci_low = mean - ic95
        ci_high = mean + ic95
        ic_rel = (ic95 / mean) * 100 if mean != 0 else 0.0

        rows.append({
            "mode": mode,
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
# Salvar CSV da prévia
# ==========================================================
csv_path = OUT_DIR / "preview_ci_by_bw_mode.csv"

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

for mode in MODES:
    xs = []
    ys = []
    yerr = []

    for r in rows:
        if r["mode"] != mode or r["n"] == 0:
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
            label=mode.upper()
        )

plt.xlabel("Largura de banda (MHz)")
plt.ylabel("Vazão agregada média (Mbps)")
plt.title("Prévia dos baselines: vazão agregada média com IC95%")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path = OUT_DIR / "preview_throughput_ci_baselines.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"[OK] Gráfico salvo em: {fig_path}")

# ==========================================================
# Gráfico 2: IC relativo por banda
# ==========================================================
plt.figure(figsize=(12, 6))

for mode in MODES:
    xs = []
    ys = []

    for r in rows:
        if r["mode"] != mode or r["n"] == 0:
            continue

        xs.append(r["bandwidth_mhz"])
        ys.append(r["ic_rel"])

    if xs:
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            label=mode.upper()
        )

plt.axhline(20, linestyle="--", linewidth=1.2)
plt.xlabel("Largura de banda (MHz)")
plt.ylabel("IC relativo (%)")
plt.title("Prévia dos baselines: IC relativo da vazão agregada")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path2 = OUT_DIR / "preview_ic_relative_baselines.png"
plt.savefig(fig_path2, dpi=300)
plt.close()

print(f"[OK] Gráfico salvo em: {fig_path2}")

# ==========================================================
# Resumo no terminal
# ==========================================================
print()
print("=" * 90)
print("RESUMO POR MODO E BANDA")
print("=" * 90)

for r in rows:
    if r["n"] == 0:
        print(f"{r['mode'].upper():>3} | BW={r['bandwidth_mhz']:>3} MHz | n=0")
    else:
        print(
            f"{r['mode'].upper():>3} | "
            f"BW={r['bandwidth_mhz']:>3} MHz | "
            f"n={r['n']:>3} | "
            f"média={r['mean']:.4f} Mbps | "
            f"IC95=±{r['ic95']:.4f} | "
            f"ICrel={r['ic_rel']:.2f}% | "
            f"mediana={r['median']:.4f}"
        )
