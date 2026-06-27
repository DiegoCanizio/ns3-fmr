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

LABELS = {
    "rr": "RR",
    "pf": "PF",
    "mr": "MR",
}

# Cores fixas apenas para manter consistência entre figuras.
COLORS = {
    "rr": "#4C78A8",
    "pf": "#F58518",
    "mr": "#54A24B",
}

MARKERS = {
    "rr": "o",
    "pf": "s",
    "mr": "^",
}

OUT_DIR = RUN_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# t crítico aproximado para IC95% e IC99%
# ==========================================================
def tcrit(n, confidence):
    if n <= 1:
        return 0.0

    df = n - 1

    if confidence == 0.95:
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

    if confidence == 0.99:
        table = {
            1: 63.657,
            2: 9.925,
            3: 5.841,
            4: 4.604,
            5: 4.032,
            6: 3.707,
            7: 3.499,
            8: 3.355,
            9: 3.250,
            10: 3.169,
            11: 3.106,
            12: 3.055,
            13: 3.012,
            14: 2.977,
            15: 2.947,
            16: 2.921,
            17: 2.898,
            18: 2.878,
            19: 2.861,
            20: 2.845,
            21: 2.831,
            22: 2.819,
            23: 2.807,
            24: 2.797,
            25: 2.787,
            26: 2.779,
            27: 2.771,
            28: 2.763,
            29: 2.756,
            30: 2.750,
        }

        if df in table:
            return table[df]
        if df <= 40:
            return 2.704
        if df <= 60:
            return 2.660
        if df <= 100:
            return 2.626
        return 2.576

    raise ValueError("confidence deve ser 0.95 ou 0.99")

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
                "ic95_rel": "",
                "ci95_low": "",
                "ci95_high": "",
                "ic99": "",
                "ic99_rel": "",
                "ci99_low": "",
                "ci99_high": "",
            })
            continue

        mean = statistics.mean(values)
        median = statistics.median(values)
        std = statistics.stdev(values) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 1 else 0.0

        ic95 = tcrit(n, 0.95) * se
        ic99 = tcrit(n, 0.99) * se

        rows.append({
            "mode": mode,
            "bandwidth_mhz": bw,
            "n": n,
            "mean": mean,
            "median": median,
            "std": std,
            "ic95": ic95,
            "ic95_rel": (ic95 / mean) * 100 if mean != 0 else 0.0,
            "ci95_low": mean - ic95,
            "ci95_high": mean + ic95,
            "ic99": ic99,
            "ic99_rel": (ic99 / mean) * 100 if mean != 0 else 0.0,
            "ci99_low": mean - ic99,
            "ci99_high": mean + ic99,
        })

# ==========================================================
# Salvar CSV
# ==========================================================
csv_path = OUT_DIR / "ci95_ci99_baselines_by_bw_shaded.csv"

with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "mode",
        "bandwidth_mhz",
        "n",
        "mean",
        "median",
        "std",
        "ic95",
        "ic95_rel",
        "ci95_low",
        "ci95_high",
        "ic99",
        "ic99_rel",
        "ci99_low",
        "ci99_high",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] CSV salvo em: {csv_path}")

# ==========================================================
# Função para gerar gráfico sombreado
# ==========================================================
def plot_shaded_ci(ci_col, ci_label, out_stem):
    plt.figure(figsize=(12, 6))

    for mode in MODES:
        mode_rows = [
            r for r in rows
            if r["mode"] == mode and r["n"] != 0
        ]

        mode_rows = sorted(mode_rows, key=lambda r: r["bandwidth_mhz"])

        xs = [r["bandwidth_mhz"] for r in mode_rows]
        means = [r["mean"] for r in mode_rows]
        cis = [r[ci_col] for r in mode_rows]

        lower = [m - c for m, c in zip(means, cis)]
        upper = [m + c for m, c in zip(means, cis)]

        color = COLORS[mode]

        # Faixa sombreada.
        plt.fill_between(
            xs,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0
        )

        # Linha da média.
        plt.plot(
            xs,
            means,
            marker=MARKERS[mode],
            linewidth=2.0,
            markersize=5.5,
            color=color,
            label=f"{LABELS[mode]} média ± {ci_label}"
        )

    plt.xlabel("Largura de banda (MHz)")
    plt.ylabel("Vazão agregada média (Mbps)")
    plt.title(f"Baselines: vazão agregada média com {ci_label}")
    plt.grid(True, alpha=0.30)
    plt.legend()
    plt.tight_layout()

    png_path = OUT_DIR / f"{out_stem}.png"
    pdf_path = OUT_DIR / f"{out_stem}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"[OK] Gráfico salvo em: {png_path}")
    print(f"[OK] Gráfico salvo em: {pdf_path}")

# ==========================================================
# Gerar figuras
# ==========================================================
plot_shaded_ci("ic95", "IC95%", "throughput_shaded_ci95_baselines")
plot_shaded_ci("ic99", "IC99%", "throughput_shaded_ci99_baselines")

# ==========================================================
# Resumo no terminal
# ==========================================================
print()
print("=" * 120)
print("RESUMO DOS BASELINES POR BANDA")
print("=" * 120)

for r in rows:
    if r["n"] == 0:
        print(f"{LABELS[r['mode']]:>3} | BW={r['bandwidth_mhz']:>3} MHz | n=0")
    else:
        print(
            f"{LABELS[r['mode']]:>3} | "
            f"BW={r['bandwidth_mhz']:>3} MHz | "
            f"n={r['n']:>3} | "
            f"média={r['mean']:.4f} Mbps | "
            f"IC95=±{r['ic95']:.4f} ({r['ic95_rel']:.2f}%) | "
            f"IC99=±{r['ic99']:.4f} ({r['ic99_rel']:.2f}%) | "
            f"mediana={r['median']:.4f}"
        )
