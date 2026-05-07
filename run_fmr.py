#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executa a matriz experimental do IA-FMR no ns-3 e gera automaticamente:
  1) tabelas comparativas de QoS/throughput;
  2) trade-off throughput × fairness;
  3) diagnósticos selecionados de fairness/starvation;
  4) estatísticas e gráficos de backlog/starvation real.

Uso típico:
  python3 run_fmr_dynamic_experiments.py

Ou com run id manual:
  python3 run_fmr_dynamic_experiments.py --run-id teste_final

Observação metodológica:
  O modelo IA-FMR é treinado offline. Este script apenas executa a inferência no ns-3.
  O dinamismo aqui é tratado como dinamismo dos fluxos de tráfego/carga, não mobilidade.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1) CONFIGURAÇÃO GERAL, FÁCIL DE EDITAR
# ==========================================================

BASE_DIR = Path(os.environ.get("FMR_BASE_DIR", "~/ns3-fmr")).expanduser().resolve()
BIN = Path(os.environ.get("FMR_BIN", str(BASE_DIR / "build/scratch/ns3.46-fmr-compara-qos-default"))).expanduser().resolve()
MODEL_DIR = Path(os.environ.get("FMR_MODEL_DIR", str(BASE_DIR / "scratch/fmr_ai/models"))).expanduser().resolve()
VENV_ACTIVATE = BASE_DIR / ".venv/bin/activate"

MODES = ["rr", "pf", "mr", "fmr_rl"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "IA-FMR"}
MODE_ORDER = {m: i for i, m in enumerate(MODES)}
MARKERS = {"rr": "o", "pf": "s", "mr": "^", "fmr_rl": "D"}
COLORS = {
    "rr": "#4C78A8",
    "pf": "#F58518",
    "mr": "#54A24B",
    "fmr_rl": "#B279A2",
}

# Centralização visual dos gráficos.
FIGSIZE_SINGLE = (8.5, 5.3)
FIGSIZE_WIDE = (11.5, 5.8)
FIG_DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
LINE_WIDTH = 1.6
GRID_ALPHA = 0.35

# Parâmetros ns-3 comuns.
COMMON_NS3_ARGS = {
    "simTime": "5s",
    "scenarioRadiusMin": "20",
    "scenarioRadiusMax": "150",
    "enableMobility": "1",       # mobilidade pode existir no cenário, mas não é a hipótese central de dinamismo
    "mobilitySpeedMin": "0.5",
    "mobilitySpeedMax": "1.5",
    "mobilityBounds": "200",
    "mobilityDistance": "5",
    "udpPacketSize": "3000",
    "EnableSlotCsv": "1",
    "SlotCsvAppend": "0",
    "SlotCsvFlush": "1",
    "EnableUeSnapshotCsv": "1",
    "UeSnapshotPeriod": "100ms",
    "EnableFlowSummaryCsv": "1",
    "logging": "1",
}

# Parâmetros do agente IA-FMR.
AGENT_ARGS = {
    "shm": "4096",
    "is-creator": "0",
    "b2": "1",
    "deterministic": "1",
    "print-every": "50",
    "wait-seconds": "120",
    "n-ues": "9",
    "obs-dim": "29",
    "act-dim": "10",
    "a-min": "0.80",
    "a-max": "0.98",
    "alpha-temp": "0.70",
    "default-alpha": "0.80",
}

# Tau por largura de banda. Ajuste aqui se necessário.
TAU_BY_BW = {10: 0.65, 20: 0.65, 50: 0.65, 100: 0.65}


@dataclass
class Scenario:
    name: str
    purpose: str
    bandwidths_mhz: list[int]
    lambda_value: int
    num_dl_flows_per_ue: int
    sim_time: str = "5s"
    udp_packet_size: int = 3000
    profile: str = "static"
    phase: str = "single"
    phase_order: int = 0
    extra_ns3_args: dict[str, str] = field(default_factory=dict)


# ==========================================================
# MATRIZ EXPERIMENTAL
# ==========================================================
#
# IMPORTANTE:
# O binário ns3.46-fmr-compara-qos-default recebe --lambda como valor fixo
# por execução. Portanto, enquanto o C++ não expuser um gerador ON/OFF ou
# uma agenda temporal de lambda dentro da mesma simulação, o dinamismo dos
# fluxos é modelado aqui como uma sequência controlada de fases de carga.
#
# Essa escolha é propositalmente conservadora, porque você já observou crash
# quando lambda fica alto demais. Assim, o perfil dinâmico abaixo evita valores
# extremos como 1200 e mantém o pico padrão em 850. Se o simulador suportar
# mais carga com estabilidade, basta ajustar SAFE_DYNAMIC_LAMBDA_PROFILE.

SAFE_MAX_LAMBDA = int(os.environ.get("FMR_SAFE_MAX_LAMBDA", "850"))

# Perfil dinâmico seguro: baixa carga -> aumento -> burst -> recuperação -> novo burst.
# Formato: (fase, lambda, fluxos DL por UE, duração).
SAFE_DYNAMIC_LAMBDA_PROFILE = [
    ("p01_low_load",      250,  6, "6s"),
    ("p02_ramp_up",       450,  8, "6s"),
    ("p03_safe_burst",    700, 10, "8s"),
    ("p04_recovery",      350,  8, "6s"),
    ("p05_second_burst",  850, 10, "8s"),
]

# Caso queira testar um limite superior depois, faça por variável de ambiente:
#   FMR_SAFE_MAX_LAMBDA=900 python3 run_fmr_dynamic_experiments.py --only-scenario dynamic_p05_second_burst
# O script corta automaticamente qualquer lambda acima de SAFE_MAX_LAMBDA.


def cap_lambda(value: int) -> int:
    return min(int(value), SAFE_MAX_LAMBDA)


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    # 1) Referência estável. Serve como base para comparar com as fases dinâmicas.
    scenarios.append(Scenario(
        name="steady_reference",
        purpose="Referência estável com carga moderada, usada como ponto base antes das fases dinâmicas.",
        bandwidths_mhz=[20],
        lambda_value=300,
        num_dl_flows_per_ue=6,
        sim_time="6s",
        profile="static_reference",
        phase="steady",
        phase_order=0,
    ))

    # 2) Congestionamento por largura de banda. Mantém lambda seguro e varia BW.
    scenarios.append(Scenario(
        name="congested_bw_sweep",
        purpose="Cenário congestionado por limitação de banda; avalia robustez em 10, 20 e 50 MHz.",
        bandwidths_mhz=[10, 20, 50],
        lambda_value=600,
        num_dl_flows_per_ue=10,
        sim_time="6s",
        profile="bw_congestion",
        phase="single",
        phase_order=0,
    ))

    # 3) Perfil dinâmico por fases. Este é o bloco central para provar dinamismo dos fluxos.
    #    Usa 10 MHz para forçar competição e 20 MHz para verificar se o comportamento se mantém.
    for idx, (phase, lam, flows, sim_time) in enumerate(SAFE_DYNAMIC_LAMBDA_PROFILE, start=1):
        scenarios.append(Scenario(
            name=f"dynamic_{phase}",
            purpose=(
                "Fase do perfil dinâmico de tráfego: baixa carga, aumento, burst, "
                "recuperação ou segundo burst. Usada para avaliar resposta a variações de fluxo/carga."
            ),
            bandwidths_mhz=[10, 20],
            lambda_value=cap_lambda(lam),
            num_dl_flows_per_ue=flows,
            sim_time=sim_time,
            profile="dynamic_flow_profile",
            phase=phase,
            phase_order=idx,
        ))

    # 4) Estresse controlado opcional, ainda abaixo do limite seguro por padrão.
    scenarios.append(Scenario(
        name="stress_safe_limit",
        purpose="Estresse controlado no limite seguro de lambda para observar backlog, starvation e perda sem forçar crash.",
        bandwidths_mhz=[10],
        lambda_value=SAFE_MAX_LAMBDA,
        num_dl_flows_per_ue=10,
        sim_time="8s",
        profile="safe_stress",
        phase="safe_limit",
        phase_order=99,
    ))

    return scenarios


SCENARIOS: list[Scenario] = build_scenarios()

# Se o fmr-compara-qos passar a expor argumentos específicos de tráfego ON/OFF ou bursts,
# adicione-os em extra_ns3_args dentro de build_scenarios(). Exemplo hipotético:
# extra_ns3_args={"trafficProfile": "onoff", "burstPeriod": "500ms", "offPeriod": "200ms"}


# ==========================================================
# 2) UTILITÁRIOS GERAIS
# ==========================================================

def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": FIG_DPI,
    })


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def bw_to_hz(bw_mhz: int) -> int:
    return int(bw_mhz * 1_000_000)


def tau_for_bw(bw_mhz: int) -> float:
    return float(TAU_BY_BW.get(bw_mhz, 0.65))


def label(mode: str) -> str:
    return LABELS.get(mode, mode)


def ensure_executable(path: Path) -> None:
    if not path.exists() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Binário não encontrado ou não executável: {path}\nRode: ./ns3 build")


def cleanup_ai() -> None:
    subprocess.run("pkill -f 'scratch/fmr_ai/agent.py' || true", shell=True)
    for pattern in ["/dev/shm/ns3ai_fmr*", "/dev/shm/fmr_cpp2py*", "/dev/shm/fmr_py2cpp*", "/dev/shm/fmr_lock*"]:
        subprocess.run(f"rm -f {pattern} || true", shell=True)


def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def build_ns3_args(base: dict[str, Any]) -> list[str]:
    args = []
    for k, v in base.items():
        args.append(f"--{k}={v}")
    return args


# ==========================================================
# 3) EXECUÇÃO DO NS-3 E DO AGENTE IA-FMR
# ==========================================================

def run_mode(run_dir: Path, scenario: Scenario, bw_mhz: int, mode: str) -> None:
    bw_hz = bw_to_hz(bw_mhz)
    tau = tau_for_bw(bw_mhz)
    mode_dir = run_dir / scenario.name / f"bw{bw_mhz}" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"[RUN] scenario={scenario.name} | bw={bw_mhz}MHz | mode={mode}")
    print("=" * 72)

    ns3_args = dict(COMMON_NS3_ARGS)
    ns3_args.update({
        "schedulerMode": mode,
        "EnableNs3Ai": "0",
        "FmrTau": tau,
        "simTime": scenario.sim_time,
        "bandwidth": bw_hz,
        "lambda": scenario.lambda_value,
        "udpPacketSize": scenario.udp_packet_size,
        "numDlFlowsPerUe": scenario.num_dl_flows_per_ue,
        "SlotCsvPath": mode_dir / f"slot_log_{mode}.csv",
        "UeSnapshotCsvPath": mode_dir / f"ue_snapshot_{mode}.csv",
        "FlowSummaryCsvPath": mode_dir / f"flow_summary_{mode}.csv",
        "simTag": "summary.txt",
        "outputDir": mode_dir,
    })
    ns3_args.update(scenario.extra_ns3_args)

    cmd = [str(BIN)] + build_ns3_args(ns3_args)
    rc = run_cmd(cmd, mode_dir / "ns3.log", cwd=BASE_DIR)
    if rc != 0:
        raise RuntimeError(f"Falha no ns-3: scenario={scenario.name}, bw={bw_mhz}, mode={mode}, rc={rc}")


def start_agent(run_dir: Path, scenario: Scenario, bw_mhz: int, mode_dir: Path) -> subprocess.Popen:
    tau = tau_for_bw(bw_mhz)
    model_path = MODEL_DIR / f"model_{bw_mhz}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    segment = f"ns3ai_fmr_{scenario.name}_bw{bw_mhz}"
    cpp2py = f"fmr_cpp2py_{scenario.name}_bw{bw_mhz}"
    py2cpp = f"fmr_py2cpp_{scenario.name}_bw{bw_mhz}"
    lock = f"fmr_lock_{scenario.name}_bw{bw_mhz}"

    cleanup_ai()
    mode_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE_DIR}/contrib/ai/model/gym-interface/py:{env.get('PYTHONPATH', '')}"

    agent_cmd = [
        "bash", "-lc",
        "source " + str(VENV_ACTIVATE) + " && " +
        "python3 -u scratch/fmr_ai/agent.py " +
        f"--model {model_path} " +
        f"--segment {segment} --cpp2py {cpp2py} --py2cpp {py2cpp} --lock {lock} " +
        " ".join(f"--{k} {v}" for k, v in AGENT_ARGS.items()) +
        f" --tau {tau}"
    ]

    log_path = mode_dir / "agent_fmr_rl.log"
    log = log_path.open("w", encoding="utf-8")
    print("[AGENT]", " ".join(agent_cmd))
    proc = subprocess.Popen(agent_cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, preexec_fn=os.setsid)

    assert proc.stdout is not None

    def _drain_stdout(p: subprocess.Popen, file_obj):
        for line in p.stdout:  # type: ignore[union-attr]
            print(line, end="")
            file_obj.write(line)
            file_obj.flush()

    import threading
    t = threading.Thread(target=_drain_stdout, args=(proc, log), daemon=True)
    t.start()
    time.sleep(2)

    proc._fmr_agent_log = log  # type: ignore[attr-defined]
    proc._fmr_ai_names = {"segment": segment, "cpp2py": cpp2py, "py2cpp": py2cpp, "lock": lock}  # type: ignore[attr-defined]
    return proc


def stop_agent(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        proc._fmr_agent_log.close()  # type: ignore[attr-defined]
    except Exception:
        pass
    cleanup_ai()


def run_fmr_rl(run_dir: Path, scenario: Scenario, bw_mhz: int) -> None:
    bw_hz = bw_to_hz(bw_mhz)
    tau = tau_for_bw(bw_mhz)
    mode = "fmr_rl"
    mode_dir = run_dir / scenario.name / f"bw{bw_mhz}" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"[RUN] scenario={scenario.name} | bw={bw_mhz}MHz | mode=IA-FMR")
    print("=" * 72)

    agent_proc = start_agent(run_dir, scenario, bw_mhz, mode_dir)
    names = agent_proc._fmr_ai_names  # type: ignore[attr-defined]

    ns3_args = dict(COMMON_NS3_ARGS)
    ns3_args.update({
        "schedulerMode": "fmr_rl",
        "EnableNs3Ai": "1",
        "AiCppIsCreator": "1",
        "AiVerbose": "1",
        "AiShmSize": AGENT_ARGS["shm"],
        "AiSegmentName": names["segment"],
        "AiCpp2PyName": names["cpp2py"],
        "AiPy2CppName": names["py2cpp"],
        "AiLockableName": names["lock"],
        "FmrTau": tau,
        "simTime": scenario.sim_time,
        "bandwidth": bw_hz,
        "lambda": scenario.lambda_value,
        "udpPacketSize": scenario.udp_packet_size,
        "numDlFlowsPerUe": scenario.num_dl_flows_per_ue,
        "SlotCsvPath": mode_dir / f"slot_log_{mode}.csv",
        "UeSnapshotCsvPath": mode_dir / f"ue_snapshot_{mode}.csv",
        "FlowSummaryCsvPath": mode_dir / f"flow_summary_{mode}.csv",
        "simTag": "summary.txt",
        "outputDir": mode_dir,
    })
    ns3_args.update(scenario.extra_ns3_args)

    try:
        cmd = [str(BIN)] + build_ns3_args(ns3_args)
        rc = run_cmd(cmd, mode_dir / "ns3.log", cwd=BASE_DIR)
        if rc != 0:
            raise RuntimeError(f"Falha no ns-3 IA-FMR: scenario={scenario.name}, bw={bw_mhz}, rc={rc}")
    finally:
        stop_agent(agent_proc)


def run_all_experiments(run_dir: Path, scenarios: list[Scenario]) -> None:
    ensure_executable(BIN)
    cleanup_ai()

    rows = []
    for s in scenarios:
        for bw in s.bandwidths_mhz:
            rows.append({
                "scenario": s.name,
                "purpose": s.purpose,
                "bandwidth_mhz": bw,
                "lambda": s.lambda_value,
                "num_dl_flows_per_ue": s.num_dl_flows_per_ue,
                "sim_time": s.sim_time,
                "udp_packet_size": s.udp_packet_size,
                "profile": s.profile,
                "phase": s.phase,
                "phase_order": s.phase_order,
                "extra_ns3_args": str(s.extra_ns3_args),
            })
    meta_dir = run_dir / "00_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(meta_dir / "scenario_matrix.csv", index=False)

    for scenario in scenarios:
        print("\n" + "#" * 72)
        print(f"[SCENARIO] {scenario.name}")
        print(f"[PURPOSE] {scenario.purpose}")
        print("#" * 72)
        for bw in scenario.bandwidths_mhz:
            for mode in ["rr", "pf", "mr"]:
                run_mode(run_dir, scenario, bw, mode)
            run_fmr_rl(run_dir, scenario, bw)


# ==========================================================
# 4) LEITURA, MÉTRICAS E TABELAS
# ==========================================================

def jain_index(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    ss = float((x ** 2).sum())
    if s <= 0.0 or ss <= 0.0:
        return 0.0
    return float((s * s) / (len(x) * ss))


def scenario_bw_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for scenario_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("00_") and not p.name.startswith("plots") and not p.name.startswith("tables")]):
        for bw_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("bw")], key=lambda p: int(p.name.replace("bw", ""))):
            out.append((scenario_dir.name, bw_dir))
    return out


def load_flow_summary(run_dir: Path) -> pd.DataFrame:
    dfs = []
    for scenario, bw_dir in scenario_bw_dirs(run_dir):
        for mode in MODES:
            path = bw_dir / mode / f"flow_summary_{mode}.csv"
            if not path.exists():
                print(f"[WARN] missing flow summary: {path}")
                continue
            df = pd.read_csv(path)
            df["scenario"] = scenario
            df["bandwidth"] = bw_dir.name
            df["mode"] = mode
            df["run_id"] = run_dir.name
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def summarize_flows(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_id, scenario, bandwidth, mode), df in df_all.groupby(["run_id", "scenario", "bandwidth", "mode"], sort=False):
        thr = df["throughput_mbps"]
        delay = df["mean_delay_ms"] if "mean_delay_ms" in df else pd.Series(dtype=float)
        jitter = df["mean_jitter_ms"] if "mean_jitter_ms" in df else pd.Series(dtype=float)
        loss = df["loss_ratio"] if "loss_ratio" in df else pd.Series(dtype=float)
        rows.append({
            "run_id": run_id,
            "scenario": scenario,
            "bandwidth": bandwidth,
            "mode": mode,
            "n_flows": int(len(df)),
            "aggregate_throughput_mbps": float(thr.sum()),
            "mean_flow_throughput_mbps": float(thr.mean()),
            "min_flow_throughput_mbps": float(thr.min()),
            "p5_flow_throughput_mbps": float(thr.quantile(0.05)),
            "max_flow_throughput_mbps": float(thr.max()),
            "std_flow_throughput_mbps": float(thr.std(ddof=0)),
            "mean_delay_ms": float(delay.mean()) if not delay.empty else np.nan,
            "max_delay_ms": float(delay.max()) if not delay.empty else np.nan,
            "mean_jitter_ms": float(jitter.mean()) if not jitter.empty else np.nan,
            "mean_loss_ratio": float(loss.mean()) if not loss.empty else np.nan,
            "max_loss_ratio": float(loss.max()) if not loss.empty else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bw_order"] = df["bandwidth"].str.replace("bw", "", regex=False).astype(int)
        df["mode_order"] = df["mode"].map(MODE_ORDER)
        df = df.sort_values(["scenario", "bw_order", "mode_order"]).drop(columns=["bw_order", "mode_order"]).reset_index(drop=True)
    return df


def load_scenario_metadata(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "00_metadata" / "scenario_matrix.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def attach_scenario_metadata(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if df.empty or meta.empty or "scenario" not in df.columns:
        return df
    cols = [c for c in ["scenario", "profile", "phase", "phase_order", "lambda", "num_dl_flows_per_ue", "sim_time"] if c in meta.columns]
    if not cols:
        return df
    meta_small = meta[cols].drop_duplicates(subset=["scenario"])
    return df.merge(meta_small, on="scenario", how="left")


def infer_total_rbg(bw_dir: Path) -> float:
    fmr_path = bw_dir / "fmr_rl" / "slot_log_fmr_rl.csv"
    df = pd.read_csv(fmr_path)
    g = df.groupby(["time_s", "beam_id"])["alloc_rbg"].sum()
    return float(round(g.median()))


def get_all_rntis(bw_dir: Path) -> list[int]:
    rr_path = bw_dir / "rr" / "slot_log_rr.csv"
    df = pd.read_csv(rr_path)
    return sorted(int(x) for x in df["rnti"].unique())


def load_slot_log_normalized(path: Path, total_rbg: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time_s", "beam_id", "rnti", "alloc_rbg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    slot_sum = df.groupby(["time_s", "beam_id"])["alloc_rbg"].transform("sum").astype(float)
    df["alloc_norm"] = np.where(slot_sum > 0, df["alloc_rbg"].astype(float) * float(total_rbg) / slot_sum, 0.0)
    return df


def slot_matrix(df: pd.DataFrame, all_rntis: list[int], value_col: str = "alloc_norm") -> pd.DataFrame:
    rows = []
    for _, g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = g.groupby("rnti")[value_col].sum().reindex(all_rntis, fill_value=0.0)
        rows.append(alloc)
    if not rows:
        return pd.DataFrame(columns=all_rntis)
    return pd.DataFrame(rows).reset_index(drop=True)


def load_slot_data(run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    data = {}
    for scenario, bw_dir in scenario_bw_dirs(run_dir):
        total_rbg = infer_total_rbg(bw_dir)
        all_rntis = get_all_rntis(bw_dir)
        for mode in MODES:
            path = bw_dir / mode / f"slot_log_{mode}.csv"
            if not path.exists():
                continue
            df = load_slot_log_normalized(path, total_rbg)
            mat = slot_matrix(df, all_rntis)
            data[(scenario, bw_dir.name, mode)] = {
                "df": df,
                "mat": mat,
                "jain": mat.apply(jain_index, axis=1),
                "active": (mat > 0).sum(axis=1),
                "zero_fraction_per_slot": (mat <= 0).mean(axis=1),
                "zero_fraction_per_ue": (mat <= 0).mean(axis=0),
                "total_rbg": total_rbg,
                "all_rntis": all_rntis,
            }
    return data


def summarize_tradeoff(flow_summary: pd.DataFrame, slot_data: dict[tuple[str, str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, r in flow_summary.iterrows():
        key = (r["scenario"], r["bandwidth"], r["mode"])
        d = slot_data.get(key)
        if d is None:
            continue
        rows.append({
            "run_id": r["run_id"],
            "scenario": r["scenario"],
            "bandwidth": r["bandwidth"],
            "mode": r["mode"],
            "aggregate_throughput_mbps": r["aggregate_throughput_mbps"],
            "mean_flow_throughput_mbps": r["mean_flow_throughput_mbps"],
            "p5_flow_throughput_mbps": r["p5_flow_throughput_mbps"],
            "jain_rbg_slot_mean": float(d["jain"].mean()),
            "mean_active_ues_per_slot": float(d["active"].mean()),
            "mean_zero_ue_percent": float(d["zero_fraction_per_slot"].mean() * 100.0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bw_order"] = df["bandwidth"].str.replace("bw", "", regex=False).astype(int)
        df["mode_order"] = df["mode"].map(MODE_ORDER)
        df = df.sort_values(["scenario", "bw_order", "mode_order"]).drop(columns=["bw_order", "mode_order"]).reset_index(drop=True)
    return df


def summarize_backlog(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_ue_rows = []

    for scenario, bw_dir in scenario_bw_dirs(run_dir):
        all_rntis = get_all_rntis(bw_dir)
        for mode in MODES:
            path = bw_dir / mode / f"slot_log_{mode}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            required = {"time_s", "beam_id", "rnti", "buf_req", "alloc_rbg"}
            if not required.issubset(df.columns):
                print(f"[WARN] backlog columns missing in {path}")
                continue

            rows = []
            for _, g in df.groupby(["time_s", "beam_id"], sort=False):
                slot = g.groupby("rnti").agg(buf_req=("buf_req", "max"), alloc_rbg=("alloc_rbg", "sum")).reindex(all_rntis, fill_value=0)
                rows.append(slot)
            if not rows:
                continue
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

            summary_rows.append({
                "scenario": scenario,
                "bandwidth": bw_dir.name,
                "mode": mode,
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
            })

            full["need_service"] = need_service
            full["served"] = served
            full["backlog_not_served"] = backlog_not_served
            per_ue = full.groupby("rnti").agg(
                mean_buf_req=("buf_req", "mean"),
                max_buf_req=("buf_req", "max"),
                mean_alloc_rbg=("alloc_rbg", "mean"),
                pct_slots_need_service=("need_service", lambda x: x.mean() * 100.0),
                pct_slots_served=("served", lambda x: x.mean() * 100.0),
                pct_backlog_not_served=("backlog_not_served", lambda x: x.mean() * 100.0),
            ).reset_index()
            per_ue.insert(0, "mode", mode)
            per_ue.insert(0, "bandwidth", bw_dir.name)
            per_ue.insert(0, "scenario", scenario)
            per_ue_rows.append(per_ue)

    return pd.DataFrame(summary_rows), pd.concat(per_ue_rows, ignore_index=True) if per_ue_rows else pd.DataFrame()


def consecutive_zero_runs(values: np.ndarray) -> list[int]:
    runs, current = [], 0
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


# ==========================================================
# 5) PLOTS
# ==========================================================

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {path}")


def mode_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(mode_order=df["mode"].map(MODE_ORDER)).sort_values("mode_order").drop(columns="mode_order")


def plot_tradeoff(tradeoff: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / "01_tradeoff_throughput_fairness"
    for (scenario, bw), sub in tradeoff.groupby(["scenario", "bandwidth"], sort=False):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        sub = mode_sort(sub)
        for _, row in sub.iterrows():
            m = row["mode"]
            ax.scatter(row["jain_rbg_slot_mean"], row["aggregate_throughput_mbps"], marker=MARKERS.get(m, "o"), s=130, color=COLORS.get(m), label=label(m))
            ax.annotate(label(m), (row["jain_rbg_slot_mean"], row["aggregate_throughput_mbps"]), textcoords="offset points", xytext=(6, 6), fontsize=TICK_SIZE)
        ax.set_xlabel("Mean slot-level Jain index over RBG allocation")
        ax.set_ylabel("Aggregate throughput (Mbps)")
        ax.set_title(f"Throughput-fairness trade-off — {scenario} — {bw}")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"tradeoff_{scenario}_{bw}.png")


def plot_flow_metric_bars(flow_summary: pd.DataFrame, out_root: Path) -> None:
    metrics = [
        ("aggregate_throughput_mbps", "Aggregate throughput (Mbps)", "02_qos_throughput"),
        ("p5_flow_throughput_mbps", "P5 flow throughput (Mbps)", "02_qos_throughput"),
        ("mean_delay_ms", "Mean delay (ms)", "03_qos_delay_jitter_loss"),
        ("mean_jitter_ms", "Mean jitter (ms)", "03_qos_delay_jitter_loss"),
        ("mean_loss_ratio", "Mean loss ratio", "03_qos_delay_jitter_loss"),
    ]
    for col, ylabel, folder in metrics:
        if col not in flow_summary.columns:
            continue
        out_dir = out_root / folder
        for (scenario, bw), sub in flow_summary.groupby(["scenario", "bandwidth"], sort=False):
            sub = mode_sort(sub)
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar([label(m) for m in sub["mode"]], sub[col], color=[COLORS.get(m) for m in sub["mode"]])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — {scenario} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            savefig(fig, out_dir / f"{col}_{scenario}_{bw}.png")


def plot_fairness_selected(slot_data: dict[tuple[str, str, str], dict[str, Any]], out_root: Path) -> None:
    # Gráfico 2: boxplot do Jain instantâneo.
    out_dir_2 = out_root / "04_fairness_jain_boxplot"
    # Gráfico 5: starvation instantâneo por UE.
    out_dir_5 = out_root / "05_starvation_per_ue"
    # Gráfico 6: starvation instantâneo por slot.
    out_dir_6 = out_root / "06_starvation_timeseries"
    # Gráficos 9 e 10: summary bars.
    out_dir_9_10 = out_root / "07_starvation_summary_bars"

    keys = sorted(set((s, bw) for (s, bw, _) in slot_data.keys()), key=lambda x: (x[0], int(x[1].replace("bw", ""))))
    for scenario, bw in keys:
        present = [m for m in MODES if (scenario, bw, m) in slot_data]
        if not present:
            continue
        all_rntis = slot_data[(scenario, bw, present[0])]["all_rntis"]

        # 2 - Boxplot Jain instantâneo.
        labels = [label(m) for m in present]
        values = [slot_data[(scenario, bw, m)]["jain"].dropna().values for m in present]
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        bp = ax.boxplot(values, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, m in zip(bp["boxes"], present):
            patch.set_facecolor(COLORS.get(m, "gray"))
            patch.set_alpha(0.65)
        ax.set_ylabel("Slot-level Jain index")
        ax.set_title(f"Boxplot do Jain instantâneo — {scenario} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        savefig(fig, out_dir_2 / f"2_boxplot_jain_instantaneo_{scenario}_{bw}.png")

        # 5 - Starvation por UE.
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        x = np.arange(len(all_rntis))
        width = 0.8 / max(1, len(present))
        for i, m in enumerate(present):
            y = slot_data[(scenario, bw, m)]["zero_fraction_per_ue"].reindex(all_rntis).values * 100.0
            offset = (i - (len(present) - 1) / 2) * width
            ax.bar(x + offset, y, width=width, label=label(m), color=COLORS.get(m))
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in all_rntis])
        ax.set_xlabel("UE (RNTI)")
        ax.set_ylabel("% de slots com alloc = 0")
        ax.set_title(f"Starvation instantâneo por UE — {scenario} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir_5 / f"5_starvation_instantaneo_por_ue_{scenario}_{bw}.png")

        # 6 - Starvation por slot.
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for m in present:
            y = slot_data[(scenario, bw, m)]["zero_fraction_per_slot"].reset_index(drop=True) * 100.0
            ax.plot(y.index, y.values, linewidth=LINE_WIDTH, label=label(m), color=COLORS.get(m))
        ax.set_xlabel("Slot index")
        ax.set_ylabel("% de UEs sem alocação no slot")
        ax.set_title(f"Starvation instantâneo por slot — {scenario} — {bw}")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir_6 / f"6_starvation_instantaneo_por_slot_{scenario}_{bw}.png")

        # 9 - Mean active UEs per slot.
        rows = []
        for m in present:
            d = slot_data[(scenario, bw, m)]
            rows.append({
                "mode": m,
                "mean_active_ues": float(d["active"].mean()),
                "mean_zero_ue_percent": float(d["zero_fraction_per_slot"].mean() * 100.0),
            })
        df = pd.DataFrame(rows)
        for idx, (col, ylabel) in enumerate([
            ("mean_active_ues", "Mean active UEs per slot"),
            ("mean_zero_ue_percent", "Mean % UEs without allocation"),
        ], start=9):
            df = mode_sort(df)
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar([label(m) for m in df["mode"]], df[col], color=[COLORS.get(m) for m in df["mode"]])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — {scenario} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            savefig(fig, out_dir_9_10 / f"{idx}_{col}_{scenario}_{bw}.png")


def plot_zero_runs_cdf(slot_data: dict[tuple[str, str, str], dict[str, Any]], out_root: Path) -> None:
    out_dir = out_root / "08_consecutive_no_service_cdf"
    keys = sorted(set((s, bw) for (s, bw, _) in slot_data.keys()), key=lambda x: (x[0], int(x[1].replace("bw", ""))))
    for scenario, bw in keys:
        present = [m for m in MODES if (scenario, bw, m) in slot_data]
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for m in present:
            mat = slot_data[(scenario, bw, m)]["mat"]
            runs: list[int] = []
            for col in mat.columns:
                runs.extend(consecutive_zero_runs((mat[col].values <= 0).astype(int)))
            if not runs:
                runs = [0]
            x = np.sort(np.asarray(runs, dtype=float))
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, linewidth=LINE_WIDTH, label=label(m), color=COLORS.get(m))
        ax.set_xlabel("Duração consecutiva sem alocação (slots)")
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF de períodos consecutivos sem serviço — {scenario} — {bw}")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"cdf_periodos_sem_servico_{scenario}_{bw}.png")


def plot_backlog(backlog_summary: pd.DataFrame, backlog_per_ue: pd.DataFrame, out_root: Path) -> None:
    # Métricas principais de backlog/starvation real.
    main_metrics = [
        ("pct_buf_gt0_alloc_eq0", "% casos com buf_req > 0 e alloc = 0", "09_backlog_starvation_real"),
        ("mean_served_ues_per_slot", "Média de UEs atendidos por slot", "10_backlog_service_capacity"),
        ("sum_buf_req_mean_per_slot", "Soma média de buf_req por slot", "11_backlog_load"),
    ]
    for col, ylabel, folder in main_metrics:
        if col not in backlog_summary.columns:
            continue
        out_dir = out_root / folder
        for (scenario, bw), sub in backlog_summary.groupby(["scenario", "bandwidth"], sort=False):
            sub = mode_sort(sub)
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar([label(m) for m in sub["mode"]], sub[col], color=[COLORS.get(m) for m in sub["mode"]])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — {scenario} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            savefig(fig, out_dir / f"{col}_{scenario}_{bw}.png")

    # Backlog vs atendidos.
    out_dir = out_root / "10_backlog_service_capacity"
    for (scenario, bw), sub in backlog_summary.groupby(["scenario", "bandwidth"], sort=False):
        sub = mode_sort(sub)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["mean_backlogged_ues_per_slot"], width, label="UEs com backlog")
        ax.bar(x + width / 2, sub["mean_served_ues_per_slot"], width, label="UEs atendidos")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Média por slot")
        ax.set_title(f"UEs com backlog vs UEs atendidos — {scenario} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"backlog_vs_atendidos_{scenario}_{bw}.png")

    # Jain do buffer vs Jain da alocação.
    out_dir = out_root / "12_backlog_allocation_alignment"
    for (scenario, bw), sub in backlog_summary.groupby(["scenario", "bandwidth"], sort=False):
        sub = mode_sort(sub)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["mean_jain_buf_req_per_slot"], width, label="Jain do buffer")
        ax.bar(x + width / 2, sub["mean_jain_alloc_per_slot"], width, label="Jain da alocação")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Jain médio por slot")
        ax.set_title(f"Jain do backlog vs Jain da alocação — {scenario} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"jain_buffer_vs_alocacao_{scenario}_{bw}.png")

    # Backlog sem atendimento por UE.
    if not backlog_per_ue.empty:
        out_dir = out_root / "09_backlog_starvation_real"
        for (scenario, bw), sub in backlog_per_ue.groupby(["scenario", "bandwidth"], sort=False):
            modes = [m for m in MODES if m in sub["mode"].unique()]
            pivot = sub.pivot(index="rnti", columns="mode", values="pct_backlog_not_served").reindex(columns=modes).sort_index()
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, len(pivot.columns))
            for i, m in enumerate(pivot.columns):
                offset = (i - (len(pivot.columns) - 1) / 2) * width
                ax.bar(x + offset, pivot[m].values, width=width, label=label(m), color=COLORS.get(m))
            ax.set_xticks(x)
            ax.set_xticklabels([str(r) for r in pivot.index])
            ax.set_xlabel("UE (RNTI)")
            ax.set_ylabel("% slots com buf_req > 0 e alloc = 0")
            ax.set_title(f"Backlog sem atendimento por UE — {scenario} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            ax.legend(frameon=True)
            savefig(fig, out_dir / f"backlog_sem_atendimento_por_ue_{scenario}_{bw}.png")


def plot_dynamic_profile_summary(flow_summary: pd.DataFrame, tradeoff: pd.DataFrame, backlog_summary: pd.DataFrame, out_root: Path) -> None:
    """Plots across dynamic phases to support the traffic-dynamism argument.

    These plots are not slot-continuous because the current ns-3 binary accepts a
    fixed --lambda per execution. They compare the ordered phases of the same
    dynamic traffic profile: low load, ramp-up, burst, recovery, second burst.
    """
    out_dir = out_root / "13_dynamic_flow_profile_phases"

    def prep(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df.empty or "profile" not in df.columns or value_col not in df.columns:
            return pd.DataFrame()
        sub = df[df["profile"] == "dynamic_flow_profile"].copy()
        if sub.empty:
            return sub
        sub["phase_order"] = pd.to_numeric(sub["phase_order"], errors="coerce")
        sub["lambda"] = pd.to_numeric(sub["lambda"], errors="coerce")
        return sub.sort_values(["bandwidth", "mode", "phase_order"])

    plots = [
        (flow_summary, "aggregate_throughput_mbps", "Aggregate throughput (Mbps)", "dynamic_aggregate_throughput"),
        (flow_summary, "p5_flow_throughput_mbps", "P5 flow throughput (Mbps)", "dynamic_p5_throughput"),
        (flow_summary, "mean_delay_ms", "Mean delay (ms)", "dynamic_mean_delay"),
        (tradeoff, "jain_rbg_slot_mean", "Mean slot-level Jain over RBG", "dynamic_jain_rbg"),
        (backlog_summary, "pct_buf_gt0_alloc_eq0", "% buf_req > 0 and alloc = 0", "dynamic_real_starvation"),
        (backlog_summary, "sum_buf_req_mean_per_slot", "Mean total buf_req per slot", "dynamic_backlog_load"),
    ]

    for df, value_col, ylabel, filename in plots:
        sub = prep(df, value_col)
        if sub.empty:
            continue
        for bw, bw_sub in sub.groupby("bandwidth", sort=False):
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            for mode in [m for m in MODES if m in bw_sub["mode"].unique()]:
                msub = bw_sub[bw_sub["mode"] == mode].sort_values("phase_order")
                x = msub["phase_order"].to_numpy(dtype=float)
                y = msub[value_col].to_numpy(dtype=float)
                ax.plot(x, y, marker=MARKERS.get(mode, "o"), linewidth=LINE_WIDTH, label=label(mode), color=COLORS.get(mode))
            phases = bw_sub.drop_duplicates("phase_order").sort_values("phase_order")
            xticklabels = [f"{r.phase}\nλ={int(r['lambda'])}" for _, r in phases.iterrows()]
            ax.set_xticks(phases["phase_order"].to_numpy(dtype=float))
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("Dynamic traffic phase")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Dynamic traffic profile — {ylabel} — {bw}")
            ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
            ax.legend(frameon=True)
            savefig(fig, out_dir / f"{filename}_{bw}.png")


def generate_all_outputs(run_dir: Path) -> None:
    setup_matplotlib()
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots_by_evidence"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = load_scenario_metadata(run_dir)

    print("\n[POST] Loading flow summaries...")
    df_all_flows = load_flow_summary(run_dir)
    if df_all_flows.empty:
        raise RuntimeError("Nenhum flow_summary encontrado.")
    flow_summary = attach_scenario_metadata(summarize_flows(df_all_flows), meta)
    flow_summary.to_csv(tables_dir / "flow_comparison_long.csv", index=False)

    print("[POST] Loading slot data...")
    slot_data = load_slot_data(run_dir)
    tradeoff = attach_scenario_metadata(summarize_tradeoff(flow_summary, slot_data), meta)
    tradeoff.to_csv(tables_dir / "tradeoff_points.csv", index=False)

    print("[POST] Computing backlog statistics...")
    backlog_summary, backlog_per_ue = summarize_backlog(run_dir)
    backlog_summary = attach_scenario_metadata(backlog_summary, meta)
    backlog_summary.to_csv(tables_dir / "backlog_summary.csv", index=False)
    if not backlog_per_ue.empty:
        backlog_per_ue.to_csv(tables_dir / "backlog_per_ue.csv", index=False)

    print("[POST] Generating plots...")
    plot_tradeoff(tradeoff, plots_dir)
    plot_flow_metric_bars(flow_summary, plots_dir)
    plot_fairness_selected(slot_data, plots_dir)
    plot_zero_runs_cdf(slot_data, plots_dir)
    plot_backlog(backlog_summary, backlog_per_ue, plots_dir)
    plot_dynamic_profile_summary(flow_summary, tradeoff, backlog_summary, plots_dir)

    print("\n[DONE] Tables:")
    for p in sorted(tables_dir.glob("*.csv")):
        print("  ", p)
    print("\n[DONE] Plots root:")
    print("  ", plots_dir)


# ==========================================================
# 6) CLI
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa experimentos dinâmicos IA-FMR no ns-3 e gera tabelas/gráficos.")
    parser.add_argument("--run-id", default=None, help="Nome manual do run. Se omitido, usa data/hora.")
    parser.add_argument("--skip-run", action="store_true", help="Não executa ns-3; apenas gera tabelas/gráficos de um run existente.")
    parser.add_argument("--existing-run-dir", default=None, help="Diretório de run existente para --skip-run.")
    parser.add_argument("--only-scenario", default=None, help="Executa apenas um cenário pelo nome.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"dynamic_qos_{timestamp()}"

    if args.skip_run:
        if not args.existing_run_dir:
            raise ValueError("Use --existing-run-dir junto com --skip-run")
        run_dir = Path(args.existing_run_dir).expanduser().resolve()
    else:
        run_dir = BASE_DIR / "compare_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        scenarios = SCENARIOS
        if args.only_scenario:
            scenarios = [s for s in SCENARIOS if s.name == args.only_scenario]
            if not scenarios:
                raise ValueError(f"Cenário não encontrado: {args.only_scenario}")
        print(f"[INFO] RUN_ID={run_id}")
        print(f"[INFO] RUN_DIR={run_dir}")
        print(f"[INFO] BASE_DIR={BASE_DIR}")
        print(f"[INFO] BIN={BIN}")
        print(f"[INFO] MODEL_DIR={MODEL_DIR}")
        run_all_experiments(run_dir, scenarios)

    generate_all_outputs(run_dir)


if __name__ == "__main__":
    main()
