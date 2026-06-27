#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script definitivo para execução e análise dos experimentos dinâmicos IA-FMR no ns-3.

Objetivos:
  1) executar RR, PF, MR e IA-FMR para múltiplas bandas, cenários e repetições;
  2) gerar tabelas completas de QoS, throughput, fairness, backlog e starvation real;
  3) gerar gráficos principais em português, organizados por evidência;
  4) gerar gráficos temporais a partir dos slot logs;
  5) permitir pós-processamento de runs existentes com --skip-run.

Observação:
  O IA-FMR é treinado offline. Este script executa apenas inferência no ns-3.
  O dinamismo considerado refere-se às fases de carga/tráfego, não à mobilidade dos UEs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import os
import signal
import subprocess
import sys
import time
import gc
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1) CONFIGURAÇÃO GERAL
# ==========================================================

BASE_DIR = Path(os.environ.get("FMR_BASE_DIR", "~/ns3-fmr")).expanduser().resolve()
BIN = Path(os.environ.get("FMR_BIN", str(BASE_DIR / "build/scratch/ns3.46-fmr-compara-qos-default"))).expanduser().resolve()
MODEL_DIR = Path(os.environ.get("FMR_MODEL_DIR", str(BASE_DIR / "scratch/fmr_ai/models"))).expanduser().resolve()
VENV_ACTIVATE = BASE_DIR / ".venv/bin/activate"

MODES = ["rr", "pf", "mr", "fmr_rl"]
CLASSIC_MODES = ["rr", "pf", "mr"]
LABELS = {"rr": "RR", "pf": "PF", "mr": "MR", "fmr_rl": "IA-FMR"}
MODE_ORDER = {m: i for i, m in enumerate(MODES)}
MARKERS = {"rr": "o", "pf": "s", "mr": "^", "fmr_rl": "D"}
COLORS = {
    "rr": "#4C78A8",
    "pf": "#F58518",
    "mr": "#54A24B",
    "fmr_rl": "#B279A2",
}

# Parâmetros visuais.
FIGSIZE_SINGLE = (8.8, 5.4)
FIGSIZE_WIDE = (12.0, 5.8)
FIGSIZE_TALL = (10.0, 7.0)
FIG_DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 10
LINE_WIDTH = 1.6
GRID_ALPHA = 0.35

# Parâmetros de execução.
FMR_CONTINUE_ON_ERROR = os.environ.get("FMR_CONTINUE_ON_ERROR", "1") not in {"0", "false", "False", "no"}
FMR_MAX_WORKERS_CLASSIC = int(os.environ.get("FMR_MAX_WORKERS_CLASSIC", "1"))
FMR_RUN_CLASSIC_PARALLEL = FMR_MAX_WORKERS_CLASSIC > 1
FMR_MAX_WORKERS_POST = int(os.environ.get("FMR_MAX_WORKERS_POST", "12"))
FMR_POST_CHUNKSIZE = int(os.environ.get("FMR_POST_CHUNKSIZE", "250000"))



# Sementes/repetições. Ex.: FMR_SEEDS="1 2 3 4 5"
def _parse_int_list(env_name: str, default: str) -> list[int]:
    raw = os.environ.get(env_name, default)
    out: list[int] = []
    for x in raw.replace(",", " ").split():
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out or [1]

SEEDS = _parse_int_list("FMR_SEEDS", os.environ.get("N_REPEATS", "1"))
if len(SEEDS) == 1 and "N_REPEATS" in os.environ and "FMR_SEEDS" not in os.environ:
    try:
        n = max(1, int(os.environ.get("N_REPEATS", "1")))
        SEEDS = list(range(1, n + 1))
    except Exception:
        SEEDS = [1]

# Parâmetros ns-3 comuns.
COMMON_NS3_ARGS = {
    "simTime": "5s",
    "udpAppStartTime": os.environ.get("FMR_UDP_APP_START_TIME", "400ms"),
    "warmupTime": os.environ.get("FMR_WARMUP_TIME", "0s"),
    "drainTime": os.environ.get("FMR_DRAIN_TIME", "0s"),
    "scenarioRadiusMin": "20",
    "scenarioRadiusMax": "150",
    "enableMobility": "1",
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
    "logging": "0",
    "positionMode": os.environ.get("FMR_POSITION_MODE", "fixed_profile"),
    "enableMobility": os.environ.get("FMR_ENABLE_MOBILITY", "0"),
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

TAU_BY_BW = {bw: 0.65 for bw in range(10, 101, 5)}

# Parâmetros para estimar vazão temporal a partir do slot_log.
SCS_KHZ = float(os.environ.get("FMR_SCS_KHZ", "30"))
CODE_RATE = float(os.environ.get("FMR_CODE_RATE", "0.85"))
TEMPORAL_ROLLING_WINDOW = int(os.environ.get("FMR_TEMPORAL_ROLLING_WINDOW", "25"))

MCS_EFFICIENCY = {
    0: 0.2344, 1: 0.3066, 2: 0.3770, 3: 0.4902, 4: 0.6016, 5: 0.7402,
    6: 0.8770, 7: 1.0273, 8: 1.1758, 9: 1.3262, 10: 1.3281, 11: 1.4766,
    12: 1.6953, 13: 1.9141, 14: 2.1602, 15: 2.4063, 16: 2.5703, 17: 2.7305,
    18: 3.0293, 19: 3.3223, 20: 3.6094, 21: 3.9023, 22: 4.2129, 23: 4.5234,
    24: 4.8164, 25: 5.1152, 26: 5.3320, 27: 5.5547,
}

# Correção metodológica da vazão temporal estimada a partir do slot_log.
# As métricas oficiais de vazão continuam vindo do flow_summary.
TEMPORAL_THROUGHPUT_SCALE_BY_MODE = {
    "rr": float(os.environ.get("FMR_TEMP_SCALE_RR", str(1.0 / 13.0))),
    "pf": float(os.environ.get("FMR_TEMP_SCALE_PF", str(1.0 / 13.0))),
    "mr": float(os.environ.get("FMR_TEMP_SCALE_MR", str(1.0 / 13.0))),
    "fmr_rl": float(os.environ.get("FMR_TEMP_SCALE_FMR_RL", "1.0")),
}

@dataclass
class Scenario:
    name: str
    purpose: str
    bandwidths_mhz: list[int]
    lambda_value: int
    num_dl_flows_per_ue: int
    sim_time: str = "5s"
    udp_packet_size: int = 3000
    extra_ns3_args: dict[str, str] = field(default_factory=dict)


def _parse_bw_list() -> list[int]:
    raw = os.environ.get("BW_LIST", "100")
    out = []
    for x in raw.replace(",", " ").split():
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out or [10, 20, 30, 40, 50]


def _parse_seconds(value: str, default: float = 30.0) -> float:
    value = str(value or "").strip().lower()
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000.0
        if value.endswith("s"):
            return float(value[:-1])
        return float(value)
    except Exception:
        return default


def _fmt_seconds(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}s"
    return f"{seconds:.3f}s"

REQUESTED_BWS = _parse_bw_list()
TOTAL_SIM_SECONDS = _parse_seconds(os.environ.get("SIM_TIME", "30s"), default=30.0)
N_DYNAMIC_PHASES = 5
PHASE_SIM_TIME = _fmt_seconds(max(2.0, TOTAL_SIM_SECONDS / N_DYNAMIC_PHASES))
SAFE_MAX_LAMBDA = int(os.environ.get("FMR_SAFE_MAX_LAMBDA", "850"))


def _safe_lambda(value: int) -> int:
    return min(int(value), SAFE_MAX_LAMBDA)

PHASE_NAMES = os.environ.get("FMR_PHASE_NAMES", "low_load,ramp_up,safe_burst,recovery,second_burst")
DEFAULT_PHASE_DURATIONS = ",".join([PHASE_SIM_TIME] * N_DYNAMIC_PHASES)
PHASE_DURATIONS = os.environ.get("FMR_PHASE_DURATIONS", DEFAULT_PHASE_DURATIONS)
# Para experimentos de alta carga, defina explicitamente:
# export FMR_PHASE_LAMBDAS="200,350,550,250,650"
PHASE_LAMBDAS = os.environ.get("FMR_PHASE_LAMBDAS", "5,10,15,8,20")
PHASE_FLOWS = os.environ.get("FMR_PHASE_FLOWS", "1,1,1,1,1")
TDD_PATTERN = os.environ.get("FMR_TDD_PATTERN", "DL|DL|DL|DL|UL|DL|DL|DL|DL|UL|")

SCENARIO_NAME = os.environ.get("FMR_SCENARIO_NAME", f"dynamic_continuous_{int(round(TOTAL_SIM_SECONDS))}s")

SCENARIOS: list[Scenario] = [
    Scenario(
        SCENARIO_NAME,
        "Cenário contínuo com cinco fases de tráfego na mesma simulação.",
        REQUESTED_BWS,
        _safe_lambda(200),
        4,
        _fmt_seconds(TOTAL_SIM_SECONDS),
        extra_ns3_args={
            "dynamicTraffic": "1",
            "phaseDurations": PHASE_DURATIONS,
            "phaseLambdas": PHASE_LAMBDAS,
            "tddPattern": TDD_PATTERN,
        },
    ),
]

SCENARIO_LABELS = {
    "dynamic_continuous_30s": "Dinâmica contínua",
}

# ==========================================================
# 2) UTILITÁRIOS
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
        "axes.unicode_minus": False,
    })


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def bw_to_hz(bw_mhz: int) -> int:
    return int(bw_mhz * 1_000_000)


def tau_for_bw(bw_mhz: int) -> float:
    return float(TAU_BY_BW.get(bw_mhz, 0.65))


def label(mode: str) -> str:
    return LABELS.get(mode, mode)


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def ensure_executable(path: Path) -> None:
    if not path.exists() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Binário não encontrado ou não executável: {path}\nRode: ./ns3 build")


def cleanup_ai() -> None:
    subprocess.run("pkill -f 'scratch/fmr_ai/agent.py' || true", shell=True)
    for pattern in ["/dev/shm/ns3ai_fmr*", "/dev/shm/fmr_cpp2py*", "/dev/shm/fmr_py2cpp*", "/dev/shm/fmr_lock*"]:
        subprocess.run(f"rm -f {pattern} || true", shell=True)


def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[CMD]", " ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return int(proc.wait())


def build_ns3_args(base: dict[str, Any]) -> list[str]:
    return [f"--{k}={v}" for k, v in base.items()]


def seed_dir_name(seed: int) -> str:
    return f"seed_{seed:03d}"


def get_mode_dir(run_dir: Path, seed: int, scenario: Scenario | str, bw_mhz: int, mode: str) -> Path:
    scenario_name = scenario.name if isinstance(scenario, Scenario) else str(scenario)
    if len(SEEDS) <= 1 and not os.environ.get("FMR_FORCE_SEED_DIR", "0") == "1":
        # Mantém compatibilidade visual com o layout antigo quando há uma única seed.
        return run_dir / scenario_name / f"bw{bw_mhz}" / mode
    return run_dir / seed_dir_name(seed) / scenario_name / f"bw{bw_mhz}" / mode


def is_mode_complete(mode_dir: Path, mode: str) -> bool:
    """Considera uma simulação reaproveitável quando os CSVs científicos existem."""
    required = [
        mode_dir / f"flow_summary_{mode}.csv",
        mode_dir / f"slot_log_{mode}.csv",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def model_exists_for_bw(bw_mhz: int) -> bool:
    return (MODEL_DIR / f"model_{bw_mhz}.zip").exists()

# ==========================================================
# 3) EXECUÇÃO
# ==========================================================

def base_ns3_args(run_dir: Path, scenario: Scenario, bw_mhz: int, mode: str, seed: int, mode_dir: Path) -> dict[str, Any]:
    ns3_args = dict(COMMON_NS3_ARGS)
    ns3_args.update({
        "schedulerMode": mode,
        "EnableNs3Ai": "0",
        "FmrTau": tau_for_bw(bw_mhz),
        "simTime": scenario.sim_time,
        "bandwidth": bw_to_hz(bw_mhz),
        "lambda": scenario.lambda_value,
        "udpPacketSize": scenario.udp_packet_size,
        "numDlFlowsPerUe": scenario.num_dl_flows_per_ue,
        "rngRun": seed,
        "SlotCsvPath": mode_dir / f"slot_log_{mode}.csv",
        "UeSnapshotCsvPath": mode_dir / f"ue_snapshot_{mode}.csv",
        "FlowSummaryCsvPath": mode_dir / f"flow_summary_{mode}.csv",
        "simTag": "summary.txt",
        "outputDir": mode_dir,
    })
    ns3_args.update(scenario.extra_ns3_args)
    return ns3_args


def run_mode(run_dir: Path, scenario: Scenario, bw_mhz: int, mode: str, seed: int, skip_existing: bool = False, force: bool = False) -> None:
    mode_dir = get_mode_dir(run_dir, seed, scenario, bw_mhz, mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and not force and is_mode_complete(mode_dir, mode):
        print(f"[SKIP] seed={seed} | scenario={scenario.name} | bw={bw_mhz}MHz | mode={mode} já concluído")
        return
    print("\n" + "=" * 80)
    print(f"[RUN] seed={seed} | scenario={scenario.name} | bw={bw_mhz}MHz | mode={mode}")
    print("=" * 80)
    ns3_args = base_ns3_args(run_dir, scenario, bw_mhz, mode, seed, mode_dir)
    cmd = [str(BIN)] + build_ns3_args(ns3_args)
    rc = run_cmd(cmd, mode_dir / "ns3.log", cwd=BASE_DIR)
    if rc != 0:
        raise RuntimeError(f"Falha no ns-3: seed={seed}, scenario={scenario.name}, bw={bw_mhz}, mode={mode}, rc={rc}")


def start_agent(run_dir: Path, scenario: Scenario, bw_mhz: int, mode_dir: Path, seed: int) -> subprocess.Popen:
    model_path = MODEL_DIR / f"model_{bw_mhz}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    unique = f"{run_dir.name}_{seed}_{scenario.name}_bw{bw_mhz}"
    segment = f"ns3ai_fmr_{unique}"
    cpp2py = f"fmr_cpp2py_{unique}"
    py2cpp = f"fmr_py2cpp_{unique}"
    lock = f"fmr_lock_{unique}"

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
        f" --tau {tau_for_bw(bw_mhz)}"
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
    threading.Thread(target=_drain_stdout, args=(proc, log), daemon=True).start()
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


def run_fmr_rl(run_dir: Path, scenario: Scenario, bw_mhz: int, seed: int, skip_existing: bool = False, force: bool = False) -> None:
    mode = "fmr_rl"
    mode_dir = get_mode_dir(run_dir, seed, scenario, bw_mhz, mode)
    mode_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and not force and is_mode_complete(mode_dir, mode):
        print(f"[SKIP] seed={seed} | scenario={scenario.name} | bw={bw_mhz}MHz | mode=IA-FMR já concluído")
        return
    if not model_exists_for_bw(bw_mhz):
        print(f"[SKIP] Modelo IA-FMR ausente para {bw_mhz} MHz: {MODEL_DIR / f'model_{bw_mhz}.zip'}")
        return
    print("\n" + "=" * 80)
    print(f"[RUN] seed={seed} | scenario={scenario.name} | bw={bw_mhz}MHz | mode=IA-FMR")
    print("=" * 80)
    agent_proc = start_agent(run_dir, scenario, bw_mhz, mode_dir, seed)
    names = agent_proc._fmr_ai_names  # type: ignore[attr-defined]
    ns3_args = base_ns3_args(run_dir, scenario, bw_mhz, mode, seed, mode_dir)
    ns3_args.update({
        "EnableNs3Ai": "1",
        "AiCppIsCreator": "1",
        "AiVerbose": "1",
        "AiShmSize": AGENT_ARGS["shm"],
        "AiSegmentName": names["segment"],
        "AiCpp2PyName": names["cpp2py"],
        "AiPy2CppName": names["py2cpp"],
        "AiLockableName": names["lock"],
    })
    try:
        cmd = [str(BIN)] + build_ns3_args(ns3_args)
        rc = run_cmd(cmd, mode_dir / "ns3.log", cwd=BASE_DIR)
        if rc != 0:
            raise RuntimeError(f"Falha no ns-3 IA-FMR: seed={seed}, scenario={scenario.name}, bw={bw_mhz}, rc={rc}")
    finally:
        stop_agent(agent_proc)


def _classic_task(args: tuple[Path, Scenario, int, str, int, bool, bool]) -> dict[str, Any] | None:
    run_dir, scenario, bw, mode, seed, skip_existing, force = args
    try:
        run_mode(run_dir, scenario, bw, mode, seed, skip_existing=skip_existing, force=force)
        return None
    except Exception as e:
        return {"seed": seed, "scenario": scenario.name, "bandwidth_mhz": bw, "mode": mode, "error": str(e)}


def run_all_experiments(run_dir: Path, scenarios: list[Scenario], modes_to_run: list[str] | None = None, skip_existing: bool = False, force: bool = False) -> None:
    ensure_executable(BIN)
    cleanup_ai()

    meta_dir = run_dir / "00_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = []
    for seed in SEEDS:
        for s in scenarios:
            for bw in s.bandwidths_mhz:
                matrix_rows.append({
                    "seed": seed,
                    "scenario": s.name,
                    "purpose": s.purpose,
                    "bandwidth_mhz": bw,
                    "lambda": s.lambda_value,
                    "num_dl_flows_per_ue": s.num_dl_flows_per_ue,
                    "sim_time": s.sim_time,
                    "udp_packet_size": s.udp_packet_size,
                    "extra_ns3_args": str(s.extra_ns3_args),
                })
    pd.DataFrame(matrix_rows).to_csv(meta_dir / "scenario_matrix.csv", index=False)

    failures: list[dict[str, Any]] = []
    modes_to_run = modes_to_run or list(MODES)
    classic_modes_to_run = [m for m in CLASSIC_MODES if m in modes_to_run]
    classic_tasks: list[tuple[Path, Scenario, int, str, int, bool, bool]] = []
    for seed in SEEDS:
        for scenario in scenarios:
            for bw in scenario.bandwidths_mhz:
                for mode in classic_modes_to_run:
                    classic_tasks.append((run_dir, scenario, bw, mode, seed, skip_existing, force))

    if FMR_RUN_CLASSIC_PARALLEL and classic_tasks:
        print(f"[INFO] Executando RR/PF/MR em paralelo com {FMR_MAX_WORKERS_CLASSIC} workers")
        with futures.ThreadPoolExecutor(max_workers=FMR_MAX_WORKERS_CLASSIC) as ex:
            for result in ex.map(_classic_task, classic_tasks):
                if result:
                    failures.append(result)
                    print(f"[ERROR] {result}")
                    if not FMR_CONTINUE_ON_ERROR:
                        pd.DataFrame(failures).to_csv(meta_dir / "failed_runs.csv", index=False)
                        raise RuntimeError(result["error"])
    else:
        for task in classic_tasks:
            result = _classic_task(task)
            if result:
                failures.append(result)
                print(f"[ERROR] {result}")
                if not FMR_CONTINUE_ON_ERROR:
                    pd.DataFrame(failures).to_csv(meta_dir / "failed_runs.csv", index=False)
                    raise RuntimeError(result["error"])

    # IA-FMR sequencial por segurança com ns3-ai/shared memory.
    if "fmr_rl" in modes_to_run:
        for seed in SEEDS:
            for scenario in scenarios:
                for bw in scenario.bandwidths_mhz:
                    try:
                        run_fmr_rl(run_dir, scenario, bw, seed, skip_existing=skip_existing, force=force)
                    except Exception as e:
                        failure = {"seed": seed, "scenario": scenario.name, "bandwidth_mhz": bw, "mode": "fmr_rl", "error": str(e)}
                        failures.append(failure)
                        print(f"[ERROR] {failure}")
                        if not FMR_CONTINUE_ON_ERROR:
                            pd.DataFrame(failures).to_csv(meta_dir / "failed_runs.csv", index=False)
                            raise

    if failures:
        pd.DataFrame(failures).to_csv(meta_dir / "failed_runs.csv", index=False)
        print(f"[WARN] Algumas execuções falharam. Veja: {meta_dir / 'failed_runs.csv'}")

# ==========================================================
# 4) LEITURA E MÉTRICAS
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


def parse_bw_name(name: str) -> int:
    return int(str(name).replace("bw", ""))


def iter_mode_dirs(run_dir: Path):
    # Layout novo: run/seed_001/scenario/bwXX/mode
    seed_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    if seed_dirs:
        for seed_dir in sorted(seed_dirs):
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except Exception:
                seed = 1
            for scenario_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                if scenario_dir.name.startswith(("00_", "plots", "tables")):
                    continue
                for bw_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("bw")], key=lambda p: parse_bw_name(p.name)):
                    for mode in MODES:
                        mode_dir = bw_dir / mode
                        if mode_dir.is_dir():
                            yield seed, scenario_dir.name, bw_dir.name, mode, mode_dir
    else:
        # Layout antigo: run/scenario/bwXX/mode
        for scenario_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if scenario_dir.name.startswith(("00_", "plots", "tables")):
                continue
            for bw_dir in sorted([p for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("bw")], key=lambda p: parse_bw_name(p.name)):
                for mode in MODES:
                    mode_dir = bw_dir / mode
                    if mode_dir.is_dir():
                        yield 1, scenario_dir.name, bw_dir.name, mode, mode_dir


def load_flow_summary(run_dir: Path) -> pd.DataFrame:
    dfs = []
    for seed, scenario, bandwidth, mode, mode_dir in iter_mode_dirs(run_dir):
        path = mode_dir / f"flow_summary_{mode}.csv"
        if not path.exists():
            print(f"[WARN] missing flow summary: {path}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        df["scenario"] = scenario
        df["bandwidth"] = bandwidth
        df["bandwidth_mhz"] = parse_bw_name(bandwidth)
        df["mode"] = mode
        df["run_id"] = run_dir.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_flows(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["run_id", "seed", "scenario", "bandwidth", "bandwidth_mhz", "mode"]
    for key, df in df_all.groupby(group_cols, sort=False):
        run_id, seed, scenario, bandwidth, bandwidth_mhz, mode = key
        thr = df["throughput_mbps"]
        delay = df["mean_delay_ms"] if "mean_delay_ms" in df else pd.Series(dtype=float)
        jitter = df["mean_jitter_ms"] if "mean_jitter_ms" in df else pd.Series(dtype=float)
        loss = df["loss_ratio"] if "loss_ratio" in df else pd.Series(dtype=float)
        rows.append({
            "run_id": run_id,
            "seed": int(seed),
            "scenario": scenario,
            "bandwidth": bandwidth,
            "bandwidth_mhz": int(bandwidth_mhz),
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
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mode_order"] = out["mode"].map(MODE_ORDER)
        out = out.sort_values(["seed", "scenario", "bandwidth_mhz", "mode_order"]).drop(columns="mode_order").reset_index(drop=True)
    return out


def find_mode_dir(run_dir: Path, seed: int, scenario: str, bandwidth: str, mode: str) -> Path | None:
    candidates = [
        run_dir / seed_dir_name(seed) / scenario / bandwidth / mode,
        run_dir / scenario / bandwidth / mode,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def infer_total_rbg_from_bw(run_dir: Path, seed: int, scenario: str, bandwidth: str) -> float:
    # Prefere FMR, mas aceita qualquer modo disponível.
    for mode in ["fmr_rl", "rr", "pf", "mr"]:
        mode_dir = find_mode_dir(run_dir, seed, scenario, bandwidth, mode)
        if not mode_dir:
            continue
        path = mode_dir / f"slot_log_{mode}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "alloc_rbg" in df.columns:
                g = df.groupby(["time_s", "beam_id"])["alloc_rbg"].sum()
                if not g.empty:
                    return float(round(g.median()))
    return np.nan


def get_all_rntis_from_bw(run_dir: Path, seed: int, scenario: str, bandwidth: str) -> list[int]:
    rntis = set()
    for mode in MODES:
        mode_dir = find_mode_dir(run_dir, seed, scenario, bandwidth, mode)
        if not mode_dir:
            continue
        path = mode_dir / f"slot_log_{mode}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, usecols=["rnti"])
                rntis.update(int(x) for x in df["rnti"].dropna().unique())
            except Exception:
                pass
    return sorted(rntis)


def load_slot_log_normalized(path: Path, total_rbg: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"time_s", "beam_id", "rnti", "alloc_rbg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    slot_sum = df.groupby(["time_s", "beam_id"])["alloc_rbg"].transform("sum").astype(float)
    df["alloc_norm"] = np.where(slot_sum > 0, df["alloc_rbg"].astype(float) * float(total_rbg) / slot_sum, 0.0)
    if "dl_mcs" in df.columns:
        df["se"] = df["dl_mcs"].map(MCS_EFFICIENCY).fillna(0.0).astype(float)
        # Mbps estimado pela alocação instantânea: SE * PRB bandwidth * code_rate.
        df["estimated_mbps"] = df["se"] * df["alloc_rbg"].astype(float) * (12.0 * SCS_KHZ * 1000.0) * CODE_RATE / 1e6
    else:
        df["estimated_mbps"] = np.nan
    return df


def slot_matrix(df: pd.DataFrame, all_rntis: list[int], value_col: str = "alloc_norm") -> pd.DataFrame:
    rows = []
    index_vals = []
    for (time_s, beam_id), g in df.groupby(["time_s", "beam_id"], sort=False):
        alloc = g.groupby("rnti")[value_col].sum().reindex(all_rntis, fill_value=0.0)
        rows.append(alloc)
        index_vals.append((time_s, beam_id))
    if not rows:
        return pd.DataFrame(columns=all_rntis)
    mat = pd.DataFrame(rows)
    mat.index = pd.MultiIndex.from_tuples(index_vals, names=["time_s", "beam_id"])
    return mat


def load_slot_data(run_dir: Path) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    data = {}
    bw_keys = sorted(set((seed, scenario, bandwidth) for seed, scenario, bandwidth, _, _ in iter_mode_dirs(run_dir)))
    for seed, scenario, bandwidth in bw_keys:
        total_rbg = infer_total_rbg_from_bw(run_dir, seed, scenario, bandwidth)
        all_rntis = get_all_rntis_from_bw(run_dir, seed, scenario, bandwidth)
        if not all_rntis or np.isnan(total_rbg):
            continue
        for mode in MODES:
            mode_dir = find_mode_dir(run_dir, seed, scenario, bandwidth, mode)
            if not mode_dir:
                continue
            path = mode_dir / f"slot_log_{mode}.csv"
            if not path.exists():
                continue
            df = load_slot_log_normalized(path, total_rbg)
            mat = slot_matrix(df, all_rntis, "alloc_norm")
            data[(seed, scenario, bandwidth, mode)] = {
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


def summarize_tradeoff(flow_summary: pd.DataFrame, slot_data: dict[tuple[int, str, str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, r in flow_summary.iterrows():
        key = (int(r["seed"]), r["scenario"], r["bandwidth"], r["mode"])
        d = slot_data.get(key)
        if d is None:
            continue
        rows.append({
            "run_id": r["run_id"],
            "seed": int(r["seed"]),
            "scenario": r["scenario"],
            "bandwidth": r["bandwidth"],
            "bandwidth_mhz": int(r["bandwidth_mhz"]),
            "mode": r["mode"],
            "aggregate_throughput_mbps": r["aggregate_throughput_mbps"],
            "mean_flow_throughput_mbps": r["mean_flow_throughput_mbps"],
            "p5_flow_throughput_mbps": r["p5_flow_throughput_mbps"],
            "jain_rbg_slot_mean": float(d["jain"].mean()),
            "mean_active_ues_per_slot": float(d["active"].mean()),
            "mean_zero_ue_percent": float(d["zero_fraction_per_slot"].mean() * 100.0),
        })
    return pd.DataFrame(rows)


def _process_one_slot_log(task: tuple[int, str, str, str, str]) -> dict[str, Any] | None:
    seed, scenario, bandwidth, mode, path_str = task
    path = Path(path_str)

    try:
        usecols = ["time_s", "beam_id", "rnti", "buf_req", "alloc_rbg", "dl_mcs"]
        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)

        required = {"time_s", "beam_id", "rnti", "buf_req", "alloc_rbg"}
        if not required.issubset(df.columns):
            print(f"[WARN] colunas ausentes em {path}")
            return None

        df["buf_req"] = pd.to_numeric(df["buf_req"], errors="coerce").fillna(0.0)
        df["alloc_rbg"] = pd.to_numeric(df["alloc_rbg"], errors="coerce").fillna(0.0)
        df["rnti"] = pd.to_numeric(df["rnti"], errors="coerce").fillna(-1).astype(int)

        if "dl_mcs" in df.columns:
            df["dl_mcs"] = pd.to_numeric(df["dl_mcs"], errors="coerce").fillna(0).astype(int)
            df["se"] = df["dl_mcs"].map(MCS_EFFICIENCY).fillna(0.0)
            df["vazao_estimada_mbps"] = (
                df["se"]
                * df["alloc_rbg"]
                * (12.0 * SCS_KHZ * 1000.0)
                * CODE_RATE
                / 1e6
            )
        else:
            df["vazao_estimada_mbps"] = 0.0

        slot_cols = ["time_s", "beam_id"]

        slot_rnti = (
            df.groupby(slot_cols + ["rnti"], sort=False)
            .agg(
                buf_req=("buf_req", "max"),
                alloc_rbg=("alloc_rbg", "sum"),
                vazao_estimada_mbps=("vazao_estimada_mbps", "sum"),
            )
            .reset_index()
        )

        all_rntis = sorted(slot_rnti["rnti"].unique())
        n_rntis = max(1, len(all_rntis))

        slot_rnti["need_service"] = slot_rnti["buf_req"] > 0
        slot_rnti["served"] = slot_rnti["alloc_rbg"] > 0
        slot_rnti["backlog_not_served"] = slot_rnti["need_service"] & (~slot_rnti["served"])

        per_slot = (
            slot_rnti.groupby(slot_cols, sort=False)
            .agg(
                backlog_total_bytes=("buf_req", "sum"),
                ues_com_backlog=("need_service", "sum"),
                ues_atendidos=("served", "sum"),
                starvation_real_pct=("backlog_not_served", lambda x: float(x.mean() * 100.0)),
                vazao_estimada_mbps=("vazao_estimada_mbps", "sum"),
            )
            .reset_index()
        )

        alloc_mat = (
            slot_rnti.pivot_table(
                index=slot_cols,
                columns="rnti",
                values="alloc_rbg",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(columns=all_rntis, fill_value=0.0)
        )

        buf_mat = (
            slot_rnti.pivot_table(
                index=slot_cols,
                columns="rnti",
                values="buf_req",
                aggfunc="max",
                fill_value=0.0,
            )
            .reindex(columns=all_rntis, fill_value=0.0)
        )

        jain_alloc = alloc_mat.apply(jain_index, axis=1)
        jain_buf = buf_mat.apply(jain_index, axis=1)

        active_ues = (alloc_mat > 0).sum(axis=1)
        zero_fraction_per_slot = (alloc_mat <= 0).mean(axis=1)
        zero_fraction_per_ue = (alloc_mat <= 0).mean(axis=0)

        temporal_backlog = per_slot.copy()
        temporal_backlog["seed"] = seed
        temporal_backlog["scenario"] = scenario
        temporal_backlog["bandwidth"] = bandwidth
        temporal_backlog["bandwidth_mhz"] = parse_bw_name(bandwidth)
        temporal_backlog["mode"] = mode
        temporal_backlog["jain_backlog"] = jain_buf.values
        temporal_backlog["jain_alocacao"] = jain_alloc.values

        temporal_thr = per_slot[["time_s", "beam_id", "vazao_estimada_mbps"]].copy()
        temporal_thr["seed"] = seed
        temporal_thr["scenario"] = scenario
        temporal_thr["bandwidth"] = bandwidth
        temporal_thr["bandwidth_mhz"] = parse_bw_name(bandwidth)
        temporal_thr["mode"] = mode

        temporal_ue_thr = slot_rnti[["time_s", "beam_id", "rnti", "vazao_estimada_mbps"]].copy()
        temporal_ue_thr["seed"] = seed
        temporal_ue_thr["scenario"] = scenario
        temporal_ue_thr["bandwidth"] = bandwidth
        temporal_ue_thr["bandwidth_mhz"] = parse_bw_name(bandwidth)
        temporal_ue_thr["mode"] = mode

        per_ue = (
            slot_rnti.groupby("rnti")
            .agg(
                mean_buf_req=("buf_req", "mean"),
                max_buf_req=("buf_req", "max"),
                mean_alloc_rbg=("alloc_rbg", "mean"),
                pct_slots_need_service=("need_service", lambda x: float(x.mean() * 100.0)),
                pct_slots_served=("served", lambda x: float(x.mean() * 100.0)),
                pct_backlog_not_served=("backlog_not_served", lambda x: float(x.mean() * 100.0)),
            )
            .reset_index()
        )
        per_ue.insert(0, "mode", mode)
        per_ue.insert(0, "bandwidth_mhz", parse_bw_name(bandwidth))
        per_ue.insert(0, "bandwidth", bandwidth)
        per_ue.insert(0, "scenario", scenario)
        per_ue.insert(0, "seed", seed)

        backlog_summary = pd.DataFrame([{
            "seed": seed,
            "scenario": scenario,
            "bandwidth": bandwidth,
            "bandwidth_mhz": parse_bw_name(bandwidth),
            "mode": mode,
            "n_slots": int(per_slot.shape[0]),
            "n_rntis": int(n_rntis),
            "mean_buf_req": float(slot_rnti["buf_req"].mean()),
            "median_buf_req": float(slot_rnti["buf_req"].median()),
            "max_buf_req": float(slot_rnti["buf_req"].max()),
            "sum_buf_req_mean_per_slot": float(per_slot["backlog_total_bytes"].mean()),
            "sum_buf_req_max_per_slot": float(per_slot["backlog_total_bytes"].max()),
            "mean_backlogged_ues_per_slot": float(per_slot["ues_com_backlog"].mean()),
            "mean_served_ues_per_slot": float(per_slot["ues_atendidos"].mean()),
            "pct_buf_gt0_alloc_eq0": float(slot_rnti["backlog_not_served"].mean() * 100.0),
            "pct_buf_eq0_alloc_eq0": float(((~slot_rnti["need_service"]) & (~slot_rnti["served"])).mean() * 100.0),
            "pct_buf_gt0_alloc_gt0": float((slot_rnti["need_service"] & slot_rnti["served"]).mean() * 100.0),
            "mean_jain_buf_req_per_slot": float(jain_buf.mean()),
            "mean_jain_alloc_per_slot": float(jain_alloc.mean()),
        }])

        slot_summary = pd.DataFrame([{
            "seed": seed,
            "scenario": scenario,
            "bandwidth": bandwidth,
            "bandwidth_mhz": parse_bw_name(bandwidth),
            "mode": mode,
            "jain_rbg_slot_mean": float(jain_alloc.mean()),
            "mean_active_ues_per_slot": float(active_ues.mean()),
            "mean_zero_ue_percent": float(zero_fraction_per_slot.mean() * 100.0),
        }])

        slot_light = {
            "jain": jain_alloc,
            "active": active_ues,
            "zero_fraction_per_slot": zero_fraction_per_slot,
            "zero_fraction_per_ue": zero_fraction_per_ue,
            "all_rntis": all_rntis,
        }

        del df, slot_rnti, alloc_mat, buf_mat
        gc.collect()

        return {
            "key": (seed, scenario, bandwidth, mode),
            "slot_summary": slot_summary,
            "backlog_summary": backlog_summary,
            "backlog_per_ue": per_ue,
            "temporal_backlog": temporal_backlog,
            "temporal_thr": temporal_thr,
            "temporal_ue_thr": temporal_ue_thr,
            "slot_light": slot_light,
        }

    except Exception as e:
        print(f"[ERROR] falha processando {path}: {e}")
        return None


def process_slot_logs_parallel(run_dir: Path):
    tasks = []

    for seed, scenario, bandwidth, mode, mode_dir in iter_mode_dirs(run_dir):
        path = mode_dir / f"slot_log_{mode}.csv"
        if path.exists():
            tasks.append((seed, scenario, bandwidth, mode, str(path)))

    print(f"[POST] Processando {len(tasks)} slot logs com {FMR_MAX_WORKERS_POST} workers...")

    slot_data = {}
    slot_summary_rows = []
    backlog_summary_rows = []
    backlog_per_ue_rows = []
    temporal_backlog_rows = []
    temporal_thr_rows = []
    temporal_ue_thr_rows = []

    with futures.ProcessPoolExecutor(max_workers=FMR_MAX_WORKERS_POST) as ex:
        futs = [ex.submit(_process_one_slot_log, t) for t in tasks]

        done = 0
        for fut in futures.as_completed(futs):
            done += 1
            result = fut.result()

            if result is None:
                continue

            key = result["key"]
            slot_data[key] = result["slot_light"]

            slot_summary_rows.append(result["slot_summary"])
            backlog_summary_rows.append(result["backlog_summary"])
            backlog_per_ue_rows.append(result["backlog_per_ue"])
            temporal_backlog_rows.append(result["temporal_backlog"])
            temporal_thr_rows.append(result["temporal_thr"])
            temporal_ue_thr_rows.append(result["temporal_ue_thr"])

            if done % 5 == 0 or done == len(futs):
                print(f"[POST] slot logs processados: {done}/{len(futs)}")

    slot_summary = pd.concat(slot_summary_rows, ignore_index=True) if slot_summary_rows else pd.DataFrame()
    backlog_summary = pd.concat(backlog_summary_rows, ignore_index=True) if backlog_summary_rows else pd.DataFrame()
    backlog_per_ue = pd.concat(backlog_per_ue_rows, ignore_index=True) if backlog_per_ue_rows else pd.DataFrame()
    temporal_backlog = pd.concat(temporal_backlog_rows, ignore_index=True) if temporal_backlog_rows else pd.DataFrame()
    temporal_thr = pd.concat(temporal_thr_rows, ignore_index=True) if temporal_thr_rows else pd.DataFrame()
    temporal_ue_thr = pd.concat(temporal_ue_thr_rows, ignore_index=True) if temporal_ue_thr_rows else pd.DataFrame()

    gc.collect()

    return slot_data, slot_summary, backlog_summary, backlog_per_ue, temporal_backlog, temporal_thr, temporal_ue_thr


def summarize_backlog(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, backlog_summary, backlog_per_ue, temporal_backlog, _, _ = process_slot_logs_parallel(run_dir)
    return backlog_summary, backlog_per_ue, temporal_backlog


def summarize_temporal_throughput(slot_data: dict[tuple[int, str, str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg_rows = []
    ue_rows = []
    for (seed, scenario, bandwidth, mode), d in slot_data.items():
        df = d["df"].copy()
        scale = float(TEMPORAL_THROUGHPUT_SCALE_BY_MODE.get(mode, 1.0))
        if "estimated_mbps" in df.columns:
            df["estimated_mbps"] = pd.to_numeric(df["estimated_mbps"], errors="coerce") * scale
        if "estimated_mbps" not in df.columns or df["estimated_mbps"].isna().all():
            continue
        per_slot = df.groupby(["time_s", "beam_id"], sort=False)["estimated_mbps"].sum().reset_index()
        per_slot["seed"] = seed
        per_slot["scenario"] = scenario
        per_slot["bandwidth"] = bandwidth
        per_slot["bandwidth_mhz"] = parse_bw_name(bandwidth)
        per_slot["mode"] = mode
        per_slot = per_slot.rename(columns={"estimated_mbps": "vazao_estimada_mbps"})
        agg_rows.append(per_slot)

        per_ue = df.groupby(["time_s", "beam_id", "rnti"], sort=False)["estimated_mbps"].sum().reset_index()
        per_ue["seed"] = seed
        per_ue["scenario"] = scenario
        per_ue["bandwidth"] = bandwidth
        per_ue["bandwidth_mhz"] = parse_bw_name(bandwidth)
        per_ue["mode"] = mode
        per_ue = per_ue.rename(columns={"estimated_mbps": "vazao_estimada_mbps"})
        ue_rows.append(per_ue)
    return (
        pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame(),
        pd.concat(ue_rows, ignore_index=True) if ue_rows else pd.DataFrame(),
    )


def merge_metrics(flow_summary: pd.DataFrame, tradeoff: pd.DataFrame, backlog_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "scenario", "bandwidth", "bandwidth_mhz", "mode"]
    out = flow_summary.copy()
    if not tradeoff.empty:
        cols = keys + ["jain_rbg_slot_mean", "mean_active_ues_per_slot", "mean_zero_ue_percent"]
        out = out.merge(tradeoff[cols], on=keys, how="left")
    if not backlog_summary.empty:
        bcols = keys + [
            "mean_backlogged_ues_per_slot", "mean_served_ues_per_slot", "pct_buf_gt0_alloc_eq0",
            "sum_buf_req_mean_per_slot", "sum_buf_req_max_per_slot", "mean_jain_buf_req_per_slot",
            "mean_jain_alloc_per_slot",
        ]
        out = out.merge(backlog_summary[bcols], on=keys, how="left")
    if not out.empty:
        out["mode_order"] = out["mode"].map(MODE_ORDER)
        out = out.sort_values(["bandwidth_mhz", "scenario", "mode_order", "seed"]).drop(columns="mode_order")
    return out


def summarize_repetitions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["scenario", "bandwidth", "bandwidth_mhz", "mode"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"seed", "bandwidth_mhz"}]
    rows = []
    for key, sub in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        row["n_repetitions"] = int(sub["seed"].nunique()) if "seed" in sub else len(sub)
        for c in numeric_cols:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_std"] = float(sub[c].std(ddof=0)) if len(sub) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mode_order"] = out["mode"].map(MODE_ORDER)
        out = out.sort_values(["bandwidth_mhz", "scenario", "mode_order"]).drop(columns="mode_order")
    return out


def _ci95(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = len(vals)
    if n <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / math.sqrt(n))


def summarize_by_bw(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["bandwidth", "bandwidth_mhz", "mode"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"seed", "bandwidth_mhz"}]
    rows = []
    for key, sub in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        row["n_samples"] = len(sub)
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else len(sub)
        for c in numeric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
            row[f"{c}_ci95"] = _ci95(vals)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mode_order"] = out["mode"].map(MODE_ORDER)
        out = out.sort_values(["bandwidth_mhz", "mode_order"]).drop(columns="mode_order")
    return out

# ==========================================================
# 5) PLOTS
# ==========================================================

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


def rolling_series(s: pd.Series, window: int = TEMPORAL_ROLLING_WINDOW) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def plot_tradeoff(tradeoff: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / "01_visao_geral"
    if tradeoff.empty:
        return
    # Por cenário e banda.
    for (scenario, bw), sub in tradeoff.groupby(["scenario", "bandwidth"], sort=False):
        # média das seeds, se houver.
        sub = sub.groupby(["mode"], as_index=False).agg({
            "jain_rbg_slot_mean": "mean",
            "aggregate_throughput_mbps": "mean",
        })
        sub = mode_sort(sub)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for _, row in sub.iterrows():
            m = row["mode"]
            ax.scatter(row["jain_rbg_slot_mean"], row["aggregate_throughput_mbps"], marker=MARKERS.get(m, "o"), s=130, color=COLORS.get(m), label=label(m))
            ax.annotate(label(m), (row["jain_rbg_slot_mean"], row["aggregate_throughput_mbps"]), textcoords="offset points", xytext=(6, 6), fontsize=TICK_SIZE)
        ax.set_xlabel("Índice de Jain médio da alocação de RBGs")
        ax.set_ylabel("Vazão agregada (Mbps)")
        ax.set_title(f"Trade-off vazão-fairness — {scenario_label(scenario)} — {bw}")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"tradeoff_{scenario}_{bw}.png")


def plot_consolidated_by_band(summary_bw: pd.DataFrame, out_root: Path) -> None:
    if summary_bw.empty:
        return
    out_dir = out_root / "01_visao_geral"
    metrics = [
        ("aggregate_throughput_mbps_mean", "Vazão agregada média (Mbps)", "vazao_agregada_por_banda.png"),
        ("p5_flow_throughput_mbps_mean", "Vazão do percentil 5 (Mbps)", "p5_por_banda.png"),
        ("jain_rbg_slot_mean_mean", "Índice de Jain médio da alocação de RBGs", "jain_rbg_por_banda.png"),
        ("pct_buf_gt0_alloc_eq0_mean", "Starvation real (%)", "starvation_real_por_banda.png"),
        ("mean_served_ues_per_slot_mean", "UEs atendidos por slot", "ues_atendidos_por_banda.png"),
        ("mean_loss_ratio_mean", "Taxa de perda", "taxa_perda_por_banda.png"),
        ("mean_delay_ms_mean", "Atraso médio (ms)", "atraso_medio_por_banda.png"),
        ("mean_jitter_ms_mean", "Jitter médio (ms)", "jitter_medio_por_banda.png"),
    ]
    for col, ylabel, fname in metrics:
        if col not in summary_bw.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ci_col = col.replace("_mean", "_ci95")
        for mode in MODES:
            sub = summary_bw[summary_bw["mode"] == mode].sort_values("bandwidth_mhz")
            if sub.empty:
                continue
            yerr = sub[ci_col] if ci_col in sub.columns else None
            ax.errorbar(
                sub["bandwidth_mhz"],
                sub[col],
                yerr=yerr,
                marker=MARKERS.get(mode, "o"),
                linewidth=LINE_WIDTH,
                capsize=3 if yerr is not None else 0,
                label=label(mode),
                color=COLORS.get(mode),
            )
        ax.set_xlabel("Largura de banda (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " por largura de banda")
        set_bw_ticks(ax)
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / fname)


def plot_tradeoff_combined_by_band(summary_bw: pd.DataFrame, out_root: Path) -> None:
    if summary_bw.empty:
        return
    required = {
        "jain_rbg_slot_mean_mean",
        "aggregate_throughput_mbps_mean",
        "jain_rbg_slot_mean_ci95",
        "aggregate_throughput_mbps_ci95",
    }
    if not {"jain_rbg_slot_mean_mean", "aggregate_throughput_mbps_mean"}.issubset(summary_bw.columns):
        return
    out_dir = out_root / "01_visao_geral"
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    for mode in MODES:
        sub = summary_bw[summary_bw["mode"] == mode].sort_values("bandwidth_mhz")
        if sub.empty:
            continue
        xerr = sub["jain_rbg_slot_mean_ci95"] if "jain_rbg_slot_mean_ci95" in sub.columns else None
        yerr = sub["aggregate_throughput_mbps_ci95"] if "aggregate_throughput_mbps_ci95" in sub.columns else None
        ax.errorbar(
            sub["jain_rbg_slot_mean_mean"],
            sub["aggregate_throughput_mbps_mean"],
            xerr=xerr,
            yerr=yerr,
            marker=MARKERS.get(mode, "o"),
            linewidth=LINE_WIDTH,
            capsize=3,
            color=COLORS.get(mode),
            label=label(mode),
        )
        for _, row in sub.iterrows():
            ax.text(
                row["jain_rbg_slot_mean_mean"] + 0.004,
                row["aggregate_throughput_mbps_mean"],
                f"{int(row['bandwidth_mhz'])}",
                fontsize=8,
                color=COLORS.get(mode),
            )
    ax.set_xlabel("Índice de Jain médio da alocação de RBGs")
    ax.set_ylabel("Vazão agregada média (Mbps)")
    ax.set_title("Trade-off combinado por largura de banda")
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.legend(frameon=True)
    savefig(fig, out_dir / "tradeoff_combinado_por_banda.png")


def plot_qos_bars(flow_summary: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / "05_qos"
    metrics = [
        ("aggregate_throughput_mbps", "Vazão agregada (Mbps)"),
        ("p5_flow_throughput_mbps", "Vazão do percentil 5 (Mbps)"),
        ("mean_delay_ms", "Atraso médio (ms)"),
        ("mean_loss_ratio", "Taxa de perda média"),
    ]
    for col, ylabel in metrics:
        if col not in flow_summary.columns:
            continue
        for (scenario, bw), sub in flow_summary.groupby(["scenario", "bandwidth"], sort=False):
            sub = sub.groupby("mode", as_index=False)[col].mean()
            sub = mode_sort(sub)
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar([label(m) for m in sub["mode"]], sub[col], color=[COLORS.get(m) for m in sub["mode"]])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — {scenario_label(scenario)} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            savefig(fig, out_dir / f"{col}_{scenario}_{bw}.png")


def plot_starvation(slot_data: dict[tuple[int, str, str, str], dict[str, Any]], backlog_summary: pd.DataFrame, backlog_per_ue: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / "03_starvation"
    # Starvation por slot a partir de alocação = 0, para comparação visual.
    keys = sorted(set((scenario, bw) for (_, scenario, bw, _) in slot_data.keys()), key=lambda x: (x[0], parse_bw_name(x[1])))
    for scenario, bw in keys:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        for mode in MODES:
            series_all = []
            for (seed, sc, b, m), d in slot_data.items():
                if sc == scenario and b == bw and m == mode:
                    y = d["zero_fraction_per_slot"].reset_index(drop=True) * 100.0
                    series_all.append(y)
            if not series_all:
                continue
            # média por índice de slot entre seeds.
            max_len = max(len(s) for s in series_all)
            arr = np.full((len(series_all), max_len), np.nan)
            for i, s in enumerate(series_all):
                arr[i, :len(s)] = s.values
            ymean = pd.Series(np.nanmean(arr, axis=0))
            ax.plot(ymean.index, rolling_series(ymean), linewidth=LINE_WIDTH, label=label(mode), color=COLORS.get(mode))
        ax.set_xlabel("Índice do slot")
        ax.set_ylabel("UEs sem alocação no slot (%)")
        ax.set_title(f"Starvation instantâneo por slot — {scenario_label(scenario)} — {bw}")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"starvation_instantaneo_por_slot_{scenario}_{bw}.png")

    # Starvation real agregada.
    if not backlog_summary.empty and "pct_buf_gt0_alloc_eq0" in backlog_summary.columns:
        for (scenario, bw), sub in backlog_summary.groupby(["scenario", "bandwidth"], sort=False):
            sub = sub.groupby("mode", as_index=False)["pct_buf_gt0_alloc_eq0"].mean()
            sub = mode_sort(sub)
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
            ax.bar([label(m) for m in sub["mode"]], sub["pct_buf_gt0_alloc_eq0"], color=[COLORS.get(m) for m in sub["mode"]])
            ax.set_ylabel("Starvation real (%)")
            ax.set_title(f"Backlog pendente sem alocação — {scenario_label(scenario)} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            savefig(fig, out_dir / f"starvation_real_{scenario}_{bw}.png")

    # Backlog sem atendimento por UE.
    if not backlog_per_ue.empty and "pct_backlog_not_served" in backlog_per_ue.columns:
        for (scenario, bw), sub in backlog_per_ue.groupby(["scenario", "bandwidth"], sort=False):
            sub_mean = sub.groupby(["rnti", "mode"], as_index=False)["pct_backlog_not_served"].mean()
            modes = [m for m in MODES if m in sub_mean["mode"].unique()]
            pivot = sub_mean.pivot(index="rnti", columns="mode", values="pct_backlog_not_served").reindex(columns=modes).sort_index()
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, len(pivot.columns))
            for i, m in enumerate(pivot.columns):
                offset = (i - (len(pivot.columns) - 1) / 2) * width
                ax.bar(x + offset, pivot[m].values, width=width, label=label(m), color=COLORS.get(m))
            ax.set_xticks(x)
            ax.set_xticklabels([str(r) for r in pivot.index])
            ax.set_xlabel("UE (RNTI)")
            ax.set_ylabel("Slots com backlog sem alocação (%)")
            ax.set_title(f"Backlog sem atendimento por UE — {scenario_label(scenario)} — {bw}")
            ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
            ax.legend(frameon=True)
            savefig(fig, out_dir / f"backlog_sem_atendimento_por_ue_{scenario}_{bw}.png")


def plot_backlog(backlog_summary: pd.DataFrame, out_root: Path) -> None:
    if backlog_summary.empty:
        return
    out_dir = out_root / "04_backlog"
    for (scenario, bw), sub in backlog_summary.groupby(["scenario", "bandwidth"], sort=False):
        sub = sub.groupby("mode", as_index=False).agg({
            "mean_backlogged_ues_per_slot": "mean",
            "mean_served_ues_per_slot": "mean",
            "mean_jain_buf_req_per_slot": "mean",
            "mean_jain_alloc_per_slot": "mean",
            "sum_buf_req_mean_per_slot": "mean",
        })
        sub = mode_sort(sub)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["mean_backlogged_ues_per_slot"], width, label="UEs com backlog")
        ax.bar(x + width / 2, sub["mean_served_ues_per_slot"], width, label="UEs atendidos")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Média por slot")
        ax.set_title(f"UEs com backlog vs UEs atendidos — {scenario_label(scenario)} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"backlog_vs_atendidos_{scenario}_{bw}.png")

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.bar(x - width / 2, sub["mean_jain_buf_req_per_slot"], width, label="Jain do backlog")
        ax.bar(x + width / 2, sub["mean_jain_alloc_per_slot"], width, label="Jain da alocação")
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in sub["mode"]])
        ax.set_ylabel("Índice de Jain médio")
        ax.set_title(f"Jain do backlog vs Jain da alocação — {scenario_label(scenario)} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)
        savefig(fig, out_dir / f"jain_backlog_vs_alocacao_{scenario}_{bw}.png")

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.bar([label(m) for m in sub["mode"]], sub["sum_buf_req_mean_per_slot"], color=[COLORS.get(m) for m in sub["mode"]])
        ax.set_ylabel("Backlog médio total por slot (bytes)")
        ax.set_title(f"Backlog médio total — {scenario_label(scenario)} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        savefig(fig, out_dir / f"backlog_total_medio_{scenario}_{bw}.png")

def plot_backlog_per_ue(backlog_per_ue: pd.DataFrame, out_root: Path) -> None:
    if backlog_per_ue.empty:
        return

    out_dir = out_root / "04_backlog"

    for (scenario, bw), sub in backlog_per_ue.groupby(
        ["scenario", "bandwidth"],
        sort=False
    ):

        # média entre seeds
        sub_mean = (
            sub.groupby(["rnti", "mode"], as_index=False)["mean_buf_req"]
            .mean()
        )

        modes = [m for m in MODES if m in sub_mean["mode"].unique()]

        pivot = (
            sub_mean.pivot(
                index="rnti",
                columns="mode",
                values="mean_buf_req"
            )
            .reindex(columns=modes)
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(pivot.columns))

        for i, m in enumerate(pivot.columns):
            offset = (i - (len(pivot.columns) - 1) / 2) * width

            ax.bar(
                x + offset,
                pivot[m].values,
                width=width,
                label=label(m),
                color=COLORS.get(m),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"UE {r}" for r in pivot.index])

        ax.set_xlabel("UE")
        ax.set_ylabel("Backlog médio (bytes)")
        ax.set_title(
            f"Backlog médio por UE — "
            f"{scenario_label(scenario)} — {bw}"
        )

        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        ax.legend(frameon=True)

        savefig(
            fig,
            out_dir / f"backlog_medio_por_ue_{scenario}_{bw}.png"
        )

def plot_temporal(temporal_backlog: pd.DataFrame, temporal_thr: pd.DataFrame, temporal_ue_thr: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / "02_temporal"
    # Backlog total ao longo do tempo.
    if not temporal_backlog.empty:
        for (scenario, bw), sub0 in temporal_backlog.groupby(["scenario", "bandwidth"], sort=False):
            for metric, ylabel, fname in [
                ("backlog_total_bytes", "Backlog total (bytes)", "backlog_total_tempo"),
                ("starvation_real_pct", "Starvation real (%)", "starvation_real_tempo"),
                ("ues_atendidos", "UEs atendidos", "ues_atendidos_tempo"),
            ]:
                fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
                for mode in MODES:
                    sub = sub0[sub0["mode"] == mode]
                    if sub.empty or metric not in sub.columns:
                        continue
                    # média por tempo entre seeds.
                    g = sub.groupby("time_s")[metric].mean().sort_index()
                    ax.plot(g.index, rolling_series(g), linewidth=LINE_WIDTH, label=label(mode), color=COLORS.get(mode))
                ax.set_xlabel("Tempo de simulação (s)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ylabel} ao longo do tempo — {scenario_label(scenario)} — {bw}")
                ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
                ax.legend(frameon=True)
                savefig(fig, out_dir / f"{fname}_{scenario}_{bw}.png")

    # Vazão agregada estimada ao longo do tempo.
    if not temporal_thr.empty:
        for (scenario, bw), sub0 in temporal_thr.groupby(["scenario", "bandwidth"], sort=False):
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            for mode in MODES:
                sub = sub0[sub0["mode"] == mode]
                if sub.empty:
                    continue
                g = (
                    sub.groupby(["seed", "time_s"])["vazao_estimada_mbps"]
                    .sum()
                    .groupby("time_s")
                    .mean()
                    .sort_index()
                )
                ax.plot(g.index, rolling_series(g), linewidth=LINE_WIDTH, label=label(mode), color=COLORS.get(mode))
            ax.set_xlabel("Tempo de simulação (s)")
            ax.set_ylabel("Vazão estimada (Mbps)")
            ax.set_title(f"Vazão estimada ao longo do tempo — {scenario_label(scenario)} — {bw}")
            ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
            ax.legend(frameon=True)
            savefig(fig, out_dir / f"vazao_estimada_tempo_{scenario}_{bw}.png")

    # Vazão estimada por UE ao longo do tempo. Apenas IA-FMR e MR por padrão, para não poluir.
    if not temporal_ue_thr.empty:
        modes_to_plot = [m.strip() for m in os.environ.get("FMR_TEMPORAL_UE_MODES", "fmr_rl mr").split()]
        for (scenario, bw, mode), sub0 in temporal_ue_thr.groupby(["scenario", "bandwidth", "mode"], sort=False):
            if mode not in modes_to_plot:
                continue
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            for rnti, sub in sub0.groupby("rnti"):
                g = (
                    sub.groupby(["seed", "time_s"])["vazao_estimada_mbps"]
                    .sum()
                    .groupby("time_s")
                    .mean()
                    .sort_index()
                )
                ax.plot(g.index, rolling_series(g), linewidth=1.1, alpha=0.85, label=f"UE {int(rnti)}")
            ax.set_xlabel("Tempo de simulação (s)")
            ax.set_ylabel("Vazão estimada por UE (Mbps)")
            ax.set_title(f"Vazão estimada por UE — {label(mode)} — {scenario_label(scenario)} — {bw}")
            ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
            ax.legend(frameon=True, ncols=3)
            savefig(fig, out_dir / f"vazao_por_ue_tempo_{mode}_{scenario}_{bw}.png")


def plot_appendix(slot_data: dict[tuple[int, str, str, str], dict[str, Any]], out_root: Path) -> None:
    out_dir = out_root / "06_apendice"
    # Boxplot do Jain instantâneo, útil como material complementar.
    keys = sorted(set((scenario, bw) for (_, scenario, bw, _) in slot_data.keys()), key=lambda x: (x[0], parse_bw_name(x[1])))
    for scenario, bw in keys:
        values, labels = [], []
        for mode in MODES:
            series = []
            for (seed, sc, b, m), d in slot_data.items():
                if sc == scenario and b == bw and m == mode:
                    series.extend(d["jain"].dropna().values.tolist())
            if series:
                values.append(np.asarray(series))
                labels.append(label(mode))
        if not values:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        bp = ax.boxplot(values, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, lab in zip(bp["boxes"], labels):
            mode = next((m for m, l in LABELS.items() if l == lab), None)
            patch.set_facecolor(COLORS.get(mode, "gray"))
            patch.set_alpha(0.65)
        ax.set_ylabel("Índice de Jain instantâneo")
        ax.set_title(f"Boxplot do Jain instantâneo — {scenario_label(scenario)} — {bw}")
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA)
        savefig(fig, out_dir / f"boxplot_jain_instantaneo_{scenario}_{bw}.png")

# ==========================================================
# 6) GERAÇÃO DE SAÍDAS
# ==========================================================

def write_tables(tables_dir: Path, **dfs: pd.DataFrame) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    excel_max_rows = 1_048_576
    xlsx_path = tables_dir / "resultados_consolidados.xlsx"

    for name, df in dfs.items():
        if df is None or df.empty:
            continue

        df.to_csv(tables_dir / f"{name}.csv", index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in dfs.items():
            if df is None or df.empty:
                continue

            if len(df) > excel_max_rows:
                print(
                    f"[WARN] {name} tem {len(df)} linhas. "
                    f"Salvo apenas como CSV, não incluído no XLSX."
                )
                continue

            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[OK] saved: {xlsx_path}")


def set_bw_ticks(ax, data=None) -> None:
    if data is not None and "bandwidth_mhz" in data.columns:
        bw_ticks = sorted(data["bandwidth_mhz"].dropna().astype(int).unique())
    else:
        bw_ticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    ax.set_xticks(bw_ticks)
    ax.set_xlim(min(bw_ticks) - 2, max(bw_ticks) + 2)

def generate_all_outputs(run_dir: Path, plot_groups: set[str] | None = None) -> None:
    setup_matplotlib()
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[POST] Carregando flow summaries...")
    df_all_flows = load_flow_summary(run_dir)
    if df_all_flows.empty:
        raise RuntimeError("Nenhum flow_summary encontrado.")
    flow_summary = summarize_flows(df_all_flows)

    print("[POST] Processando slot logs em paralelo...")
    (
        slot_data,
        slot_summary,
        backlog_summary,
        backlog_per_ue,
        temporal_backlog,
        temporal_thr,
        temporal_ue_thr,
    ) = process_slot_logs_parallel(run_dir)

    print("[POST] Montando trade-off...")
    tradeoff = flow_summary.merge(
        slot_summary[
            [
                "seed",
                "scenario",
                "bandwidth",
                "bandwidth_mhz",
                "mode",
                "jain_rbg_slot_mean",
                "mean_active_ues_per_slot",
                "mean_zero_ue_percent",
            ]
        ],
        on=["seed", "scenario", "bandwidth", "bandwidth_mhz", "mode"],
        how="left",
    )

    tradeoff = tradeoff[
        [
            "run_id",
            "seed",
            "scenario",
            "bandwidth",
            "bandwidth_mhz",
            "mode",
            "aggregate_throughput_mbps",
            "mean_flow_throughput_mbps",
            "p5_flow_throughput_mbps",
            "jain_rbg_slot_mean",
            "mean_active_ues_per_slot",
            "mean_zero_ue_percent",
        ]
    ]

    print("[POST] Consolidando métricas...")
    metrics_long = merge_metrics(flow_summary, tradeoff, backlog_summary)
    summary_by_scenario = summarize_repetitions(metrics_long)
    summary_by_bw = summarize_by_bw(metrics_long)

    print("[POST] Salvando tabelas...")
    write_tables(
        tables_dir,
        flow_comparison_long=flow_summary,
        tradeoff_points=tradeoff,
        backlog_summary=backlog_summary,
        backlog_per_ue=backlog_per_ue,
        temporal_backlog=temporal_backlog,
        temporal_throughput_estimated=temporal_thr,
        temporal_ue_throughput_estimated=temporal_ue_thr,
        metrics_long=metrics_long,
        summary_by_scenario=summary_by_scenario,
        summary_by_bw=summary_by_bw,
    )

    print("[POST] Gerando gráficos em português...")
    plot_groups = plot_groups or {"all"}
    do_all = "all" in plot_groups

    if do_all or "tradeoff" in plot_groups:
        plot_tradeoff(tradeoff, plots_dir)
        plot_tradeoff_combined_by_band(summary_by_bw, plots_dir)
    if do_all or "overview" in plot_groups:
        plot_consolidated_by_band(summary_by_bw, plots_dir)
    if do_all or "temporal" in plot_groups:
        plot_temporal(temporal_backlog, temporal_thr, temporal_ue_thr, plots_dir)
    if do_all or "starvation" in plot_groups:
        plot_starvation(slot_data, backlog_summary, backlog_per_ue, plots_dir)
    if do_all or "backlog" in plot_groups:
        plot_backlog(backlog_summary, plots_dir)
        plot_backlog_per_ue(backlog_per_ue, plots_dir)
    if do_all or "qos" in plot_groups:
        plot_qos_bars(flow_summary, plots_dir)

    if (do_all or "appendix" in plot_groups) and os.environ.get("FMR_GENERATE_APPENDIX", "1") not in {"0", "false", "False", "no"}:
        plot_appendix(slot_data, plots_dir)

    print("\n[DONE] Tabelas:")
    for p in sorted(tables_dir.glob("*.csv")):
        print("  ", p)

    print("\n[DONE] Planilha:")
    print("  ", tables_dir / "resultados_consolidados.xlsx")

    print("\n[DONE] Gráficos:")
    print("  ", plots_dir)

# ==========================================================
# 7) CLI
# ==========================================================

def _parse_cli_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for item in values:
        for x in str(item).replace(",", " ").split():
            out.append(int(x))
    return out or None


def _parse_cli_str_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for item in values:
        out.extend([x.strip() for x in str(item).replace(",", " ").split() if x.strip()])
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa experimentos dinâmicos IA-FMR no ns-3 e gera tabelas/gráficos.")
    parser.add_argument("--run-id", default=None, help="Nome manual do run. Se omitido, usa data/hora.")
    parser.add_argument("--skip-run", action="store_true", help="Não executa ns-3; apenas gera tabelas/gráficos de um run existente.")
    parser.add_argument("--existing-run-dir", default=None, help="Diretório de run existente para --skip-run, --only-post ou --only-plots.")
    parser.add_argument("--only-scenario", default=None, help="Executa apenas um cenário pelo nome.")
    parser.add_argument("--only-post", action="store_true", help="Alias para --skip-run quando usado com --existing-run-dir.")
    parser.add_argument("--only-plots", action="store_true", help="Não executa ns-3; gera apenas tabelas/gráficos de um run existente.")
    parser.add_argument("--models-dir", default=None, help="Diretório com model_<BW>.zip para o IA-FMR.")
    parser.add_argument("--only-bw", nargs="*", default=None, help="Bandas específicas em MHz. Ex.: --only-bw 10 20 50")
    parser.add_argument("--only-seed", nargs="*", default=None, help="Seeds específicas. Ex.: --only-seed 1 2 3")
    parser.add_argument("--only-mode", nargs="*", default=None, choices=MODES, help="Escalonadores específicos: rr pf mr fmr_rl")
    parser.add_argument("--plot-groups", nargs="*", default=None, help="Grupos de gráficos: all overview tradeoff temporal starvation backlog qos appendix")
    parser.add_argument("--skip-existing", action="store_true", help="Reaproveita simulações com flow_summary e slot_log já existentes.")
    parser.add_argument("--force", action="store_true", help="Reexecuta mesmo quando os CSVs já existem.")
    return parser.parse_args()


def main() -> None:
    global MODEL_DIR, SEEDS

    args = parse_args()

    if args.models_dir:
        MODEL_DIR = Path(args.models_dir).expanduser().resolve()

    cli_bws = _parse_cli_int_list(args.only_bw)
    cli_seeds = _parse_cli_int_list(args.only_seed)
    cli_modes = _parse_cli_str_list(args.only_mode)
    plot_groups = set(_parse_cli_str_list(args.plot_groups) or ["all"])

    if cli_seeds:
        SEEDS = cli_seeds

    if args.only_post or args.only_plots:
        args.skip_run = True

    run_id = args.run_id or f"dynamic_qos_{timestamp()}"

    if args.skip_run:
        if not args.existing_run_dir:
            raise ValueError("Use --existing-run-dir junto com --skip-run, --only-post ou --only-plots")
        run_dir = Path(args.existing_run_dir).expanduser().resolve()
    else:
        run_dir = BASE_DIR / "compare_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        scenarios = SCENARIOS

        if args.only_scenario:
            scenarios = [s for s in scenarios if s.name == args.only_scenario]
            if not scenarios:
                raise ValueError(f"Cenário não encontrado: {args.only_scenario}")

        if cli_bws:
            scenarios = [
                Scenario(
                    name=s.name,
                    purpose=s.purpose,
                    bandwidths_mhz=[bw for bw in s.bandwidths_mhz if bw in cli_bws],
                    lambda_value=s.lambda_value,
                    num_dl_flows_per_ue=s.num_dl_flows_per_ue,
                    sim_time=s.sim_time,
                    udp_packet_size=s.udp_packet_size,
                    extra_ns3_args=s.extra_ns3_args,
                )
                for s in scenarios
            ]
            scenarios = [s for s in scenarios if s.bandwidths_mhz]
            if not scenarios:
                raise ValueError(f"Nenhuma banda válida após filtro --only-bw={cli_bws}")

        modes_to_run = cli_modes or list(MODES)

        print(f"[INFO] RUN_ID={run_id}")
        print(f"[INFO] RUN_DIR={run_dir}")
        print(f"[INFO] BASE_DIR={BASE_DIR}")
        print(f"[INFO] BIN={BIN}")
        print(f"[INFO] MODEL_DIR={MODEL_DIR}")
        print(f"[INFO] BWS={[bw for s in scenarios for bw in s.bandwidths_mhz]}")
        print(f"[INFO] SEEDS={SEEDS}")
        print(f"[INFO] MODES={modes_to_run}")
        print(f"[INFO] SKIP_EXISTING={args.skip_existing} FORCE={args.force}")
        print(f"[INFO] CLASSIC_PARALLEL={FMR_RUN_CLASSIC_PARALLEL} workers={FMR_MAX_WORKERS_CLASSIC}")
        run_all_experiments(
            run_dir,
            scenarios,
            modes_to_run=modes_to_run,
            skip_existing=args.skip_existing,
            force=args.force,
        )

    generate_all_outputs(run_dir, plot_groups=plot_groups)


if __name__ == "__main__":
    main()
