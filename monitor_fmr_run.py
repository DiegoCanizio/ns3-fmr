#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict


DEFAULT_SCENARIO = "dynamic_continuous_30s"


def now_ts() -> float:
    return time.time()


def fmt_dt(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "-"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def parse_csv_or_space(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def parse_bws(values: list[str] | None) -> list[int]:
    return [int(x) for x in parse_csv_or_space(values)]


def parse_modes(values: list[str] | None) -> list[str]:
    return parse_csv_or_space(values)


def read_seeds(path: Path | None) -> list[int]:
    if path is None:
        return []

    txt = path.read_text(errors="ignore")

    # Remove comentários, mas permite seeds separadas por espaço, vírgula ou quebra de linha.
    clean_lines = []
    for line in txt.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            clean_lines.append(line)

    clean = "\n".join(clean_lines)
    return [int(x) for x in re.findall(r"\d+", clean)]


def seed_dir_name(seed: int | str) -> str:
    if isinstance(seed, int):
        return f"seed_{seed:03d}"
    s = str(seed)
    if s.startswith("seed_"):
        return s
    return f"seed_{int(s):03d}"


def infer_seeds(run_dir: Path) -> list[str]:
    return sorted([p.name for p in run_dir.glob("seed_*") if p.is_dir()])


def infer_bws(run_dir: Path, scenario: str) -> list[int]:
    bws = set()
    for p in run_dir.glob(f"seed_*/{scenario}/bw*"):
        if p.is_dir():
            m = re.match(r"bw(\d+)$", p.name)
            if m:
                bws.add(int(m.group(1)))
    return sorted(bws)


def infer_modes(run_dir: Path, scenario: str) -> list[str]:
    modes = set()
    for p in run_dir.glob(f"seed_*/{scenario}/bw*/*"):
        if p.is_dir():
            modes.add(p.name)
    return sorted(modes)


def task_key(seed_name: str, bw: int, mode: str) -> str:
    return f"{seed_name}|bw{bw}|{mode}"


def is_complete(mode_dir: Path, mode: str) -> bool:
    required = [
        mode_dir / f"flow_summary_{mode}.csv",
        mode_dir / f"slot_log_{mode}.csv",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def completion_mtime(mode_dir: Path, mode: str) -> float | None:
    files = [
        mode_dir / f"flow_summary_{mode}.csv",
        mode_dir / f"slot_log_{mode}.csv",
        mode_dir / "summary.txt",
        mode_dir / "ns3.log",
    ]
    mtimes = [p.stat().st_mtime for p in files if p.exists()]
    return max(mtimes) if mtimes else None


def has_started(mode_dir: Path) -> bool:
    if not mode_dir.exists():
        return False
    try:
        next(mode_dir.iterdir())
        return True
    except StopIteration:
        return False


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "created_at": now_ts(),
        "tasks": {},
    }


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def pct(a: int, b: int) -> str:
    if b <= 0:
        return "0.0%"
    return f"{100.0 * a / b:5.1f}%"


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def describe_eta(completed_times: list[float], remaining: int, now: float) -> tuple[str, str, float | None]:
    if remaining <= 0:
        return "finalizado", fmt_dt(now), 0.0

    if len(completed_times) >= 2:
        first = min(completed_times)
        last = max(completed_times)
        elapsed = max(1.0, last - first)
        rate = (len(completed_times) - 1) / elapsed
        if rate > 0:
            eta_seconds = remaining / rate
            return fmt_duration(eta_seconds), fmt_dt(now + eta_seconds), rate

    return "-", "-", None


def print_table(title: str, rows: list[tuple[str, int, int, int]]) -> None:
    print()
    print(title)
    print("-" * min(100, shutil.get_terminal_size((100, 20)).columns))
    print(f"{'Item':<18} {'Concluídas':>12} {'Iniciadas':>10} {'Total':>8} {'%':>8}")
    for item, done, started, total in rows:
        print(f"{item:<18} {done:>12} {started:>10} {total:>8} {pct(done, total):>8}")


def scan(args, state: dict) -> dict:
    run_dir = args.run_dir
    scenario = args.scenario
    tnow = now_ts()

    if args.seeds:
        seed_names = [seed_dir_name(s) for s in args.seeds]
    else:
        seed_names = infer_seeds(run_dir)

    bws = args.bws or infer_bws(run_dir, scenario)
    modes = args.modes or infer_modes(run_dir, scenario)

    tasks = []
    for seed_name in seed_names:
        for bw in bws:
            for mode in modes:
                mode_dir = run_dir / seed_name / scenario / f"bw{bw}" / mode
                key = task_key(seed_name, bw, mode)
                started = has_started(mode_dir)
                complete = is_complete(mode_dir, mode)
                ctime = completion_mtime(mode_dir, mode) if complete else None

                rec = state["tasks"].setdefault(key, {})
                rec.setdefault("seed", seed_name)
                rec.setdefault("bw", bw)
                rec.setdefault("mode", mode)

                if started and "first_started_seen" not in rec:
                    rec["first_started_seen"] = tnow

                if complete and "completed_seen" not in rec:
                    rec["completed_seen"] = tnow

                if complete and ctime:
                    rec["completed_mtime"] = ctime

                tasks.append({
                    "key": key,
                    "seed": seed_name,
                    "bw": bw,
                    "mode": mode,
                    "dir": mode_dir,
                    "started": started,
                    "complete": complete,
                    "completed_mtime": ctime,
                    "state": rec,
                })

    return {
        "now": tnow,
        "run_dir": run_dir,
        "scenario": scenario,
        "seed_names": seed_names,
        "bws": bws,
        "modes": modes,
        "tasks": tasks,
    }


def report(snapshot: dict, state: dict, args) -> None:
    tnow = snapshot["now"]
    tasks = snapshot["tasks"]
    total = len(tasks)
    done = sum(1 for t in tasks if t["complete"])
    started = sum(1 for t in tasks if t["started"])
    pending = total - done

    completed_times = [t["completed_mtime"] for t in tasks if t["complete"] and t["completed_mtime"]]
    eta, finish_at, rate = describe_eta(completed_times, pending, tnow)

    seed_total_tasks = len(snapshot["bws"]) * len(snapshot["modes"])
    per_seed_done = defaultdict(int)
    per_seed_started = defaultdict(int)
    for t in tasks:
        if t["complete"]:
            per_seed_done[t["seed"]] += 1
        if t["started"]:
            per_seed_started[t["seed"]] += 1

    seeds_total = len(snapshot["seed_names"])
    seeds_done = sum(1 for s in snapshot["seed_names"] if per_seed_done[s] == seed_total_tasks and seed_total_tasks > 0)
    seeds_started = sum(1 for s in snapshot["seed_names"] if per_seed_started[s] > 0)

    durations = []
    for t in tasks:
        rec = t["state"]
        if "first_started_seen" in rec and "completed_seen" in rec:
            dur = rec["completed_seen"] - rec["first_started_seen"]
            if dur >= 0:
                durations.append(dur)

    avg_task_duration = sum(durations) / len(durations) if durations else None

    if args.clear:
        clear_screen()

    print("=" * min(100, shutil.get_terminal_size((100, 20)).columns))
    print("Monitor FMR")
    print("=" * min(100, shutil.get_terminal_size((100, 20)).columns))
    print(f"Agora:             {fmt_dt(tnow)}")
    print(f"Run:               {snapshot['run_dir']}")
    print(f"Cenário:           {snapshot['scenario']}")
    print(f"Bandas:            {','.join(map(str, snapshot['bws'])) if snapshot['bws'] else '-'}")
    print(f"Modos:             {','.join(snapshot['modes']) if snapshot['modes'] else '-'}")
    print(f"Seeds:             {seeds_total}")
    print()
    print(f"Simulações:        {done}/{total} concluídas  {pct(done, total)}")
    print(f"Iniciadas:         {started}/{total}")
    print(f"Pendentes:         {pending}")
    print(f"Progresso:         {progress_bar(done, total)}")
    print()
    print(f"Seeds completas:   {seeds_done}/{seeds_total}  {pct(seeds_done, seeds_total)}")
    print(f"Seeds iniciadas:   {seeds_started}/{seeds_total}")
    print()
    print(f"ETA estimado:      {eta}")
    print(f"Término estimado:  {finish_at}")
    if rate:
        print(f"Ritmo estimado:    {rate * 3600:.2f} simulações/hora")
    else:
        print(f"Ritmo estimado:    -")
    print(f"Tempo médio por simulação observada pelo monitor: {fmt_duration(avg_task_duration)}")
    print()
    print("Observação: o tempo médio por simulação só fica preciso se o monitor estiver rodando desde o começo.")

    rows_mode = []
    for mode in snapshot["modes"]:
        sub = [t for t in tasks if t["mode"] == mode]
        rows_mode.append((
            mode,
            sum(1 for t in sub if t["complete"]),
            sum(1 for t in sub if t["started"]),
            len(sub),
        ))
    print_table("Progresso por modo", rows_mode)

    rows_bw = []
    for bw in snapshot["bws"]:
        sub = [t for t in tasks if t["bw"] == bw]
        rows_bw.append((
            f"bw{bw}",
            sum(1 for t in sub if t["complete"]),
            sum(1 for t in sub if t["started"]),
            len(sub),
        ))
    print_table("Progresso por banda", rows_bw)

    incomplete = [t for t in tasks if not t["complete"]]
    incomplete_started = [t for t in incomplete if t["started"]]
    if incomplete_started[:10]:
        print()
        print("Algumas simulações iniciadas e ainda não concluídas")
        print("-" * min(100, shutil.get_terminal_size((100, 20)).columns))
        for t in incomplete_started[:10]:
            rec = t["state"]
            started_at = rec.get("first_started_seen")
            print(f"{t['seed']} bw{t['bw']} {t['mode']}  iniciado há {fmt_duration(tnow - started_at if started_at else None)}")

    print()
    print(f"Próxima atualização em {args.interval}s. Para sair: Ctrl+C")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Monitora runs do run_fmr.py.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--bws", nargs="*", default=None, help="Ex.: --bws 10,15,20 ou --bws 10 15 20")
    parser.add_argument("--modes", nargs="*", default=None, help="Ex.: --modes rr pf mr")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela a cada atualização.")
    args = parser.parse_args()

    args.bws = parse_bws(args.bws)
    args.modes = parse_modes(args.modes)
    args.seeds = read_seeds(args.seeds_file)
    args.clear = not args.no_clear

    state_path = args.run_dir / ".monitor_fmr_state.json"

    while True:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        state = load_state(state_path)
        snapshot = scan(args, state)
        save_state(state_path, state)
        report(snapshot, state, args)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
