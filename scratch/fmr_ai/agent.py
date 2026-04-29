#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ctypes
import os
import signal
import sys
import time
from typing import Dict

import numpy as np
import ns3ai_gym_msg_py
from stable_baselines3 import PPO

_STOP = False

# MUST match contrib/nr/model/nr-fmr-ai-msg.h
FMR_AI_MAX_UES = 9
FMR_AI_MAGIC = 0xF0A11
FMR_AI_VER = 1


class FmrAiObs(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("slot", ctypes.c_uint64),
        ("beam_hash", ctypes.c_uint32),
        ("num_ues", ctypes.c_uint32),
        ("total_rbg", ctypes.c_uint32),
        ("rnti", ctypes.c_uint16 * FMR_AI_MAX_UES),
        ("dl_mcs", ctypes.c_uint16 * FMR_AI_MAX_UES),
        ("buf_req", ctypes.c_uint32 * FMR_AI_MAX_UES),
    ]


class FmrAiAct(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("alpha_next", ctypes.c_float),
        ("num_ues", ctypes.c_uint32),
        ("rnti", ctypes.c_uint16 * FMR_AI_MAX_UES),
        ("alloc_rbg", ctypes.c_uint16 * FMR_AI_MAX_UES),
    ]


def _sigint_handler(signum, frame):
    global _STOP
    _STOP = True
    print("\n[agent] SIGINT, stopping...")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0.0:
        return 0.0
    return float((s * s) / (x.size * np.sum(x * x) + 1e-12))


def softmax_tau(logits: np.ndarray, tau: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    tau = max(1e-6, float(tau))
    z = logits / tau
    z = z - np.max(z)
    e = np.exp(z)
    s = float(e.sum())
    if s <= 0:
        return np.full(logits.shape, 1.0 / logits.size, dtype=np.float64)
    return e / s


def largest_remainder(shares: np.ndarray, total: int) -> np.ndarray:
    shares = np.asarray(shares, dtype=np.float64)
    n = shares.size

    if n == 0:
        return np.zeros(0, dtype=np.uint16)

    if total <= 0:
        return np.zeros(n, dtype=np.uint16)

    shares = np.clip(shares, 0.0, None)

    if float(shares.sum()) <= 0.0:
        shares = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        shares = shares / float(shares.sum())

    exact = shares * float(total)
    base = np.floor(exact).astype(np.int64)
    rem = exact - base

    missing = int(total - int(base.sum()))
    if missing > 0:
        order = np.argsort(-rem)
        for k in range(missing):
            base[int(order[k % n])] += 1
    elif missing < 0:
        order = np.argsort(-base)
        excess = -missing
        for idx in order:
            if excess <= 0:
                break
            dec = min(int(base[int(idx)]), excess)
            base[int(idx)] -= dec
            excess -= dec

    base[base < 0] = 0
    return base.astype(np.uint16)


def load_ppo(model_path: str) -> PPO:
    custom_objects = {"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2}
    print(f"[agent] loading PPO: {model_path}")
    model = PPO.load(model_path, custom_objects=custom_objects, device="cpu")
    print("[agent] model loaded")
    return model


class AgentState:
    def __init__(self, n_ues: int, alpha_init: float):
        self.n_ues = int(n_ues)
        self.last_alloc_frac = np.zeros(self.n_ues, dtype=np.float32)
        self.last_tp_norm = np.zeros(self.n_ues, dtype=np.float32)
        self.last_jain = 0.0
        self.current_alpha = float(alpha_init)

    def build_obs(self, dl_mcs: np.ndarray, n_active: int) -> np.ndarray:
        mcs_pad = np.zeros(self.n_ues, dtype=np.float32)
        n = min(int(n_active), self.n_ues)

        if n > 0:
            mcs = np.asarray(dl_mcs[:n], dtype=np.float32)
            denom = max(1.0, float(np.max(mcs)))
            mcs_pad[:n] = mcs / denom

        obs = np.concatenate(
            [
                mcs_pad,
                self.last_alloc_frac,
                self.last_tp_norm,
                np.array([self.last_jain, self.current_alpha], dtype=np.float32),
            ],
            axis=0,
        )

        return obs.astype(np.float32)

    def update_after_action(
        self,
        alloc_active: np.ndarray,
        dl_mcs: np.ndarray,
        n_active: int,
        alpha: float,
    ) -> None:
        alloc_pad = np.zeros(self.n_ues, dtype=np.float32)
        tp_pad = np.zeros(self.n_ues, dtype=np.float32)

        n = min(int(n_active), self.n_ues)
        if n > 0:
            alloc = np.asarray(alloc_active[:n], dtype=np.float32)
            mcs = np.asarray(dl_mcs[:n], dtype=np.float32)

            total_alloc = max(1e-6, float(alloc.sum()))
            alloc_pad[:n] = alloc / total_alloc

            # Approximation compatible with the offline env scale:
            # tp_norm ~ (SE/MCS proxy * alloc) / ideal
            # We do not have exact SE table here, so MCS is used as a monotonic proxy.
            best_mcs = max(1.0, float(np.max(mcs)))
            ideal = best_mcs * max(1.0, float(alloc.sum()))
            tp_proxy = mcs * alloc
            tp_pad[:n] = tp_proxy / max(1e-6, ideal)

        self.last_alloc_frac = alloc_pad
        self.last_tp_norm = tp_pad
        self.last_jain = jain_index(alloc_pad)
        self.current_alpha = float(alpha)


class Ns3AiIface:
    def __init__(
        self,
        shm_size: int,
        segment: str,
        cpp2py: str,
        py2cpp: str,
        lock: str,
        *,
        is_creator: bool = False,
        b1: bool = False,
        b2: bool = False,
    ):
        self.iface = ns3ai_gym_msg_py.Ns3AiMsgInterfaceImpl(
            bool(is_creator),
            bool(b1),
            bool(b2),
            int(shm_size),
            str(segment),
            str(cpp2py),
            str(py2cpp),
            str(lock),
        )
        self._obs_msg = None
        self._obs_buf = None
        self._act_msg = None
        self._act_buf = None

    def recv_begin(self) -> bool:
        self.iface.PyRecvBegin()
        return True

    def recv_end(self) -> None:
        self.iface.PyRecvEnd()

    def send_begin(self) -> bool:
        self.iface.PySendBegin()
        return True

    def send_end(self) -> None:
        self.iface.PySendEnd()

    def get_finished(self) -> bool:
        return bool(self.iface.PyGetFinished())

    def get_obs(self):
        self._obs_msg = self.iface.GetCpp2PyStruct()
        self._obs_buf = self._obs_msg.get_buffer_full()
        return FmrAiObs.from_buffer(self._obs_buf)

    def get_act(self):
        self._act_msg = self.iface.GetPy2CppStruct()
        self._act_buf = self._act_msg.get_buffer_full()
        return FmrAiAct.from_buffer(self._act_buf)


def connect_with_retry(args) -> Ns3AiIface:
    deadline = time.time() + float(args.wait_seconds)
    last = None

    while not _STOP:
        try:
            return Ns3AiIface(
                shm_size=args.shm,
                segment=args.segment,
                cpp2py=args.cpp2py,
                py2cpp=args.py2cpp,
                lock=args.lock,
                is_creator=bool(args.is_creator),
                b1=bool(args.b1),
                b2=bool(args.b2),
            )
        except RuntimeError as e:
            last = e
            if time.time() > deadline:
                raise RuntimeError(f"[agent] could not connect to shm: {e}") from e
            time.sleep(float(args.poll_interval))

    raise RuntimeError("[agent] interrupted while waiting") from last


def main():
    global _STOP
    signal.signal(signal.SIGINT, _sigint_handler)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)

    ap.add_argument("--segment", default="ns3ai_fmr")
    ap.add_argument("--cpp2py", default="fmr_cpp2py")
    ap.add_argument("--py2cpp", default="fmr_py2cpp")
    ap.add_argument("--lock", default="fmr_lock")
    ap.add_argument("--shm", type=int, default=4096)

    ap.add_argument("--obs-dim", type=int, default=29)
    ap.add_argument("--act-dim", type=int, default=10)
    ap.add_argument("--n-ues", type=int, default=9)

    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--default-alpha", type=float, default=0.80)

    # Must match the trained in_action config
    ap.add_argument("--a-min", type=float, default=0.80)
    ap.add_argument("--a-max", type=float, default=0.98)
    ap.add_argument("--alpha-temp", type=float, default=0.70)

    # Must match RewardConfig.tau used in Python training
    ap.add_argument("--tau", type=float, default=0.65)

    ap.add_argument("--wait-seconds", type=float, default=60.0)
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--turn-timeout", type=float, default=10.0)

    ap.add_argument("--is-creator", type=int, default=1)
    ap.add_argument("--b1", type=int, default=0)
    ap.add_argument("--b2", type=int, default=0)

    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"[agent] ERROR: model not found: {model_path}")
        sys.exit(1)

    model = load_ppo(model_path)

    alpha_init = float(np.clip(args.default_alpha, args.a_min, args.a_max))
    state = AgentState(n_ues=args.n_ues, alpha_init=alpha_init)

    print(f"[agent] n_ues={args.n_ues} obs_dim={args.obs_dim} act_dim={args.act_dim}")
    print(
        f"[agent] alpha: a_min={args.a_min} a_max={args.a_max} "
        f"temp={args.alpha_temp} default={alpha_init}"
    )
    print(f"[agent] allocation tau={args.tau}")
    print(
        f"[agent] shm={args.shm} segment={args.segment} cpp2py={args.cpp2py} "
        f"py2cpp={args.py2cpp} lock={args.lock}"
    )
    print(f"[agent] creator={args.is_creator} start={_now()}")

    iface = connect_with_retry(args)
    print("[agent] connected to ns3-ai shm, waiting first obs...")

    deadline = time.time() + float(args.wait_seconds)
    while not _STOP:
        if iface.recv_begin():
            break
        if time.time() > deadline:
            print("[agent] timeout waiting first obs. Exiting.")
            return
        time.sleep(float(args.poll_interval))

    step = 0
    deterministic = bool(args.deterministic)

    try:
        while not _STOP:
            obs_msg = iface.get_obs()

            if obs_msg.magic != FMR_AI_MAGIC:
                raise RuntimeError(f"obs magic mismatch: {obs_msg.magic} != {FMR_AI_MAGIC}")
            if obs_msg.version != FMR_AI_VER:
                raise RuntimeError(f"obs version mismatch: {obs_msg.version} != {FMR_AI_VER}")

            slot = int(obs_msg.slot)
            beam_hash = int(obs_msg.beam_hash)
            num_ues = int(obs_msg.num_ues)
            total_rbg = int(obs_msg.total_rbg)

            if num_ues < 0 or num_ues > args.n_ues:
                raise RuntimeError(f"invalid num_ues={num_ues}")

            n_active = max(0, min(num_ues, args.n_ues))

            rnti = np.array(list(obs_msg.rnti), dtype=np.uint16)[:n_active]
            dl_mcs = np.array(list(obs_msg.dl_mcs), dtype=np.uint16)[:n_active]
            buf_req = np.array(list(obs_msg.buf_req), dtype=np.uint32)[:n_active]

            obs = state.build_obs(dl_mcs=dl_mcs, n_active=n_active)

            if obs.size < args.obs_dim:
                obs = np.pad(obs, (0, args.obs_dim - obs.size), mode="constant", constant_values=0.0)
            else:
                obs = obs[: args.obs_dim]

            iface.recv_end()

            raw_act, _ = model.predict(obs, deterministic=deterministic)
            raw_act = np.asarray(raw_act, dtype=np.float32).reshape(-1)

            if raw_act.size < args.act_dim:
                raw_act = np.pad(raw_act, (0, args.act_dim - raw_act.size), mode="constant", constant_values=0.0)
            else:
                raw_act = raw_act[: args.act_dim]

            logits = raw_act[: args.n_ues]

            alpha_next = float(args.default_alpha)
            if args.act_dim >= args.n_ues + 1:
                raw_alpha = float(raw_act[args.n_ues])
                z = raw_alpha / max(1e-6, float(args.alpha_temp))
                sig = 1.0 / (1.0 + np.exp(-z))
                alpha_next = float(args.a_min + (args.a_max - args.a_min) * sig)

            alpha_next = float(np.clip(alpha_next, args.a_min, args.a_max))

            if n_active > 0:
                shares = softmax_tau(logits[:n_active], tau=args.tau)
                alloc_active = largest_remainder(shares, total_rbg)
            else:
                alloc_active = np.zeros((0,), dtype=np.uint16)

            t0 = time.time()
            while not _STOP:
                if iface.send_begin():
                    break
                if (time.time() - t0) > float(args.turn_timeout):
                    print("[agent] timeout on PySendBegin")
                    return
                time.sleep(float(args.poll_interval))

            act_msg = iface.get_act()
            act_msg.magic = int(FMR_AI_MAGIC)
            act_msg.version = int(FMR_AI_VER)
            act_msg.alpha_next = float(alpha_next)
            act_msg.num_ues = int(n_active)

            for i in range(FMR_AI_MAX_UES):
                if i < n_active:
                    act_msg.rnti[i] = int(rnti[i])
                    act_msg.alloc_rbg[i] = int(alloc_active[i])
                else:
                    act_msg.rnti[i] = 0
                    act_msg.alloc_rbg[i] = 0

            iface.send_end()

            state.update_after_action(
                alloc_active=alloc_active,
                dl_mcs=dl_mcs,
                n_active=n_active,
                alpha=alpha_next,
            )

            step += 1

            if args.print_every and (step % int(args.print_every) == 0):
                s = int(alloc_active.sum()) if n_active > 0 else 0
                active_rntis = ",".join(str(int(x)) for x in rnti[:n_active])
                print(
                    f"[agent] step={step} slot={slot} beam={beam_hash} n={n_active} "
                    f"total={total_rbg} sum_alloc={s} alpha={alpha_next:.3f} "
                    f"jain_last={state.last_jain:.3f} rnti={active_rntis}"
                )

            if iface.get_finished():
                print("[agent] ns-3 signaled finished. Exiting.")
                return

            t1 = time.time()
            while not _STOP:
                if iface.recv_begin():
                    break
                if iface.get_finished():
                    print("[agent] ns-3 signaled finished while waiting next obs. Exiting.")
                    return
                if (time.time() - t1) > float(args.turn_timeout):
                    print("[agent] timeout on PyRecvBegin (waiting next obs)")
                    return
                time.sleep(float(args.poll_interval))

    except Exception as e:
        import traceback

        print("[agent] EXCEPTION:", repr(e))
        traceback.print_exc()
        raise
    finally:
        print(f"[agent] stopped at step={step} time={_now()}")


if __name__ == "__main__":
    main()