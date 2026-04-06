#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import ns3ai_gym_msg_py

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def try_connect(shm, segment, cpp2py, py2cpp, lock, is_creator, b1, b2, wait_s):
    print(f"[probe] trying: creator={is_creator} b1={b1} b2={b2} at {now()}", flush=True)
    iface = ns3ai_gym_msg_py.Ns3AiMsgInterfaceImpl(
        bool(is_creator), bool(b1), bool(b2),
        int(shm),
        str(segment), str(cpp2py), str(py2cpp), str(lock)
    )
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if iface.PyRecvBegin():
            obs = iface.GetCpp2PyStruct()
            # imprime só campos que certamente existem no seu obs atual
            slot = int(getattr(obs, "slot", -1))
            beam = int(getattr(obs, "beam_hash", -1))
            n = int(getattr(obs, "num_ues", -1))
            trbg = int(getattr(obs, "total_rbg", -1))
            print(f"[probe] GOT_OBS: slot={slot} beam={beam} n={n} total_rbg={trbg}", flush=True)
            iface.PyRecvEnd()

            # tenta responder algo mínimo
            if iface.PySendBegin():
                act = iface.GetPy2CppStruct()
                if hasattr(act, "alpha_next"):
                    act.alpha_next = 0.7
                if hasattr(act, "num_ues"):
                    act.num_ues = 0
                iface.PySendEnd()
                print("[probe] SENT_ACT", flush=True)
            else:
                print("[probe] PySendBegin returned False", flush=True)
            return True

        time.sleep(0.02)

    print("[probe] TIMEOUT waiting obs", flush=True)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="ns3ai_fmr")
    ap.add_argument("--cpp2py", default="fmr_cpp2py")
    ap.add_argument("--py2cpp", default="fmr_py2cpp")
    ap.add_argument("--lock", default="fmr_lock")
    ap.add_argument("--shm", type=int, default=4096)
    ap.add_argument("--wait", type=float, default=5.0)
    ap.add_argument("--is-creator", type=int, default=0)
    args = ap.parse_args()

    combos = [(0,0),(1,0),(0,1),(1,1)]
    ok_any = False
    for b1,b2 in combos:
        try:
            ok = try_connect(args.shm, args.segment, args.cpp2py, args.py2cpp, args.lock,
                             bool(args.is_creator), b1, b2, float(args.wait))
            ok_any = ok_any or ok
        except Exception as e:
            print(f"[probe] EXCEPTION: creator={args.is_creator} b1={b1} b2={b2} err={e}", flush=True)

    print(f"[probe] DONE ok_any={ok_any}", flush=True)

if __name__ == "__main__":
    main()
