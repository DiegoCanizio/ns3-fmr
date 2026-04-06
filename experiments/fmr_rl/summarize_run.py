import sys
from pathlib import Path
import pandas as pd
import re

run_dir = Path(sys.argv[1])
agent_log = run_dir / "agent.log"
ns3_log = run_dir / "ns3.log"
slot_csv = run_dir / "default-slot-log.csv"

df = pd.read_csv(slot_csv)
df = df.rename(columns={c: c.strip() for c in df.columns})

slots = df["slot"].nunique()
ues_mean = df.groupby("slot")["rnti"].count().mean()
alloc_sum_per_slot = df.groupby("slot")["alloc_rbg"].sum()
alpha_mean = df["alpha"].mean()
alpha_std = df["alpha"].std()

result_line = None
for line in ns3_log.read_text(encoding="utf-8", errors="ignore").splitlines():
    if "[RESULT]" in line:
        result_line = line

print("Resumo do run")
print(f"run_dir: {run_dir}")
print(f"slots únicos: {slots}")
print(f"UEs médias por slot: {ues_mean:.2f}")
print(f"soma média de alloc_rbg por slot: {alloc_sum_per_slot.mean():.2f}")
print(f"alpha médio: {alpha_mean:.6f}")
print(f"alpha std:   {alpha_std:.6f}")
print(f"result line: {result_line}")
