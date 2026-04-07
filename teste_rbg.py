import re
import pandas as pd
from pathlib import Path

AGENT_LOG = Path("/home/diego/ns-3-dev/agent_run_ok.log")
SLOT_CSV = Path("/home/diego/ns-3-dev/default-slot-log.csv")

agent_pat = re.compile(
    r"\[agent\] step=(?P<step>\d+)\s+slot=(?P<slot>\d+)\s+beam=(?P<beam>\d+)\s+"
    r"n=(?P<n>\d+)\s+total=(?P<total>\d+)\s+sum_alloc=(?P<sum_alloc>\d+)\s+alpha=(?P<alpha>[0-9.]+)"
)

rows = []
for line in AGENT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = agent_pat.search(line)
    if m:
        rows.append(
            {
                "step": int(m.group("step")),
                "slot": int(m.group("slot")),
                "beam_hash_agent": str(m.group("beam")).strip(),
                "n_agent": int(m.group("n")),
                "total_agent": int(m.group("total")),
                "sum_alloc_agent": int(m.group("sum_alloc")),
                "alpha_agent": float(m.group("alpha")),
            }
        )

df_agent = pd.DataFrame(rows)
if df_agent.empty:
    raise RuntimeError("Nenhuma linha do agent foi encontrada em agent_run_ok.log")

df_slot = pd.read_csv(SLOT_CSV)
df_slot = df_slot.rename(columns={c: c.strip() for c in df_slot.columns})

required = {"slot", "beam_id", "alloc_rbg", "alpha", "rnti"}
missing = required - set(df_slot.columns)
if missing:
    raise RuntimeError(f"Colunas ausentes no CSV: {missing}")

# Como há um único beam no cenário, agregamos por slot
df_slot_agg = (
    df_slot.groupby(["slot"], as_index=False)
    .agg(
        beam_id_csv=("beam_id", "first"),
        n_csv=("rnti", lambda s: int((pd.Series(s) != 0).sum())),
        sum_alloc_csv=("alloc_rbg", "sum"),
        alpha_csv=("alpha", "mean"),
    )
)

df_cmp = df_agent.merge(
    df_slot_agg,
    on="slot",
    how="left",
)

df_cmp["ok_sum_alloc"] = df_cmp["sum_alloc_agent"] == df_cmp["sum_alloc_csv"]
df_cmp["ok_n"] = df_cmp["n_agent"] == df_cmp["n_csv"]
df_cmp["alpha_diff"] = (df_cmp["alpha_agent"] - df_cmp["alpha_csv"]).abs()
df_cmp["ok_alpha"] = df_cmp["alpha_diff"] < 1e-3

print("\nResumo geral")
print(f"linhas do agent: {len(df_agent)}")
print(f"slots comparados: {len(df_cmp)}")
print(f"soma alloc OK: {df_cmp['ok_sum_alloc'].sum()}/{len(df_cmp)}")
print(f"n UEs OK:      {df_cmp['ok_n'].sum()}/{len(df_cmp)}")
print(f"alpha OK:      {df_cmp['ok_alpha'].sum()}/{len(df_cmp)}")

print("\nPrimeiras linhas")
print(
    df_cmp[
        [
            "step", "slot", "beam_hash_agent", "beam_id_csv",
            "n_agent", "n_csv",
            "sum_alloc_agent", "sum_alloc_csv",
            "alpha_agent", "alpha_csv", "alpha_diff",
            "ok_sum_alloc", "ok_n", "ok_alpha",
        ]
    ].head(20).to_string(index=False)
)

print("\nLinhas com divergência")
df_bad = df_cmp[~(df_cmp["ok_sum_alloc"] & df_cmp["ok_alpha"])]
if df_bad.empty:
    print("Nenhuma divergência encontrada.")
else:
    print(
        df_bad[
            [
                "step", "slot", "beam_hash_agent", "beam_id_csv",
                "n_agent", "n_csv",
                "sum_alloc_agent", "sum_alloc_csv",
                "alpha_agent", "alpha_csv", "alpha_diff",
                "ok_sum_alloc", "ok_n", "ok_alpha",
            ]
        ].head(50).to_string(index=False)
    )
