import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
df = pd.read_csv(f"{DATA}/finished_runs.csv", parse_dates=["finished", "started"])
df["dur_h"] = (df.finished - df.started).dt.total_seconds() / 3600
df["month"] = df.finished.dt.to_period("M")
print("range", df.started.min(), df.finished.max(), len(df))
print(df.groupby("type").agg(n=("id", "size"), ch=("core_hours", "sum"),
                             med_games=("games", "median"), med_dur=("dur_h", "median")))
print()
piv = df.pivot_table(index="month", columns="type", values="core_hours", aggfunc="sum").fillna(0)
share = piv.div(piv.sum(axis=1), axis=0).round(3)
print("core-hour share by month (by finish month)")
print(share)
print(piv.sum(axis=1).round(0))
print()
cnt = df.pivot_table(index="month", columns="type", values="id", aggfunc="count").fillna(0)
print("count by month")
print(cnt)
print()
print("outcomes by type")
print(pd.crosstab(df.type, df.outcome))
print()
print("median duration (h) by month and type")
print(df.pivot_table(index="month", columns="type", values="dur_h", aggfunc="median").round(1))
print()
print("threads/tc combos for SMP/VVLTC/LTC")
print(df[df.type.isin(["SMP", "VVLTC", "LTC", "VLTC"])].groupby(["type", "tc", "threads"]).size())
