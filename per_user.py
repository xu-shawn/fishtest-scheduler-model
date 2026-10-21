"""Who gains and who loses: per-user latency under candidate policies vs current."""
import sys
import numpy as np, pandas as pd
import common as C, experiment as E

runs, workers, tr, cap, meas = C.setup()
kappa = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
names = sys.argv[2:] or ["cost_count_p0.5", "user_cap6", "user_cap3+add"]
base = E.load("current", kappa)[2].iloc[meas]
m = base.assign(ch=base.gd * base.games * base.threads / 3600)
info = m.groupby("username").agg(runs=("id", "size"), core_h=("ch", "sum"),
                                 heavy_share=("ch", lambda x: x[m.loc[x.index, "tcr"] > 6.5].sum() / x.sum()))
info["lat_current"] = m.groupby("username").sim_dur_h.median()
# total user "wait": sum of latencies (hours of results outstanding)
info["sum_lat_current"] = m.groupby("username").sim_dur_h.sum()
for n in names:
    r = E.load(n, kappa)[2].iloc[meas]
    info[f"lat {n}"] = r.groupby("username").sim_dur_h.median()
    info[f"Δsum {n} %"] = (r.groupby("username").sim_dur_h.sum() / info.sum_lat_current - 1) * 100
info = info[info.runs >= 8].sort_values("core_h", ascending=False)
pd.set_option("display.width", 250)
print(info.drop(columns=["sum_lat_current"]).round(1).to_string())
