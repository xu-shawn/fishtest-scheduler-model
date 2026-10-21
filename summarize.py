import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "results_rust", "replicates.csv"))
keys = ["STC med", "STC passed med", "STC->LTC pass med", "LTC med", "SMP med", "heavy med", "all mean", "active win", "cores/STC now", "top test p95 now"]
base = df[df.policy == "current"].set_index(["scenario", "seed"])
rows = []
for (scen, pol), g in df.groupby(["scenario", "policy"]):
    g = g.set_index(["scenario", "seed"])
    rel = g[keys] / base.loc[g.index, keys] - 1
    r = {"scenario": scen, "policy": pol}
    for k in keys:
        x = rel[k] * 100
        r[k] = f"{x.median():+4.0f} [{x.quantile(.1):+.0f},{x.quantile(.9):+.0f}]"
    rows.append(r)
out = pd.DataFrame(rows)
abs_ = df.groupby(["scenario", "policy"])[keys].median()
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 20); pd.set_option("display.max_rows", 200)
scen = sys.argv[1] if len(sys.argv) > 1 else "open"
order = df.policy.unique()
print(f"=== {scen}: median absolute over seeds (h)")
print(abs_.loc[scen].reindex(order).round(1).to_string())
print(f"\n=== {scen}: % change vs current, median [p10,p90] over 10 seeds")
print(out[out.scenario == scen].set_index("policy").reindex(order)[keys].to_string())
