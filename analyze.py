"""Score every finished simulation and print comparison tables."""
import glob, os, pickle, sys
import numpy as np, pandas as pd
import common as C, experiment as E

runs, workers, tr, cap, meas = C.setup()
rows = []
for f in sorted(glob.glob(os.path.join(E.OUT, "*__k*.pkl"))):
    pname, kappa, res, samples, run_cores = pickle.load(open(f, "rb"))
    m = E.metrics(res, samples, run_cores, meas)
    rows.append({"policy": pname, "kappa": kappa, **m})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(E.OUT, "summary.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
cols = ["STC med", "STC mean", "STC p90", "STC passed med", "STC->LTC pass med", "LTC med",
        "SMP med", "VVLTC med", "heavy med", "heavy mean", "all mean", "active win", "cores/STC win",
        "heavy share p95 win", "top test p95 win", "cores/STC now", "heavy share now",
        "top test p95 now", "unfinished"]
for k in sorted(df.kappa.unique(), reverse=True):
    t = df[df.kappa == k].set_index("policy")[cols]
    base = t.loc["current"]
    print(f"\n=== kappa={k}: absolute (hours; shares as fractions)")
    print(t.round(2).to_string())
    rel = t.copy()
    for c in cols:
        if c not in ("unfinished",):
            rel[c] = t[c] / base[c] - 1
    print(f"\n=== kappa={k}: change vs current")
    print((rel[cols[:-1]] * 100).round(0).astype("Int64").to_string())
