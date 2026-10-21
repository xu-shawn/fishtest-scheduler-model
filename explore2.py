import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
df = pd.read_csv(f"{DATA}/finished_runs.csv", parse_dates=["finished", "started"])
df["dur_h"] = (df.finished - df.started).dt.total_seconds() / 3600
end = df.finished.max()
recent = df[df.started >= end - pd.Timedelta(days=90)].copy()
print("last 90 days:", len(recent))
t = recent.pivot_table(index="type", columns="outcome", values="core_hours", aggfunc="sum").fillna(0)
print((t / t.values.sum()).round(3))
print()

# stopped-run game counts
for ty in ["STC", "LTC", "SMP"]:
    s = recent[(recent.type == ty) & (recent.outcome == "stopped")]
    print(ty, "stopped games quantiles", s.games.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values,
          "zero-game", (s.games == 0).sum(), "of", len(s))

# speculative detection: LTC/SMP/VLTC/VVLTC SPRT runs; was there a passed STC (same user, branch)
# that finished before this run started? Also: was there an STC at all, started before?
df = df.sort_values("started")
stc = df[df.type == "STC"]
stc_by = {k: g for k, g in stc.groupby(["username", "branch"])}

def classify(row):
    g = stc_by.get((row.username, row.branch))
    if g is None:
        return "no_stc"
    before = g[g.started <= row.started + pd.Timedelta(minutes=5)]
    if len(before) == 0:
        return "no_stc"
    passed_before = before[(before.outcome == "passed") & (before.finished <= row.started + pd.Timedelta(minutes=30))]
    if len(passed_before):
        return "after_stc_pass"
    running = before[before.finished > row.started]
    if len(running):
        return "parallel_stc"
    return "after_stc_nonpass"

big = df[df.type.isin(["LTC", "SMP", "VLTC", "VVLTC"]) & df.elo0.notna()].copy()
big["gate"] = big.apply(classify, axis=1)
big["kind"] = np.where(big.elo0 + big.elo1 < 0, "simp", "gain")
for window in [365, 90, 30]:
    b = big[big.started >= end - pd.Timedelta(days=window)]
    print(f"\n=== last {window} days, non-STC SPRT runs")
    agg = b.groupby(["type", "kind", "gate"]).agg(n=("id", "size"), passed=("outcome", lambda s: (s == "passed").sum()),
                                                   stopped=("outcome", lambda s: (s == "stopped").sum()),
                                                   ch=("core_hours", "sum"))
    tot = df[df.started >= end - pd.Timedelta(days=window)].core_hours.sum()
    agg["ch_share_all"] = (agg.ch / tot).round(3)
    print(agg)

# The same test re-submitted: duplicates of the same user+branch+tc running in parallel
