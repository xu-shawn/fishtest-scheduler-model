"""Robustness across other periods: rebuild the replay for earlier windows of the
year, recalibrate throughput habits there, and compare the same policies."""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import common as C, fishsim as fs, experiment as E, rsim as R

WINDOWS = {  # sim start, measured from, measured to, last submission replayed
    "Dec-Jan": ("2025-11-01", "2025-12-01", "2026-02-01", "2026-03-01"),
    "Mar-Apr": ("2026-02-01", "2026-03-01", "2026-05-01", "2026-06-01"),
    "May-Jul": ("2026-04-15", "2026-05-15", "2026-07-10", "2026-08-10"),
}
POLICIES = ["current", "pre_june", "additive_bonus", "spec_malus0.5", "tc_cap6", "cost_count_p0.5",
            "cost_count_p0.5+add", "user_cap3", "user_cap3+add", "user_cap4+add", "user_cap6+add"]
KEYS = ["STC med", "STC passed med", "STC->LTC pass med", "LTC med", "SMP med", "heavy med", "all mean", "active win"]


def ts(s):
    return pd.Timestamp(s + "T00:00:00+00:00")


def build(name):
    s0, m0, m1, end = WINDOWS[name]
    runs = fs.load_runs(C.DATA, C.API)
    workers = fs.load_workers(C.DATA)
    tr = fs.build_trace(runs, s0 + "T00:00:00+00:00", end + "T00:00:00+00:00", workers.cores.max())
    cap = fs.capacity_profile(runs, tr.t0, ts(end) + pd.Timedelta(days=60))
    d = tr.df
    meas = np.nonzero(((d.started >= ts(m0)) & (d.started < ts(m1)) & (d.games > 0) & ~d.open).values)[0]
    return runs, workers, tr, cap, meas


def fit_tp(res, meas, tp_prev):
    m = res.iloc[meas]
    m = m[(m.orig_dur_h > 2) & (m.sim_dur_h > 0)]
    ratio = m.sim_dur_h / m.orig_dur_h
    r_stc = ratio[m["sub"] == "STC"].median()
    g = (ratio / r_stc)[m["sub"] != "STC"].groupby([m.username, m["sub"]]).median()
    mult = np.array([g.get(k, 1.0) for k in zip(res.username, res["sub"])])
    tp = np.clip(tp_prev * mult, 25, 100)
    return np.where(res["sub"] == "STC", 100.0, tp)


def score(args):
    name, scen, seed, pol, path = args
    runs, workers, tr, cap, meas = build_cached(name)
    t = fs.closed_loop(tr, 3.0) if scen == "closed" else tr
    res, samples, rc = R.load(path, t)
    os.remove(path)
    s0, m0, m1, _ = WINDOWS[name]
    win = ((ts(m0) - ts(s0)).total_seconds(), (ts(m1) - ts(s0)).total_seconds())
    m = E.metrics(res, samples, rc, meas, win=win, recent=win)
    return {"window": name, "scenario": scen, "seed": seed, "policy": pol, **{k: m[k] for k in KEYS}}


_cache = {}
def build_cached(name):
    if name not in _cache:
        _cache[name] = build(name)
    return _cache[name]


def run_window(name, seeds):
    runs, workers, tr, cap, meas = build_cached(name)
    os.makedirs(R.WORK, exist_ok=True)
    # The formula in force at the time: pre_june before 17 Jun 2026.
    hist = "pre_june" if ts(WINDOWS[name][2]) <= ts("2026-06-17") else "current"
    tp = tr.df.tp.values.astype(float)
    tf = os.path.join(R.WORK, f"win_{name}_cal.trace")
    for it in range(3):
        R.export(tf, tr, workers, cap, meas, tp=tp, start_delay_h=0.5)
        out = os.path.join(R.WORK, f"win_{name}_cal.out")
        R.run_jobs([(tf, out, hist, 1.0)])
        res, _, _ = R.load(out, tr)
        tp = fit_tp(res, meas, tp)
    # Pick the capacity factor at which the baseline is as crowded as reality was
    # (STC median ~0.9x real, between the two scenarios of the main window).
    R.export(tf, tr, workers, cap, meas, tp=tp, start_delay_h=0.5)
    grid = [1.0, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.95]
    R.run_jobs([(tf, os.path.join(R.WORK, f"win_{name}_k{k}.out"), hist, k) for k in grid])
    best, kappa = 1e9, 1.0
    for k in grid:
        res, _, _ = R.load(os.path.join(R.WORK, f"win_{name}_k{k}.out"), tr)
        m = res.iloc[meas]
        stc = m[m["sub"] == "STC"]
        r = stc.sim_dur_h.median() / stc.orig_dur_h.median()
        print(f"   {name} kappa {k}: STC median sim/real {r:.2f}  unfinished {int(m.sim_finish.isna().sum())}")
        if abs(np.log(r / 0.9)) < best:
            best, kappa = abs(np.log(r / 0.9)), k
    m = res.iloc[meas]
    print(f"{name}: {len(meas)} measured runs; baseline formula {hist}; chosen kappa {kappa}; heavy work share "
          f"{(m.gd*m.games*m.threads)[m.tcr>6.5].sum()/(m.gd*m.games*m.threads).sum():.2f}", flush=True)
    jobs, todo = [], []
    for scen in ["open", "closed"]:
        trace = fs.closed_loop(tr, 3.0) if scen == "closed" else tr
        for seed in seeds:
            tf = os.path.join(R.WORK, f"win_{name}_{scen}_s{seed}.trace")
            R.export(tf, trace, workers, cap, meas, tp=tp, start_delay_h=0.5, seed=12345 + seed)
            for pol in POLICIES:
                out = os.path.join(R.WORK, f"win_{name}_{scen}_s{seed}_{pol}.out")
                jobs.append((tf, out, pol, kappa))
                todo.append((name, scen, seed, pol, out))
    R.run_jobs(jobs)
    with Pool(10) as p:
        return p.map(score, todo)


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
    rows = []
    for name in WINDOWS:
        rows += run_window(name, seeds)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.HERE, "results_rust", "windows.csv"), index=False)
    base = df[df.policy == "current"].set_index(["window", "scenario", "seed"])
    out = []
    for (w, sc, pol), g in df.groupby(["window", "scenario", "policy"], sort=False):
        g = g.set_index(["window", "scenario", "seed"])
        rel = (g[KEYS] / base.loc[g.index, KEYS] - 1) * 100
        out.append({"window": w, "scenario": sc, "policy": pol, **rel.median().round(0).to_dict()})
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(pd.DataFrame(out).to_string(index=False))
