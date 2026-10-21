"""Shared setup: the trace (Jun 1 -> Oct 20 2026), workers, capacity, measured runs."""
import os, pickle
import numpy as np, pandas as pd
import fishsim as fs

HERE = os.path.dirname(os.path.abspath(__file__))
# Snapshot of xu-shawn/fishtest_track data/ and fishtest /api/active_runs, 20 Oct 2026 22:45 UTC
DATA = os.path.join(HERE, "data", "fishtest_track")
API = os.path.join(HERE, "data", "active_runs.json")
SIM_START = "2026-06-01T00:00:00+00:00"
NOW = "2026-10-20T22:45:00+00:00"
M0 = pd.Timestamp("2026-07-11T00:00:00+00:00")
M1 = pd.Timestamp("2026-10-06T00:00:00+00:00")
CACHE = os.path.join(HERE, "setup.pkl")


def setup():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    runs = fs.load_runs(DATA, API)
    workers = fs.load_workers(DATA)
    tr = fs.build_trace(runs, SIM_START, NOW, workers.cores.max())
    cap = fs.capacity_profile(runs, tr.t0, pd.Timestamp(NOW))
    d = tr.df
    meas = np.nonzero(((d.started >= M0) & (d.started < M1) & (d.games > 0) & ~d.open).values)[0]
    out = (runs, workers, tr, cap, meas)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def run(policy, tp=None, start_delay_h=0.75, kappa=1.0, trace=None):
    runs, workers, tr, cap, meas = setup()
    if trace is not None:
        tr = trace
    if tp is not None:
        tr = fs.Trace(tr.df.assign(tp=tp), tr.t0, tr.t_end, tr.children)
    sim = fs.Sim(tr, workers, policy, cap, start_delay_h=start_delay_h, kappa=kappa)
    sim.run(measured=meas)
    res = fs.results_frame(sim)
    samples = pd.DataFrame([{k: v for k, v in r.items() if k != "run_cores"} for r in sim.samples])
    return res, samples, [r["run_cores"] for r in sim.samples]


def real_active_by_day(res):
    days = np.arange(0, int(res.submit.max() // fs.DAY) + 1)
    of = res.orig_finish.fillna(1e12)
    return np.array([((res.submit <= d * fs.DAY) & (of > d * fs.DAY)).sum() for d in days])


def sim_active_by_day(res):
    days = np.arange(0, int(res.submit.max() // fs.DAY) + 1)
    sf = res.sim_finish.fillna(1e12)
    return np.array([((res.sim_submit <= d * fs.DAY) & (sf > d * fs.DAY)).sum() for d in days])
