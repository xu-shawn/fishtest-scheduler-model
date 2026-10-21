import os, sys, pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
import common as C, policies as P

def job(args):
    name, kw = args
    res, samples, _ = C.run(P.Current(), **kw)
    return name, res

def report(name, res, meas):
    m = res.iloc[meas]
    ratio = m.sim_dur_h / m.orig_dur_h
    s = m[m["sub"] == "STC"]
    ra, sa = C.real_active_by_day(res), C.sim_active_by_day(res)
    w = slice(40, 127)  # measurement window in days since Jun 1
    short = s[s.orig_dur_h < 3]
    print(f"{name:28s} STC med real {s.orig_dur_h.median():5.1f} sim {s.sim_dur_h.median():5.1f} | "
          f"STC ratio med {(s.sim_dur_h/s.orig_dur_h).median():.2f} short(<3h) {(short.sim_dur_h/short.orig_dur_h).median():.2f} | "
          f"active real {ra[w].mean():5.1f} sim {sa[w].mean():5.1f}")
    print("   by class (median sim/real):", m.assign(r=ratio).groupby("sub").r.median().round(2).to_dict())

if __name__ == "__main__":
    runs, workers, tr, cap, meas = C.setup()
    stage = sys.argv[1]
    if stage == "delay":
        jobs = [(f"delay={d}", dict(start_delay_h=d)) for d in (0.0, 0.5, 1.0)]
        with Pool(3) as p:
            for name, res in p.map(job, jobs):
                report(name, res, meas)
                if name == "delay=0.5":
                    res.to_pickle(os.path.join(C.HERE, "cal_delay05.pkl"))

def fit_tp(res, meas, tp_prev):
    m = res.iloc[meas]
    m = m[(m.orig_dur_h > 2) & (m.sim_dur_h > 0)]
    ratio = m.sim_dur_h / m.orig_dur_h
    r_stc = ratio[m["sub"] == "STC"].median()
    g = (ratio / r_stc)[m["sub"] != "STC"].groupby([m.username, m["sub"]]).median()
    d = res
    key = list(zip(d.username, d["sub"]))
    mult = np.array([g.get(k, 1.0) for k in key])
    tp = np.clip(tp_prev * mult, 25, 100)  # throughput is only lowered, never raised past 100
    tp = np.where(d.open, d.tp, tp)  # live runs: real throughput from the API
    tp = np.where(d["sub"] == "STC", 100.0, tp)
    return tp, g

if __name__ == "__main__" and sys.argv[1] == "tp":
    tp = tr.df.tp.values.astype(float)
    res = pd.read_pickle(os.path.join(C.HERE, "cal_delay05.pkl"))
    for it in range(3):
        tp, g = fit_tp(res, meas, tp)
        res, _, _ = C.run(P.Current(), tp=tp, start_delay_h=0.5)
        report(f"tp iter {it}", res, meas)
    np.save(os.path.join(C.HERE, "tp_calibrated.npy"), np.asarray(tp, dtype=float), allow_pickle=False)
    t = tr.df.assign(tp=tp)
    print(t[t["sub"] != "STC"].groupby("sub").tp.describe()[["count", "mean", "25%", "50%", "75%"]].round(0))

if __name__ == "__main__" and sys.argv[1] == "kappa":
    tp = np.load(os.path.join(C.HERE, "tp_calibrated.npy"), allow_pickle=False)
    jobs = [(f"kappa={k}", dict(tp=tp, start_delay_h=0.5, kappa=k)) for k in (0.88, 0.92, 0.96)]
    with Pool(3) as p:
        for name, res in p.map(job, jobs):
            report(name, res, meas)
            res.to_pickle(os.path.join(C.HERE, f"cal_{name}.pkl"))
