import os, sys, time
import numpy as np, pandas as pd
import fishsim as fs, policies as P

import common as C
D = sys.argv[1] if len(sys.argv) > 1 else C.DATA
API = sys.argv[2] if len(sys.argv) > 2 else C.API
SIM_START = "2026-06-01T00:00:00+00:00"; NOW = "2026-10-20T22:45:00+00:00"
M0, M1 = pd.Timestamp("2026-07-11T00:00:00+00:00"), pd.Timestamp("2026-10-06T00:00:00+00:00")
runs = fs.load_runs(D, API)
workers = fs.load_workers(D)
tr = fs.build_trace(runs, SIM_START, NOW, workers.cores.max())
cap = fs.capacity_profile(runs, tr.t0, pd.Timestamp(NOW))
d = tr.df
meas = np.nonzero(((d.started >= M0) & (d.started < M1) & (d.games > 0) & ~d.open).values)[0]
print("trace runs", len(d), "measured", len(meas), "dependent", (d.parent >= 0).sum(), "open", d.open.sum())
print("capacity days", len(cap), "mean", np.mean(list(cap.values())), "nominal", (workers.cores*workers.speed).sum())
t = time.time()
sim = fs.Sim(tr, workers, P.Current(), cap).run(measured=meas)
print("sim time", round(time.time() - t, 1), "s; sim end day", sim.t / fs.DAY)
res = fs.results_frame(sim)
res.to_pickle(os.path.join(C.HERE, "val_current.pkl"))
m = res.iloc[meas]
m = m.assign(err=np.log(m.sim_dur_h.clip(0.1) / m.orig_dur_h.clip(0.1)))
print(m.groupby("sub").agg(n=("id", "size"), real_med=("orig_dur_h", "median"), sim_med=("sim_dur_h", "median"),
                           real_mean=("orig_dur_h", "mean"), sim_mean=("sim_dur_h", "mean"),
                           real_p90=("orig_dur_h", lambda x: x.quantile(.9)), sim_p90=("sim_dur_h", lambda x: x.quantile(.9)),
                           med_log_err=("err", "median")).round(1))
ok = m[(m.orig_dur_h > 0.5)]
print("corr log durations", np.corrcoef(np.log(ok.orig_dur_h), np.log(ok.sim_dur_h.clip(0.1)))[0, 1])
s = pd.DataFrame([{k: v for k, v in r.items() if k != "run_cores"} for r in sim.samples])
s["day"] = (s.t // fs.DAY).astype(int)
print(s.groupby(s.day // 7).agg(active=("active", "mean"), running=("running", "mean"), cores=("cores", "mean")).round(1).to_string())
