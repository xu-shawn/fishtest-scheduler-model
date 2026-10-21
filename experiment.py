"""Run every policy on the calibrated replay and score it."""
import os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
import common as C, fishsim as fs, policies as P

OUT = os.path.join(C.HERE, "results")
TP = np.load(os.path.join(C.HERE, "tp_calibrated.npy"), allow_pickle=False)
W0, W1 = 40 * fs.DAY, 127 * fs.DAY   # measurement window: Jul 11 -> Oct 6
NOW = 141.95 * fs.DAY
RECENT = (127 * fs.DAY, NOW)          # Oct 6 -> Oct 20, the present crowding


def job(args):
    pname, kappa = args
    t = time.time()
    res, samples, run_cores = C.run(P.BY_NAME[pname], tp=TP, start_delay_h=0.5, kappa=kappa)
    out = (pname, kappa, res, samples, run_cores)
    pickle.dump(out, open(os.path.join(OUT, f"{pname}__k{kappa}.pkl"), "wb"))
    return pname, kappa, time.time() - t


def q(x, p):
    x = x.dropna()
    return float(np.quantile(x, p)) if len(x) else np.nan


def metrics(res, samples, run_cores, meas, win=(W0, W1), recent=RECENT):
    m = res.iloc[meas]
    sub = m["sub"]
    lat = m.sim_dur_h
    out = {}
    stc = m[sub == "STC"]
    out["STC med"] = q(stc.sim_dur_h, .5)
    out["STC mean"] = stc.sim_dur_h.mean()
    out["STC p90"] = q(stc.sim_dur_h, .9)
    good = stc[(stc.kind == "gain") & (stc.outcome == "passed")]
    out["STC passed med"] = q(good.sim_dur_h, .5)
    for k in ["LTC", "SMP-STC", "SMP-LTC", "VVLTC", "VLTC", "SPSA", "Progtest"]:
        out[f"{k} med"] = q(m[sub == k].sim_dur_h, .5)
    out["SMP med"] = q(m[sub.isin(["SMP-STC", "SMP-LTC"])].sim_dur_h, .5)
    out["heavy med"] = q(m[sub.isin(["VVLTC", "VLTC", "SMP-LTC"])].sim_dur_h, .5)
    out["heavy mean"] = m[sub.isin(["VVLTC", "VLTC", "SMP-LTC"])].sim_dur_h.mean()
    # STC -> LTC pipelines that ended in a passed LTC: from STC submission to LTC result
    kids = res.iloc[meas]
    kids = kids[(kids["sub"] == "LTC") & (kids.parent >= 0) & (kids.outcome == "passed")]
    par = res.loc[kids.parent.values]
    ok = (par["sub"] == "STC").values
    pipe = (kids.sim_finish.values - par.sim_submit.values)[ok] / fs.HOUR
    out["STC->LTC pass med"] = float(np.median(pipe)) if len(pipe) else np.nan
    out["all mean"] = lat.mean()
    out["unfinished"] = int(m.sim_finish.isna().sum())
    # time series in the measurement window and in the last two weeks
    heavy = res.tcr.values > 6.5
    is_stc = (res["sub"] == "STC").values
    for label, (a, b) in {"win": win, "now": recent}.items():
        sel = [(s, rc) for s, rc in zip(samples.itertuples(), run_cores) if a <= s.t < b]
        if not sel:
            continue
        act = np.mean([s.active for s, _ in sel])
        tot = np.array([sum(rc.values()) or 1 for _, rc in sel])
        hv = np.array([sum(c for i, c in rc.items() if heavy[i]) for _, rc in sel]) / tot
        top = np.array([max(rc.values()) if rc else 0 for _, rc in sel]) / tot
        stc_c = [np.mean([c for i, c in rc.items() if is_stc[i]]) for _, rc in sel
                 if any(is_stc[i] for i in rc)]
        out[f"active {label}"] = act
        out[f"heavy share {label}"] = hv.mean()
        out[f"heavy share p95 {label}"] = float(np.quantile(hv, .95))
        out[f"top test p95 {label}"] = float(np.quantile(top, .95))
        out[f"cores/STC {label}"] = float(np.mean(stc_c))
    return out


def load(pname, kappa):
    return pickle.load(open(os.path.join(OUT, f"{pname}__k{kappa}.pkl"), "rb"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    kappas = [1.0, 0.985]
    names = sys.argv[1:] or [p.name for p in P.ALL]
    jobs = [(n, k) for k in kappas for n in names]
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        for pname, kappa, dt in pool.imap_unordered(job, jobs):
            print(f"done {pname} k={kappa} in {dt:.0f}s", flush=True)
