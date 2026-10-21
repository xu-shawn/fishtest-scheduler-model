"""Bridge to the Rust engine (rsim/): export traces, run job batches, load results
into the same frames fishsim.results_frame / Sim.samples produce."""
import os, subprocess, tempfile
import numpy as np, pandas as pd
import fishsim as fs

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "rsim", "target", "release", "fishsim-rs")
WORK = os.path.join(HERE, "rwork")


def cap_array(capacity: dict) -> np.ndarray:
    arr, cur = [], capacity[min(capacity)]
    for k in range(max(capacity) + 1):
        cur = capacity.get(k, cur)
        arr.append(cur)
    return np.array(arr)


def export(path, trace, workers, capacity, measured, tp=None, start_delay_h=0.5, seed=12345):
    d = trace.df
    n = len(d)
    tp = d.tp.values if tp is None else np.asarray(tp, dtype=float)
    delay = (np.random.default_rng(seed).exponential(start_delay_h * fs.HOUR, n)
             if start_delay_h > 0 else np.zeros(n))
    cols = np.column_stack([
        tp, d.prio.values, d.tcr.values, d.gd.values, d.threads.values, d.need.values, d.n0.values,
        d.fixed.values, d.sprt.values, (d.outcome == "passed").values, (d.outcome == "rejected").values,
        d.elo0.fillna(0).values, d.user.values, d.submit.values, (d.gate == "speculative").values,
        np.where(d.sprt.values, 16, 2), d.parent.values, d.gap.values, delay,
    ]).astype(float)
    cap = cap_array(capacity)
    with open(path, "w") as f:
        f.write(f"{n} {len(workers)} {len(cap)} {len(measured)} {trace.t_end!r}\n")
        ints = {4, 7, 8, 9, 10, 12, 14, 16}
        for row in cols:
            f.write(" ".join(str(int(v)) if j in ints else repr(float(v)) for j, v in enumerate(row)) + "\n")
        for c, s in zip(workers.cores.values, workers.speed.values):
            f.write(f"{int(c)} {float(s)!r}\n")
        f.write(" ".join(repr(float(x)) for x in cap) + "\n")
        f.write(" ".join(str(int(i)) for i in measured) + "\n")


def run_jobs(jobs, threads=None, quiet=True):
    """jobs: list of (trace_file, out_file, policy_name, kappa)."""
    os.makedirs(WORK, exist_ok=True)
    fd, jf = tempfile.mkstemp(dir=WORK, suffix=".jobs")
    with os.fdopen(fd, "w") as f:
        for j in jobs:
            f.write(" ".join(str(x) for x in j) + "\n")
    cmd = [BIN, jf] + ([str(threads)] if threads else [])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    os.remove(jf)
    if not quiet:
        print(out.stdout, end="")
    return out.stdout


def load(out_file, trace):
    with open(out_file) as f:
        n, s = map(int, f.readline().split())
        per = np.array([list(map(float, f.readline().split())) for _ in range(n)])
        rows, run_cores = [], []
        for _ in range(s):
            p = f.readline().split()
            rows.append({"t": float(p[0]), "active": int(p[1]), "running": int(p[2]), "cores": float(p[3])})
            k = int(p[4])
            run_cores.append({int(p[5 + 2 * j]): float(p[6 + 2 * j]) for j in range(k)})
    d = trace.df.copy()
    d["sim_submit"], d["sim_arrive"], d["sim_first"], d["sim_finish"] = per[:, 0], per[:, 1], per[:, 2], per[:, 3]
    d["sim_dur_h"] = (d.sim_finish - d.sim_submit) / fs.HOUR
    d["sim_core_h"] = per[:, 4] / fs.HOUR
    return d, pd.DataFrame(rows), run_cores
