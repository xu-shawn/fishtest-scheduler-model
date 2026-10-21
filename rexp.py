"""Replicated policy comparison on the Rust engine.

For each scenario (open loop at real capacity, open loop crowded, closed loop) and
each seed (a different draw of the startup delays), every policy runs on the same
trace, so differences against `current` are paired within a seed."""
import os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
import common as C, fishsim as fs, experiment as E, rsim as R

POLICIES = (["current", "pre_june", "additive_bonus", "tc_cap6", "tc_cap16", "spec_malus0.5",
             "heavy_cap25", "test_cap10"]
            + ["cost_count_p0.3", "cost_count_p0.4", "cost_count_p0.5", "cost_count_p0.65", "cost_count",
               "cost_count_p0.3+add", "cost_count_p0.5+add"]
            + [f"user_cap{s}" for s in (1, 2, 3, 4, 6)]
            + [f"user_cap{s}+add" for s in (2, 3, 4, 5, 6, 8, 12)])
SCENARIOS = {"open": (1.0, False), "crowded": (0.985, False), "closed": (1.0, True)}
OUT = os.path.join(C.HERE, "results_rust")


def score(args):
    scen, seed, pol, path, meas = args
    runs, workers, tr, cap, _ = C.setup()
    t = fs.closed_loop(tr, 3.0) if SCENARIOS[scen][1] else tr
    res, samples, rc = R.load(path, t)
    os.remove(path)
    return {"scenario": scen, "seed": seed, "policy": pol, **E.metrics(res, samples, rc, meas)}


def main(seeds):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(R.WORK, exist_ok=True)
    runs, workers, tr, cap, meas = C.setup()
    closed = fs.closed_loop(tr, 3.0)
    rows = []
    for scen, (kappa, is_closed) in SCENARIOS.items():
        t0 = time.time()
        jobs, todo = [], []
        for seed in seeds:
            tf = os.path.join(R.WORK, f"{scen}_s{seed}.trace")
            trace = closed if is_closed else tr
            # seed 0 is the delay draw used everywhere else; others perturb it
            R.export(tf, trace, workers, cap, meas, tp=E.TP, start_delay_h=0.5, seed=12345 + seed)
            for pol in POLICIES:
                out = os.path.join(R.WORK, f"{scen}_s{seed}_{pol}.out")
                jobs.append((tf, out, pol, kappa))
                todo.append((scen, seed, pol, out, meas))
        R.run_jobs(jobs)
        with Pool(10) as p:
            rows += p.map(score, todo)
        print(f"{scen}: {len(jobs)} sims scored in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "replicates.csv"), index=False)
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(list(range(n)))
