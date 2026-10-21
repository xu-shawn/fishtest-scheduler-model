import os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import common as C, fishsim as fs, policies as P, experiment as E

def job(pname):
    runs, workers, tr, cap, meas = C.setup()
    t = time.time()
    tr2 = fs.closed_loop(tr, 3.0)
    res, samples, run_cores = C.run(P.BY_NAME[pname], tp=E.TP, start_delay_h=0.5, kappa=1.0, trace=tr2)
    pickle.dump((pname, "closed", res, samples, run_cores), open(os.path.join(E.OUT, f"{pname}__closed.pkl"), "wb"))
    return pname, time.time() - t, int((tr2.df.parent >= 0).sum())

if __name__ == "__main__":
    names = sys.argv[1:]
    with Pool(len(names)) as p:
        for n, dt, k in p.imap_unordered(job, names):
            print(f"done {n} in {dt:.0f}s ({k} triggered runs)", flush=True)
