"""Event-driven model of fishtest's task scheduler.

Workers request a task whenever their current one ends (or the run it belongs to
finishes).  The server picks a run exactly like rundb.sync_request_task: lowest
(-priority, adjusted_cores > 0, (adjusted_cores + max_threads / 2) / itp), skipping
runs the worker cannot play.  Tasks are sized to ~30 minutes of games.

The workload is a replay of real runs.  A scheduling policy cannot change how many
games a test needs (the SPRT sees the same game sequence however fast it is played),
so each run keeps its real game count and only its timing changes.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TASK_DURATION = 1800.0
ITP_REFRESH = 60.0
HOUR = 3600.0
DAY = 86400.0
LLR_HI_FRACTION = 0.30  # passed SPRTs spend ~the last 30% of their games above LLR 2.0


def game_duration(tc: str) -> float:
    """Port of fishtest's estimate_game_duration."""
    base, _, inc = tc.partition("+")
    inc = float(inc or 0)
    parts = base.split("/")
    num_moves = int(parts[0]) if len(parts) == 2 else 0
    clock = parts[-1].split(":")
    t = float(clock[0]) * 60 + float(clock[1]) if len(clock) == 2 else float(clock[0])
    if num_moves > 0:
        t *= 68 / num_moves
    return (t + inc * 68) * 2 * 0.92


STC_GD = game_duration("10+0.1")


def subtype(row) -> str:
    """Finer test class used for reporting (SMP split by base time)."""
    if row["type"] == "SMP":
        return "SMP-LTC" if game_duration(row["tc"]) >= game_duration("20+0.2") else "SMP-STC"
    return row["type"]


# --------------------------------------------------------------------------- trace

@dataclass
class Trace:
    df: pd.DataFrame
    t0: pd.Timestamp
    t_end: float  # seconds since t0 of the last real observation
    children: dict = field(default_factory=dict)  # parent index -> [child indices]


def load_runs(data_dir: str, api_json: str | None) -> pd.DataFrame:
    fin = pd.read_csv(f"{data_dir}/finished_runs.csv", parse_dates=["finished", "started"])
    act = pd.read_csv(f"{data_dir}/active_runs.csv", parse_dates=["started"])
    act["outcome"] = "active"
    act["finished"] = pd.NaT
    tp, prio = {}, {}
    if api_json:
        api = json.load(open(api_json))
        for rid, r in api.items():
            tp[rid] = r["args"].get("throughput", 100)
            prio[rid] = r["args"].get("priority", 0)
    act["tp"] = act.id.map(tp).fillna(100)
    act["prio"] = act.id.map(prio).fillna(0)
    fin["tp"] = 100.0
    fin["prio"] = 0
    cols = ["id", "username", "branch", "type", "net", "tc", "threads", "elo0", "elo1",
            "outcome", "games", "elo", "started", "finished", "tp", "prio"]
    df = pd.concat([fin[cols], act[cols]], ignore_index=True)
    df["started"] = pd.to_datetime(df.started, utc=True)
    df["finished"] = pd.to_datetime(df.finished, utc=True)
    df["gd"] = df.tc.map(game_duration)
    df["tcr"] = df.threads * df.gd / STC_GD
    df["sprt"] = df.elo0.notna()
    df["kind"] = np.where(~df.sprt, "", np.where(df.elo0 + df.elo1 < 0, "simp", "gain"))
    df["sub"] = df.apply(subtype, axis=1)
    return df


def find_parents(df: pd.DataFrame) -> pd.Series:
    """For each run, the cheaper passed run on the same user+branch that it followed
    (finished shortly before this one was submitted), or -1."""
    parent = pd.Series(-1, index=df.index)
    passed = df[df.outcome == "passed"]
    by_key = {k: g for k, g in passed.groupby(["username", "branch"])}
    for i, row in df.iterrows():
        g = by_key.get((row.username, row.branch))
        if g is None:
            continue
        c = g[(g.tcr < row.tcr) & (g.finished <= row.started + pd.Timedelta(minutes=30))
              & (g.finished >= row.started - pd.Timedelta(days=7))]
        if len(c):
            parent[i] = c.finished.idxmax()
    return parent


def gate_class(df: pd.DataFrame) -> pd.Series:
    """'confirmed' if a cheaper test of the same branch passed first, 'speculative'
    otherwise; '' for runs at STC cost or below."""
    out = pd.Series("", index=df.index)
    stc_like = {}
    for k, g in df.groupby(["username", "branch"]):
        stc_like[k] = g
    for i, row in df.iterrows():
        if row.tcr <= 1.01:
            continue
        g = stc_like[(row.username, row.branch)]
        c = g[(g.tcr < row.tcr) & (g.outcome == "passed")
              & (g.finished <= row.started + pd.Timedelta(minutes=30))]
        out[i] = "confirmed" if len(c) else "speculative"
    return out


def build_trace(df: pd.DataFrame, sim_start: str, now: str, max_threads: int) -> Trace:
    t0 = pd.Timestamp(sim_start)
    now_ts = pd.Timestamp(now)
    d = df[(df.finished.isna() | (df.finished > t0)) & (df.started < now_ts)].copy()
    d = d[d.threads <= max_threads]
    d["parent_global"] = find_parents(df).reindex(d.index)
    d["gate"] = gate_class(df).reindex(d.index)
    d = d.reset_index().rename(columns={"index": "gidx"})
    d["submit"] = (d.started - t0).dt.total_seconds()
    d["orig_finish"] = (d.finished - t0).dt.total_seconds()
    d["orig_dur_h"] = (d.finished - d.started).dt.total_seconds() / HOUR
    # Runs already in progress at t0: keep only the part of their work after t0.
    d["n0"] = 0.0
    d["need"] = d.games.astype(float)
    early = d.submit < 0
    frac = (d.orig_finish / (d.orig_finish - d.submit)).where(early, 1.0).clip(0, 1)
    d.loc[early, "n0"] = d.games[early] * (1 - frac[early])
    d.loc[early, "need"] = d.games[early] * frac[early]
    d.loc[early, "submit"] = 0.0
    # Runs still active at the end of the data keep running (their real end is unknown);
    # they hold load at its present level until every measured run is done.
    d["open"] = d.outcome == "active"
    d.loc[d.open, "need"] = np.inf
    d["fixed"] = ~d.sprt  # fixed-games runs (SPSA, progtests, ...) stop at their game count
    # Map parent links to local indices.
    g2l = {g: i for i, g in enumerate(d.gidx)}
    d["parent"] = d.parent_global.map(lambda g: g2l.get(g, -1) if g >= 0 else -1).astype(int)
    d["gap"] = 0.0
    has = d.parent >= 0
    d.loc[has, "gap"] = (d.submit[has].values - d.orig_finish[d.parent[has]].values).clip(0)
    # A dependent run only waits on its parent when the parent itself is simulated.
    d.loc[has & (d.parent.map(lambda p: d.submit[p] if p >= 0 else 0) <= 0), "parent"] = -1
    children = {}
    for i, p in enumerate(d.parent):
        if p >= 0:
            children.setdefault(p, []).append(i)
    d["user"] = d.username.astype("category").cat.codes
    return Trace(d, t0, (now_ts - t0).total_seconds(), children)


def capacity_profile(df: pd.DataFrame, t0: pd.Timestamp, now: pd.Timestamp, smooth_days=3):
    """Effective cores per day, rebuilt by spreading every run's estimated core-hours
    evenly over its lifetime (the trace's own work, so the replay is self-consistent)."""
    idx = pd.date_range(t0 - pd.Timedelta(days=10), now, freq="1h")
    load = np.zeros(len(idx))
    start = idx[0]
    fin = df.finished.fillna(now)
    ch = df.gd * df.games * df.threads / HOUR
    for s, f, c in zip(df.started, fin, ch):
        if f <= start or c <= 0:
            continue
        dur = (f - s).total_seconds() / HOUR
        if dur <= 0.05:
            continue
        i0 = max(0, int((s - start).total_seconds() // HOUR))
        i1 = min(len(idx), int((f - start).total_seconds() // HOUR) + 1)
        load[i0:i1] += c / dur
    s = pd.Series(load, index=idx).resample("1D").mean()
    s = s.rolling(smooth_days, center=True, min_periods=1).mean()
    days = ((s.index - t0).total_seconds() / DAY).astype(int)
    return {int(k): float(v) for k, v in zip(days, s.values) if k >= 0}


# --------------------------------------------------------------------------- workers

def load_workers(data_dir: str) -> pd.DataFrame:
    w = pd.read_csv(f"{data_dir}/workers.csv")
    w = w[w.cores > 0].copy()
    per_core = (w.mnps / w.cores).fillna((w.mnps / w.cores).median())
    rel = per_core / (per_core * w.cores).sum() * w.cores.sum()
    w["speed"] = rel.clip(0.3, 3.0)
    return w.reset_index(drop=True)


# --------------------------------------------------------------------------- engine

class State:
    """What a policy sees when computing itp (all arrays indexed by active run)."""

    def __init__(self, sim, act):
        r = sim.runs
        self.t = sim.t
        self.act = act
        self.tp = r["tp"][act]
        self.tcr = r["tcr"][act]
        self.games = r["n0"][act] + sim.prog[act]
        self.total = r["n0"][act] + np.where(np.isinf(r["need"][act]), np.nan, r["need"][act])
        self.passed = r["passed"][act]
        self.rejected = r["rejected"][act]
        self.sprt = r["sprt"][act]
        self.elo0 = r["elo0"][act]
        self.user = r["user"][act]
        self.workers = sim.workers_on[act]
        self.cores = sim.cores_on[act]
        self.age = sim.t - sim.arrived[act]
        self.cost = r["cost"][act]
        self.remaining = np.where(np.isinf(r["need"][act]), 2e5, r["need"][act] - sim.prog[act])
        self.heavy = r["tcr"][act] > 6.5
        self.spec = r["spec"][act]
        self.submit_rank = r["submit"][act]

    def llr_hi(self):
        frac = np.where(np.isnan(self.total), 0.0, self.games / np.maximum(self.total, 1))
        return self.passed & (frac >= 1 - LLR_HI_FRACTION)

    def user_sum(self, x):
        out = np.zeros(self.user.max() + 1 if len(self.user) else 1)
        np.add.at(out, self.user, x)
        return out[self.user]


class Sim:
    def __init__(self, trace: Trace, workers: pd.DataFrame, policy, capacity: dict,
                 horizon_days: float = 400, seed: int = 0, start_delay_h: float = 0.0,
                 kappa: float = 1.0):
        d = trace.df
        self.trace = trace
        self.policy = policy
        self.N = N = len(d)
        self.runs = {
            "tp": d.tp.values.astype(float),
            "prio": d.prio.values.astype(float),
            "tcr": d.tcr.values.astype(float),
            "gd": d.gd.values.astype(float),
            "threads": d.threads.values.astype(int),
            "cost": (d.gd * d.threads).values.astype(float),
            "need": d.need.values.astype(float),
            "n0": d.n0.values.astype(float),
            "fixed": d.fixed.values.astype(bool),
            "sprt": d.sprt.values.astype(bool),
            "passed": (d.outcome == "passed").values,
            "rejected": (d.outcome == "rejected").values,
            "elo0": d.elo0.fillna(0).values.astype(float),
            "user": d.user.values.astype(int),
            "submit": d.submit.values.astype(float),
            "spec": (d.gate == "speculative").values,
            "batch": np.where(d.sprt.values, 16, 2),
        }
        self.children = trace.children
        self.parent = d.parent.values
        self.gap = d.gap.values
        # workers
        self.W = len(workers)
        self.wcores = workers.cores.values.astype(int)
        self.wspeed = workers.speed.values.astype(float)
        self.cap_nominal = float((self.wcores * self.wspeed).sum())
        capacity = {k: v * kappa for k, v in capacity.items()}
        self.capacity = capacity
        last_day = max(capacity)
        arr, cur = [], capacity[min(capacity)]
        for k in range(last_day + 1):
            cur = capacity.get(k, cur)
            arr.append(cur)
        self.cap_by_day = np.array(arr)
        # dynamic state
        self.t = 0.0
        self.prog = np.zeros(N)
        self.rate = np.zeros(N)
        self.last = np.zeros(N)
        self.cores_on = np.zeros(N)
        self.workers_on = np.zeros(N, dtype=int)
        self.committed = np.zeros(N)
        self.itp = np.ones(N)
        self.arrived = np.full(N, np.nan)
        self.first_task = np.full(N, np.nan)
        self.finished = np.full(N, np.nan)
        self.core_secs = np.zeros(N)  # integral of cores while running
        self.fin_ver = np.zeros(N, dtype=int)
        self.active = []  # list of run indices
        self.act = np.zeros(0, dtype=int)
        self.itp_time = -1e9
        self.w_run = np.full(self.W, -1)
        self.w_rate = np.zeros(self.W)
        self.w_ver = np.zeros(self.W, dtype=int)
        self.w_last = np.full(self.W, -1)
        self.w_tstart = np.zeros(self.W)
        self.idle = set()
        self.heap = []
        self.seq = 0
        self.horizon = horizon_days * DAY
        self.samples = []
        self.rng = np.random.default_rng(seed)
        # approval + first build: same draw for every policy (common random numbers)
        self.delay = (np.random.default_rng(12345).exponential(start_delay_h * HOUR, N)
                      if start_delay_h > 0 else np.zeros(N))
        self.submitted = np.full(N, np.nan)

    # -- helpers
    def push(self, t, kind, a=0, b=0):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, a, b))

    def cap_mult(self, t):
        day = min(int(t // DAY), len(self.cap_by_day) - 1)
        return self.cap_by_day[max(day, 0)] / self.cap_nominal

    def sync(self, i):
        dt = self.t - self.last[i]
        if dt > 0:
            self.prog[i] += self.rate[i] * dt
            self.core_secs[i] += self.cores_on[i] * dt
        self.last[i] = self.t

    def schedule_finish(self, i):
        self.fin_ver[i] += 1
        if self.rate[i] > 1e-12 and not math.isinf(self.runs["need"][i]):
            left = max(0.0, self.runs["need"][i] - self.prog[i])
            self.push(self.t + left / self.rate[i], 2, i, self.fin_ver[i])

    def rebuild_active(self):
        self.act = np.array(self.active, dtype=int)
        self.itp_time = -1e9

    def refresh_itp(self):
        if self.t - self.itp_time < ITP_REFRESH or len(self.act) == 0:
            return
        for i in self.act:
            self.sync(i)
        st = State(self, self.act)
        self.itp[self.act] = np.maximum(self.policy.itp(st), 1e-9)
        self.itp_time = self.t
        self._state = st

    # -- worker request (port of sync_request_task's run selection)
    def request(self, w):
        a = self.act
        if len(a) == 0:
            self.idle.add(w)
            return
        self.refresh_itp()
        r = self.runs
        cw = self.wcores[w]
        elig = r["threads"][a] <= cw
        elig &= ~(r["fixed"][a] & (self.committed[a] >= r["need"][a]))
        allowed = self.policy.allowed(self, a, w) if hasattr(self.policy, "allowed") else None
        if allowed is not None:
            elig &= allowed
        if not elig.any():
            self.idle.add(w)
            self.push(self.t + 600, 4, w)
            return
        adj = self.cores_on[a].copy()
        adj[a == self.w_last[w]] += cw
        key = (-r["prio"][a]) * 1e12 + (adj > 0) * 1e9 + (adj + cw / 2) / self.itp[a]
        key = np.where(elig, key, np.inf)
        i = int(a[int(np.argmin(key))])
        self.assign(w, i)

    def assign(self, w, i):
        r = self.runs
        self.sync(i)
        q = self.wcores[w] // r["threads"][i]
        speed = self.wspeed[w] * self.cap_mult(self.t)
        gd = r["gd"][i]
        b = r["batch"][i]
        games = TASK_DURATION / gd * q
        games = max(b, b * int(games / b + 0.5))
        if r["fixed"][i]:
            games = min(games, r["need"][i] - self.committed[i])
        rate = q * speed / gd
        self.committed[i] += games
        self.rate[i] += rate
        self.cores_on[i] += self.wcores[w]
        self.workers_on[i] += 1
        if math.isnan(self.first_task[i]):
            self.first_task[i] = self.t
        self.w_run[w] = i
        self.w_rate[w] = rate
        self.w_ver[w] += 1
        self.w_last[w] = i
        self.idle.discard(w)
        self.push(self.t + games / rate, 1, w, self.w_ver[w])
        self.schedule_finish(i)

    def release(self, w):
        i = self.w_run[w]
        if i < 0:
            return
        self.sync(i)
        self.rate[i] = max(0.0, self.rate[i] - self.w_rate[w])
        self.cores_on[i] -= self.wcores[w]
        self.workers_on[i] -= 1
        if self.workers_on[i] == 0:
            self.rate[i] = 0.0
            self.cores_on[i] = 0
        self.w_run[w] = -1
        self.w_ver[w] += 1
        if not math.isnan(self.finished[i]):
            return
        self.schedule_finish(i)

    def finish(self, i):
        self.sync(i)
        self.finished[i] = self.t
        self.active.remove(i)
        self.rebuild_active()
        for w in np.nonzero(self.w_run == i)[0]:
            self.release(w)
        for w in np.nonzero(self.w_run == -1)[0]:
            if w not in self.idle:
                self.request(w)
        for j in self.children.get(i, []):
            self.push(self.t + self.gap[j], 5, j)

    def submit(self, i):
        self.submitted[i] = self.t
        self.push(self.t + self.delay[i], 0, i)

    def arrive(self, i):
        self.arrived[i] = self.t
        if self.runs["need"][i] <= 0:
            self.finished[i] = self.t
            for j in self.children.get(i, []):
                self.push(self.t + self.gap[j], 5, j)
            return
        self.last[i] = self.t
        self.active.append(i)
        self.rebuild_active()
        for w in list(self.idle):
            self.request(w)

    def sample(self):
        a = self.act
        rec = {"t": self.t, "active": len(a), "running": int((self.workers_on[a] > 0).sum()) if len(a) else 0}
        rec["cores"] = float(self.cores_on[a].sum()) if len(a) else 0.0
        rec["run_cores"] = {int(i): float(self.cores_on[i]) for i in a if self.cores_on[i] > 0}
        self.samples.append(rec)

    def run(self, until=None, measured=None):
        """Simulate until every measured run has finished (or the horizon)."""
        until = until or self.horizon
        for i in range(self.N):
            if self.parent[i] < 0:
                self.push(self.runs["submit"][i], 5, i)
        for w in range(self.W):
            self.push(0.0, 4, w)
        self.push(0.0, 3)
        measured = np.asarray(measured if measured is not None else [], dtype=int)
        while self.heap:
            t, _, kind, a, b = heapq.heappop(self.heap)
            if t > until:
                break
            self.t = t
            if kind == 0:
                self.arrive(a)
            elif kind == 1:  # task end
                if b != self.w_ver[a]:
                    continue
                i = self.w_run[a]
                self.release(a)
                if math.isnan(self.finished[i]) and self.prog[i] >= self.runs["need"][i] - 1e-6:
                    self.finish(i)  # its last games are in; also re-queues this worker
                else:
                    self.request(a)
            elif kind == 2:  # projected run finish
                if b != self.fin_ver[a] or not math.isnan(self.finished[a]):
                    continue
                self.sync(a)
                if self.prog[a] >= self.runs["need"][a] - 1e-6:
                    self.finish(a)
                else:
                    self.schedule_finish(a)
            elif kind == 3:  # hourly sample
                self.sample()
                if len(measured) and t > self.trace.t_end and not np.isnan(self.finished[measured]).any():
                    break
                self.push(t + HOUR, 3)
            elif kind == 5:
                self.submit(a)
            elif kind == 4:  # idle worker retry
                if self.w_run[a] < 0:
                    self.idle.discard(a)
                    self.request(a)
        for i in self.active:
            self.sync(i)
        return self


# --------------------------------------------------------------------------- results

def results_frame(sim: Sim) -> pd.DataFrame:
    d = sim.trace.df.copy()
    d["sim_submit"] = sim.submitted
    d["sim_arrive"] = sim.arrived
    d["sim_first"] = sim.first_task
    d["sim_finish"] = sim.finished
    d["sim_dur_h"] = (sim.finished - sim.submitted) / HOUR
    d["sim_core_h"] = sim.core_secs / HOUR
    return d


def closed_loop(trace: Trace, window_h: float = 3.0) -> Trace:
    """Users react to latency: a run submitted within `window_h` of one of the same
    user's runs finishing is treated as triggered by that finish (the user had a
    free slot), so it is submitted `gap` after the trigger finishes in the sim."""
    d = trace.df.copy()
    sub, fin = d.submit.values, d.orig_finish.values
    parent, gap = d.parent.values.copy(), d.gap.values.copy()
    for u, g in d.groupby("user"):
        idx = g.index.values
        for i in idx:
            if parent[i] >= 0 or sub[i] <= 0:
                continue
            f = fin[idx]
            ok = (f <= sub[i]) & (f >= sub[i] - window_h * HOUR) & (sub[idx] > 0) & (idx != i)
            if ok.any():
                j = idx[ok][np.argmax(f[ok])]
                parent[i], gap[i] = j, sub[i] - fin[j]
    d["parent"], d["gap"] = parent, gap
    children = {}
    for i, p in enumerate(parent):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    return Trace(d, trace.t0, trace.t_end, children)
