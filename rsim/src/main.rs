//! Event-driven model of fishtest's task scheduler (port of fishsim.py).
//!
//! Usage: fishsim-rs <jobs file> [threads]
//! Each line of the jobs file is: <trace file> <output file> <policy> <kappa>
//! The trace file is written by rsim.py; the output is read back by rsim.py.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

const TASK_DURATION: f64 = 1800.0;
const ITP_REFRESH: f64 = 60.0;
const HOUR: f64 = 3600.0;
const DAY: f64 = 86400.0;
const LLR_HI_FRACTION: f64 = 0.30;
const HORIZON_DAYS: f64 = 400.0;

// ------------------------------------------------------------------ trace

struct Run {
    tp: f64,
    prio: f64,
    tcr: f64,
    gd: f64,
    threads: i64,
    need: f64,
    n0: f64,
    fixed: bool,
    sprt: bool,
    passed: bool,
    elo0: f64,
    user: usize,
    submit: f64,
    spec: bool,
    batch: f64,
    parent: i64,
    gap: f64,
    delay: f64,
}

struct Trace {
    runs: Vec<Run>,
    wcores: Vec<i64>,
    wspeed: Vec<f64>,
    cap_by_day: Vec<f64>,
    measured: Vec<usize>,
    t_end: f64,
    n_users: usize,
    children: Vec<Vec<usize>>,
}

fn parse_trace(path: &str) -> Trace {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut it = text.split_ascii_whitespace();
    let mut next = || it.next().expect("truncated trace file");
    let n: usize = next().parse().unwrap();
    let w: usize = next().parse().unwrap();
    let d: usize = next().parse().unwrap();
    let m: usize = next().parse().unwrap();
    let t_end: f64 = next().parse().unwrap();
    let f = |s: &str| -> f64 { s.parse().unwrap() };
    let b = |s: &str| -> bool { s != "0" };
    let mut runs = Vec::with_capacity(n);
    for _ in 0..n {
        let tp = f(next());
        let prio = f(next());
        let tcr = f(next());
        let gd = f(next());
        let threads: i64 = next().parse().unwrap();
        let need = f(next());
        let n0 = f(next());
        let fixed = b(next());
        let sprt = b(next());
        let passed = b(next());
        let _rejected = b(next());
        let elo0 = f(next());
        let user: usize = next().parse().unwrap();
        let submit = f(next());
        let spec = b(next());
        let batch = f(next());
        let parent: i64 = next().parse().unwrap();
        let gap = f(next());
        let delay = f(next());
        runs.push(Run {
            tp, prio, tcr, gd, threads, need, n0, fixed, sprt, passed, elo0, user, submit,
            spec, batch, parent, gap, delay,
        });
    }
    let mut wcores = Vec::with_capacity(w);
    let mut wspeed = Vec::with_capacity(w);
    for _ in 0..w {
        wcores.push(next().parse().unwrap());
        wspeed.push(f(next()));
    }
    let cap_by_day: Vec<f64> = (0..d).map(|_| f(next())).collect();
    let measured: Vec<usize> = (0..m).map(|_| next().parse().unwrap()).collect();
    let n_users = runs.iter().map(|r| r.user).max().unwrap_or(0) + 1;
    let mut children = vec![Vec::new(); n];
    for (i, r) in runs.iter().enumerate() {
        if r.parent >= 0 {
            children[r.parent as usize].push(i);
        }
    }
    Trace { runs, wcores, wspeed, cap_by_day, measured, t_end, n_users, children }
}

// ------------------------------------------------------------------ policies

#[derive(Clone, Copy, Debug)]
enum Policy {
    Current,
    PreJune,
    CostCount(f64),
    CostCountAdd(f64),
    UserCap(f64, bool),
    TcCap(f64),
    AdditiveBonus,
    SpecMalus(f64),
    HeavyCap(f64),
    TestCap(f64),
}

fn parse_policy(name: &str) -> Policy {
    let num = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| panic!("bad policy {name}")) };
    if name == "current" {
        return Policy::Current;
    }
    if name == "pre_june" {
        return Policy::PreJune;
    }
    if name == "additive_bonus" {
        return Policy::AdditiveBonus;
    }
    if name == "cost_count" {
        return Policy::CostCount(0.774);
    }
    if let Some(rest) = name.strip_prefix("cost_count_p") {
        return match rest.strip_suffix("+add") {
            Some(p) => Policy::CostCountAdd(num(p)),
            None => Policy::CostCount(num(rest)),
        };
    }
    if let Some(rest) = name.strip_prefix("user_cap") {
        return match rest.strip_suffix("+add") {
            Some(s) => Policy::UserCap(num(s), true),
            None => Policy::UserCap(num(rest), false),
        };
    }
    if let Some(rest) = name.strip_prefix("tc_cap") {
        return Policy::TcCap(num(rest));
    }
    if let Some(rest) = name.strip_prefix("spec_malus") {
        return Policy::SpecMalus(num(rest));
    }
    if let Some(rest) = name.strip_prefix("heavy_cap") {
        return Policy::HeavyCap(num(rest) / 100.0);
    }
    if let Some(rest) = name.strip_prefix("test_cap") {
        return Policy::TestCap(num(rest) / 100.0);
    }
    panic!("unknown policy {name}");
}

fn tc_weight(tcr: f64, power: f64, cap: f64) -> f64 {
    let x = tcr.min(cap);
    if x > 1.0 { x.powf(power) } else { 1.0 }
}

fn count_malus(c: f64) -> f64 {
    36.0 / (36.0 + c * c)
}

// ------------------------------------------------------------------ events

#[derive(Clone, Copy)]
struct Ev {
    t: f64,
    seq: u64,
    kind: u8,
    a: usize,
    b: i64,
}

impl PartialEq for Ev {
    fn eq(&self, o: &Self) -> bool {
        self.seq == o.seq
    }
}
impl Eq for Ev {}
impl PartialOrd for Ev {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ev {
    // BinaryHeap is a max-heap: reverse so the earliest (t, seq) pops first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.seq.cmp(&self.seq))
    }
}

const ARRIVE: u8 = 0;
const TASK_END: u8 = 1;
const RUN_FINISH: u8 = 2;
const SAMPLE: u8 = 3;
const RETRY: u8 = 4;
const SUBMIT: u8 = 5;

// ------------------------------------------------------------------ engine

struct Sample {
    t: f64,
    active: usize,
    running: usize,
    cores: f64,
    run_cores: Vec<(usize, f64)>,
}

struct Sim<'a> {
    tr: &'a Trace,
    policy: Policy,
    cap_nominal: f64,
    total_cores: f64,
    kappa: f64,
    t: f64,
    prog: Vec<f64>,
    rate: Vec<f64>,
    last: Vec<f64>,
    cores_on: Vec<f64>,
    workers_on: Vec<i64>,
    committed: Vec<f64>,
    itp: Vec<f64>,
    submitted: Vec<f64>,
    arrived: Vec<f64>,
    first_task: Vec<f64>,
    finished: Vec<f64>,
    core_secs: Vec<f64>,
    fin_ver: Vec<i64>,
    active: Vec<usize>,
    itp_time: f64,
    w_run: Vec<i64>,
    w_rate: Vec<f64>,
    w_ver: Vec<i64>,
    w_last: Vec<i64>,
    idle: Vec<bool>,
    heap: BinaryHeap<Ev>,
    seq: u64,
    samples: Vec<Sample>,
    ubuf: Vec<f64>,
}

impl<'a> Sim<'a> {
    fn new(tr: &'a Trace, policy: Policy, kappa: f64) -> Self {
        let n = tr.runs.len();
        let w = tr.wcores.len();
        let cap_nominal = tr.wcores.iter().zip(&tr.wspeed).map(|(&c, &s)| c as f64 * s).sum();
        let total_cores = tr.wcores.iter().sum::<i64>() as f64;
        Sim {
            tr,
            policy,
            cap_nominal,
            total_cores,
            kappa,
            t: 0.0,
            prog: vec![0.0; n],
            rate: vec![0.0; n],
            last: vec![0.0; n],
            cores_on: vec![0.0; n],
            workers_on: vec![0; n],
            committed: vec![0.0; n],
            itp: vec![1.0; n],
            submitted: vec![f64::NAN; n],
            arrived: vec![f64::NAN; n],
            first_task: vec![f64::NAN; n],
            finished: vec![f64::NAN; n],
            core_secs: vec![0.0; n],
            fin_ver: vec![0; n],
            active: Vec::new(),
            itp_time: -1e9,
            w_run: vec![-1; w],
            w_rate: vec![0.0; w],
            w_ver: vec![0; w],
            w_last: vec![-1; w],
            idle: vec![false; w],
            heap: BinaryHeap::new(),
            seq: 0,
            samples: Vec::new(),
            ubuf: vec![0.0; tr.n_users],
        }
    }

    fn push(&mut self, t: f64, kind: u8, a: usize, b: i64) {
        self.seq += 1;
        self.heap.push(Ev { t, seq: self.seq, kind, a, b });
    }

    fn cap_mult(&self, t: f64) -> f64 {
        let n = self.tr.cap_by_day.len() as i64;
        let day = ((t / DAY).floor() as i64).min(n - 1).max(0) as usize;
        self.tr.cap_by_day[day] * self.kappa / self.cap_nominal
    }

    fn sync(&mut self, i: usize) {
        let dt = self.t - self.last[i];
        if dt > 0.0 {
            self.prog[i] += self.rate[i] * dt;
            self.core_secs[i] += self.cores_on[i] * dt;
        }
        self.last[i] = self.t;
    }

    fn schedule_finish(&mut self, i: usize) {
        self.fin_ver[i] += 1;
        let need = self.tr.runs[i].need;
        if self.rate[i] > 1e-12 && need.is_finite() {
            let left = (need - self.prog[i]).max(0.0);
            let t = self.t + left / self.rate[i];
            let v = self.fin_ver[i];
            self.push(t, RUN_FINISH, i, v);
        }
    }

    fn games(&self, i: usize) -> f64 {
        self.tr.runs[i].n0 + self.prog[i]
    }

    fn done_frac(&self, i: usize) -> f64 {
        let r = &self.tr.runs[i];
        if r.need.is_finite() {
            self.games(i) / (r.n0 + r.need).max(1.0)
        } else {
            0.0
        }
    }

    fn latency_bonus(&self, i: usize) -> f64 {
        let r = &self.tr.runs[i];
        let mut b = (self.games(i) / 200_000.0).min(1.0);
        let llr_hi = r.passed && self.done_frac(i) >= 1.0 - LLR_HI_FRACTION;
        if r.sprt && llr_hi {
            b += 0.1 + 0.1 * ((r.elo0 > 0.0) as i64 as f64);
        }
        b
    }

    /// Sum of `x(i)` over the user's active runs, for every active run's user.
    fn user_sums(&mut self, x: impl Fn(&Self, usize) -> f64) -> Vec<f64> {
        let tr = self.tr;
        for &i in &self.active {
            self.ubuf[tr.runs[i].user] = 0.0;
        }
        for k in 0..self.active.len() {
            let i = self.active[k];
            let v = x(self, i);
            self.ubuf[tr.runs[i].user] += v;
        }
        self.active.iter().map(|&i| self.ubuf[tr.runs[i].user]).collect()
    }

    fn current_itp(&mut self) -> Vec<f64> {
        let tr = self.tr;
        let c = self.user_sums(|s, i| (s.workers_on[i] > 0) as i64 as f64);
        self.active
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let r = &tr.runs[i];
                r.tp * tc_weight(r.tcr, 0.774, f64::INFINITY) * count_malus(c[k]) * (1.0 + self.latency_bonus(i))
            })
            .collect()
    }

    fn refresh_itp(&mut self) {
        if self.t - self.itp_time < ITP_REFRESH || self.active.is_empty() {
            return;
        }
        for k in 0..self.active.len() {
            let i = self.active[k];
            self.sync(i);
        }
        let tr = self.tr;
        let inf = f64::INFINITY;
        let vals: Vec<f64> = match self.policy {
            Policy::Current | Policy::HeavyCap(_) | Policy::TestCap(_) => self.current_itp(),
            Policy::SpecMalus(f) => {
                let v = self.current_itp();
                self.active.iter().zip(v).map(|(&i, x)| x * if tr.runs[i].spec { f } else { 1.0 }).collect()
            }
            Policy::PreJune => {
                let c = self.user_sums(|s, i| (s.workers_on[i] > 0) as i64 as f64);
                self.active
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let r = &tr.runs[i];
                        let g = self.games(i);
                        let llr = if r.passed { 2.94 * self.done_frac(i) } else { 0.0 };
                        let mut b = if r.sprt { (llr.clamp(0.0, 2.0) + 3.0) / 3.0 } else { 1.0 };
                        if r.sprt && g > 200_000.0 {
                            b *= (g / 200_000.0).min(2.0);
                        }
                        r.tp * tc_weight(r.tcr, 0.774, inf) * count_malus(c[k]) * b
                    })
                    .collect()
            }
            Policy::CostCount(p) | Policy::CostCountAdd(p) => {
                let add = matches!(self.policy, Policy::CostCountAdd(_));
                let c = self.user_sums(|s, i| {
                    (s.workers_on[i] > 0) as i64 as f64 * tc_weight(s.tr.runs[i].tcr, p, inf)
                });
                self.active
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let r = &tr.runs[i];
                        let w = tc_weight(r.tcr, 0.774, inf);
                        let b = self.latency_bonus(i);
                        if add {
                            count_malus(c[k]) * (r.tp * w + 100.0 * b)
                        } else {
                            r.tp * w * count_malus(c[k]) * (1.0 + b)
                        }
                    })
                    .collect()
            }
            Policy::UserCap(cap, add) => {
                let l = self.user_sums(|s, i| {
                    let r = &s.tr.runs[i];
                    (s.workers_on[i] > 0) as i64 as f64 * (r.tp / 100.0 * tc_weight(r.tcr, 0.774, inf))
                });
                self.active
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let r = &tr.runs[i];
                        let base = r.tp / 100.0 * tc_weight(r.tcr, 0.774, inf);
                        let scale = (cap / l[k].max(1e-9)).min(1.0);
                        let b = self.latency_bonus(i);
                        if add { 100.0 * scale * (base + b) } else { 100.0 * scale * base * (1.0 + b) }
                    })
                    .collect()
            }
            Policy::TcCap(cap) => {
                let c = self.user_sums(|s, i| (s.workers_on[i] > 0) as i64 as f64);
                self.active
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let r = &tr.runs[i];
                        r.tp * tc_weight(r.tcr, 0.774, cap) * count_malus(c[k]) * (1.0 + self.latency_bonus(i))
                    })
                    .collect()
            }
            Policy::AdditiveBonus => {
                let c = self.user_sums(|s, i| (s.workers_on[i] > 0) as i64 as f64);
                self.active
                    .iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let r = &tr.runs[i];
                        count_malus(c[k]) * (r.tp * tc_weight(r.tcr, 0.774, inf) + 100.0 * self.latency_bonus(i))
                    })
                    .collect()
            }
        };
        for (k, &i) in self.active.iter().enumerate() {
            self.itp[i] = vals[k].max(1e-9);
        }
        self.itp_time = self.t;
    }

    // Port of sync_request_task's run selection.
    fn request(&mut self, w: usize) {
        if self.active.is_empty() {
            self.idle[w] = true;
            return;
        }
        self.refresh_itp();
        let tr = self.tr;
        let cw = tr.wcores[w];
        let cwf = cw as f64;
        let mut block_heavy = false;
        let mut test_limit = f64::INFINITY;
        match self.policy {
            Policy::HeavyCap(frac) => {
                let used: f64 = self.active.iter().filter(|&&i| tr.runs[i].tcr > 6.5).map(|&i| self.cores_on[i]).sum();
                block_heavy = used + cwf > frac * self.total_cores;
            }
            Policy::TestCap(frac) => test_limit = (frac * self.total_cores).max(cwf),
            _ => {}
        }
        let last = self.w_last[w];
        let mut best: i64 = -1;
        let mut best_key = f64::INFINITY;
        for &i in &self.active {
            let r = &tr.runs[i];
            if r.threads > cw {
                continue;
            }
            if r.fixed && self.committed[i] >= r.need {
                continue;
            }
            if block_heavy && r.tcr > 6.5 {
                continue;
            }
            if self.cores_on[i] + cwf > test_limit {
                continue;
            }
            let adj = self.cores_on[i] + if i as i64 == last { cwf } else { 0.0 };
            let key = (-r.prio) * 1e12 + ((adj > 0.0) as i64 as f64) * 1e9 + (adj + cwf / 2.0) / self.itp[i];
            if key < best_key {
                best_key = key;
                best = i as i64;
            }
        }
        if best < 0 {
            self.idle[w] = true;
            let t = self.t + 600.0;
            self.push(t, RETRY, w, 0);
            return;
        }
        self.assign(w, best as usize);
    }

    fn assign(&mut self, w: usize, i: usize) {
        let tr = self.tr;
        self.sync(i);
        let r = &tr.runs[i];
        let q = (tr.wcores[w] / r.threads) as f64;
        let speed = tr.wspeed[w] * self.cap_mult(self.t);
        let b = r.batch;
        let mut games = TASK_DURATION / r.gd * q;
        games = b.max(b * (games / b + 0.5).trunc());
        if r.fixed {
            games = games.min(r.need - self.committed[i]);
        }
        let rate = q * speed / r.gd;
        self.committed[i] += games;
        self.rate[i] += rate;
        self.cores_on[i] += tr.wcores[w] as f64;
        self.workers_on[i] += 1;
        if self.first_task[i].is_nan() {
            self.first_task[i] = self.t;
        }
        self.w_run[w] = i as i64;
        self.w_rate[w] = rate;
        self.w_ver[w] += 1;
        self.w_last[w] = i as i64;
        self.idle[w] = false;
        let t = self.t + games / rate;
        let v = self.w_ver[w];
        self.push(t, TASK_END, w, v);
        self.schedule_finish(i);
    }

    fn release(&mut self, w: usize) {
        let i = self.w_run[w];
        if i < 0 {
            return;
        }
        let i = i as usize;
        self.sync(i);
        self.rate[i] = (self.rate[i] - self.w_rate[w]).max(0.0);
        self.cores_on[i] -= self.tr.wcores[w] as f64;
        self.workers_on[i] -= 1;
        if self.workers_on[i] == 0 {
            self.rate[i] = 0.0;
            self.cores_on[i] = 0.0;
        }
        self.w_run[w] = -1;
        self.w_ver[w] += 1;
        if !self.finished[i].is_nan() {
            return;
        }
        self.schedule_finish(i);
    }

    fn finish(&mut self, i: usize) {
        self.sync(i);
        self.finished[i] = self.t;
        if let Some(pos) = self.active.iter().position(|&x| x == i) {
            self.active.remove(pos);
        }
        self.itp_time = -1e9;
        for w in 0..self.w_run.len() {
            if self.w_run[w] == i as i64 {
                self.release(w);
            }
        }
        let free: Vec<usize> = (0..self.w_run.len()).filter(|&w| self.w_run[w] == -1).collect();
        for w in free {
            if !self.idle[w] {
                self.request(w);
            }
        }
        self.wake_children(i);
    }

    fn wake_children(&mut self, i: usize) {
        let tr = self.tr;
        for &j in &tr.children[i] {
            let t = self.t + tr.runs[j].gap;
            self.push(t, SUBMIT, j, 0);
        }
    }

    fn submit(&mut self, i: usize) {
        self.submitted[i] = self.t;
        let t = self.t + self.tr.runs[i].delay;
        self.push(t, ARRIVE, i, 0);
    }

    fn arrive(&mut self, i: usize) {
        self.arrived[i] = self.t;
        if self.tr.runs[i].need <= 0.0 {
            self.finished[i] = self.t;
            self.wake_children(i);
            return;
        }
        self.last[i] = self.t;
        self.active.push(i);
        self.itp_time = -1e9;
        let idle: Vec<usize> = (0..self.idle.len()).filter(|&w| self.idle[w]).collect();
        for w in idle {
            self.request(w);
        }
    }

    fn sample(&mut self) {
        let mut s = Sample { t: self.t, active: self.active.len(), running: 0, cores: 0.0, run_cores: Vec::new() };
        for &i in &self.active {
            if self.workers_on[i] > 0 {
                s.running += 1;
            }
            s.cores += self.cores_on[i];
            if self.cores_on[i] > 0.0 {
                s.run_cores.push((i, self.cores_on[i]));
            }
        }
        self.samples.push(s);
    }

    fn run(&mut self) {
        let tr = self.tr;
        for (i, r) in tr.runs.iter().enumerate() {
            if r.parent < 0 {
                self.push(r.submit, SUBMIT, i, 0);
            }
        }
        for w in 0..tr.wcores.len() {
            self.push(0.0, RETRY, w, 0);
        }
        self.push(0.0, SAMPLE, 0, 0);
        let until = HORIZON_DAYS * DAY;
        while let Some(ev) = self.heap.pop() {
            if ev.t > until {
                break;
            }
            self.t = ev.t;
            let a = ev.a;
            match ev.kind {
                ARRIVE => self.arrive(a),
                TASK_END => {
                    if ev.b != self.w_ver[a] {
                        continue;
                    }
                    let i = self.w_run[a] as usize;
                    self.release(a);
                    if self.finished[i].is_nan() && self.prog[i] >= tr.runs[i].need - 1e-6 {
                        self.finish(i);
                    } else {
                        self.request(a);
                    }
                }
                RUN_FINISH => {
                    if ev.b != self.fin_ver[a] || !self.finished[a].is_nan() {
                        continue;
                    }
                    self.sync(a);
                    if self.prog[a] >= tr.runs[a].need - 1e-6 {
                        self.finish(a);
                    } else {
                        self.schedule_finish(a);
                    }
                }
                SAMPLE => {
                    self.sample();
                    if !tr.measured.is_empty()
                        && ev.t > tr.t_end
                        && tr.measured.iter().all(|&i| !self.finished[i].is_nan())
                    {
                        break;
                    }
                    self.push(ev.t + HOUR, SAMPLE, 0, 0);
                }
                SUBMIT => self.submit(a),
                RETRY => {
                    if self.w_run[a] < 0 {
                        self.idle[a] = false;
                        self.request(a);
                    }
                }
                _ => unreachable!(),
            }
        }
        for k in 0..self.active.len() {
            let i = self.active[k];
            self.sync(i);
        }
    }

    fn write(&self, path: &str) {
        let f = fs::File::create(path).unwrap_or_else(|e| panic!("writing {path}: {e}"));
        let mut out = BufWriter::new(f);
        writeln!(out, "{} {}", self.prog.len(), self.samples.len()).unwrap();
        for i in 0..self.prog.len() {
            writeln!(
                out,
                "{} {} {} {} {} {}",
                self.submitted[i], self.arrived[i], self.first_task[i], self.finished[i], self.core_secs[i], self.prog[i]
            )
            .unwrap();
        }
        for s in &self.samples {
            write!(out, "{} {} {} {} {}", s.t, s.active, s.running, s.cores, s.run_cores.len()).unwrap();
            for (i, c) in &s.run_cores {
                write!(out, " {} {}", i, c).unwrap();
            }
            writeln!(out).unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: fishsim-rs <jobs file> [threads]");
        std::process::exit(2);
    }
    let jobs: Vec<Vec<String>> = fs::read_to_string(&args[1])
        .expect("reading jobs file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_ascii_whitespace().map(String::from).collect())
        .collect();
    let threads = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    // Parse each distinct trace once and share it across jobs.
    let mut traces: HashMap<String, Arc<Trace>> = HashMap::new();
    for j in &jobs {
        traces.entry(j[0].clone()).or_insert_with(|| Arc::new(parse_trace(&j[0])));
    }
    let traces = Arc::new(traces);
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..threads.min(jobs.len()) {
            let (jobs, traces, next) = (jobs.clone(), traces.clone(), next.clone());
            scope.spawn(move || loop {
                let k = next.fetch_add(1, AtomicOrdering::Relaxed);
                if k >= jobs.len() {
                    break;
                }
                let j = &jobs[k];
                let tr = &traces[&j[0]];
                let policy = parse_policy(&j[2]);
                let kappa: f64 = j[3].parse().expect("kappa");
                let start = std::time::Instant::now();
                let mut sim = Sim::new(tr, policy, kappa);
                sim.run();
                sim.write(&j[1]);
                println!("done {} {} k={} in {:.2}s", j[0], j[2], kappa, start.elapsed().as_secs_f64());
            });
        }
    });
}
