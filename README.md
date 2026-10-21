# fishtest scheduling model

Replays real fishtest workloads (from xu-shawn/fishtest_track) through a model of
`rundb.sync_request_task` and compares itp policies. A scheduling policy cannot change
how many games an SPRT needs, so every test keeps its real game count; only *when* it
gets its games changes.

## Data

`data/` is a snapshot taken 20 Oct 2026 22:45 UTC, which every result here is calibrated on:

* `data/fishtest_track/`: the `data/` folder of https://github.com/xu-shawn/fishtest_track
  (a year of finished runs, active runs, workers).
* `data/active_runs.json`: one read of fishtest's `/api/active_runs`, for the live runs'
  throughput and priority settings.

To model a newer period, replace these files and rerun `calibrate.py` (stages `delay`, `tp`,
`kappa`) after moving the dates in `common.py`; `tp_calibrated.npy` holds the fitted
throughput habits for the current snapshot.

## Files

| File | What it is |
|:--|:--|
| `fishsim.py` | Trace building (runs, STC->LTC parent links, capacity profile, workers) and the reference Python engine |
| `rsim/` | Rust port of the engine (~80x faster, ~2 s per 142-day replay). `cargo build --release` |
| `rsim.py` | Python <-> Rust bridge: export a trace, run job batches, load results |
| `policies.py` | Python versions of every policy (the Rust engine has the same set, parsed by name) |
| `closed.py`, `analyze.py`, `per_user.py` | Python-engine runs and tables (closed loop, scoring, per-user impact) |
| `explore.py`, `explore2.py` | First look at the data: core share by type, speculative tests |
| `common.py` | Main window setup: sim from 1 Jun 2026, measure runs submitted 11 Jul - 6 Oct, load to 20 Oct |
| `calibrate.py` | Startup delay, per-(user, class) throughput fit (capped at 100), capacity factor scan |
| `experiment.py` | Metrics (latency by class, STC->LTC pipeline, cores per STC, heavy share, ...) |
| `rexp.py` | Main study: 27 policies x 10 seeds x {open, crowded, closed-loop}, paired vs current |
| `summarize.py` | Tables from `results_rust/replicates.csv` (`python3 summarize.py open|crowded|closed`) |
| `windows.py` | Same comparison on three earlier periods (Dec-Jan, Mar-Apr, May-Jul), recalibrated |

## Calibration (main window)

* 30 min mean startup delay (approval + first build): fixes short runs.
* Effective throughput per (user, test class) for non-STC runs, fitted from real vs
  simulated durations, only ever lowered (25-100).
* Capacity = core-hours actually done (kappa = 1). The sim is then ~20% less crowded
  than reality; kappa = 0.985 ("crowded") is ~30% more crowded. The system is right at
  saturation, so a 1-2% capacity change moves latency a lot.
* Validation: current formula reproduces the live core split (r = 0.993) and per-test
  durations (log correlation 0.90).

## Closed loop

`fishsim.closed_loop` ties any test submitted within 3 h of one of the same user's tests
finishing to that finish, so faster policies also bring submissions forward.

## Run

Needs Python 3 with numpy and pandas, and a Rust toolchain.

```sh
cd rsim && cargo build --release && cd ..
python3 rexp.py 10            # main study, ~3.5 min on 12 cores -> results_rust/replicates.csv
python3 summarize.py closed   # or open / crowded
python3 windows.py 5          # earlier periods, ~2 min -> results_rust/windows.csv
python3 validate.py           # Python engine vs reality for the current formula (~1 min)
```

The first run builds `setup.pkl` (trace cache, ~30 s). Delete it after changing the data
or `fishsim.build_trace`.

## Headline result (from `results_rust/`)

Counting each running test as `tc_ratio^0.5` in the c1f29c malus (instead of 1) and making
the latency bonus additive (`policy cost_count_p0.5+add`), across 4 periods of the past year
and open / crowded / closed-loop scenarios, vs the current formula: STC median latency -56%
to -81%, passed STC gainers -58% to -82%, STC->LTC pass -31% to -63%, LTC -35% to -72%,
heavy tests (VVLTC/VLTC/SMP-LTC) -32% to +70%.
