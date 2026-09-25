# Project status — 2026-09-08

Working notes for the PMPHET paper (`../../main.tex`). Detailed inventory; the
short version lives in session memory.

> `main.tex` at the **project root** is the live paper.
> `code/Ising/docs/paper/main.tex` is a stale duplicate ~200 lines behind — ignore or delete.

---

## Layout

```
code/Ising/
├── common/            shared by all experiments
│     quantum_portfolio_optimizer.py   QUBO core (build_qubo, qubo_to_ising,
│                                      solve_ising) + the 12-contract mock generator
│     benchmark.py                     ablation / solver / robustness routines
├── exp_12_mock/       synthetic 12 contracts (original results)
├── exp_12_real/       12 hand-picked Kalshi contracts (original results)
└── exp_200_real/      ACTIVE — 200-contract selection, empirical covariance
```

Scripts in the experiment folders carry a `sys.path` bootstrap so they can import
from `common/`. Data files sit beside the scripts that read them, so `cd` into a
folder before running.

---

## exp_200_real pipeline

Run in order. Each step caches, so later steps do not refetch.

| # | Command | Produces |
|---|---|---|
| 1 | `python fetch_universe.py` | `kalshi_universe.csv` — 62,945 open contracts, 7,896 events |
| 2 | `python select_contracts.py -n 200` | `contracts_200.csv` — 200 contracts, 33 events |
| 3 | `python fetch_history.py` | `price_history.csv` — 90 daily candles, 173 contracts |
| 4 | `python build_correlation.py` | `covariance_empirical.csv` (167×167), `event_correlation{,_shrunk}.csv`, `contracts_used.csv` |
| 5 | `python benchmark_200.py --K 3 --trials 20` | `ablation_200_K3.csv`, `benchmark_200_K3.csv`, `robustness_200_K3.csv`, `sensitivity_200_K3.csv` |

Figures / tables:
- `python plot_correlation.py --delta 0.771` → `fig_correlation.pdf` (two-panel, needs `figure*`)
- `python make_tables.py` → `tables_generated.tex` (replaces `tab:contracts`, adds `tab:categories`, replaces `tab:correlations`)
- `python scaling.py --ns 12,25,50,100,200,500,1000` → `scaling_results.csv`, then `plot_scaling.py` — **NOT YET COMPLETED**

### Selection rules (step 2)

Whole events only, never individual contracts — covariance structure lives at the
event level. Price in [0.02, 0.98], volume > 0, event size 2–15, and for
exchange-flagged mutually-exclusive events the YES prices must sum to within 10%
of 1 (a no-arbitrage check: ~40% of flagged events fail it badly, one summed to
3.94). Events ranked by traded volume so price history exists for step 4.

### Key discovery

Kalshi's `/events` endpoint exposes a **`mutually_exclusive` boolean** — 2,785
events carry it. Exclusivity is therefore exchange ground truth, replacing the
price-sum heuristic in `exp_12_real/kalshi_data.py` (which also mislabelled
`KXHIGHNY-T64`, a *less-than* market, as a nested threshold).

### Current numbers

- 167 contracts, 32 events, 496 exclusivity pairs
- structure: 137 disjoint (25 events) / 8 nested (1) / 22 unstructured (6)
- covariance is fully dense: 13,861 of 13,861 off-diagonals nonzero
- shrinkage δ = 0.771; raw sd 0.121 → shrunk 0.067
- edge = price × U(−0.10, 0.10), seed 42

---

## Benchmark suite (`benchmark_200.py`)

| Part | What | Paper table |
|---|---|---|
| A | exclusivity ablation, λ ∈ {0,1,2,5,10,20}, **best of 5 seeds** | `tab:ablation` |
| B | solver comparison: exact `C(n,K)` / SA / SciPy relax+round / CBC ×2 | `tab:benchmark` |
| C | SA robustness, 20 seeds | Results §C |
| D | β sweep, 0→2 in 6 steps, best of 5 seeds | `tab:sensitivity` |

Two guards were added after hangs:
- **No `2^n` brute force.** `benchmark.py`'s `run_solver_benchmark` calls it
  unconditionally; at n=167 that is 1.9e50 states. `run_solver_benchmark_200`
  uses `enum_cardinality` over `C(167,3)=762,355` instead (4 s).
- **CBC time limit.** `_cap_cbc_runtime(300)` monkeypatches `PULP_CBC_CMD`;
  `benchmark.py` sets no limit and the linearized QUBO has ~13,900 aux binaries.

Neither guard edits `benchmark.py`.

---

## K=3 results (run 2026-09-08, `bench200_K3.log`)

**Part A — ablation reproduced the false-hedge result on real data.**
At lambda=0 the optimiser buys `KXRATECUTCOUNT-26DEC31-T0` AND `-T1`
("Fed cuts 0 times" and "cuts 1 time"), exchange-flagged mutually exclusive.
Variance 0.0356 vs 0.2201 for the best valid portfolio — a 6x artificial
reduction — while taking HALF the edge (+0.059 vs +0.114). Unambiguously the
hedge being exploited. Stronger than the synthetic-data version in the paper.

**Part B — SA does NOT find the optimum at n=167.**

| Solver | Energy | Gap | Time |
|---|---|---|---|
| Exact C(167,3) | -18.006195 | — | 3.78 s |
| SA (Ising) | -17.998263 | 0.044% | 27.44 s |
| SciPy relax+round | 0.000000 | 100% (selects nothing) | 0.60 s |
| PuLP+CBC (QUBO) | -17.954681 | 0.286% | 300.90 s (TIMED OUT) |
| PuLP+CBC (native) | **-18.006195** | **0.000%** | 102.78 s |

CBC-native finds the exact optimum; SA misses and picks a different portfolio;
exact enumeration beats everything on time. **SA is not the hero at this scale** —
the defensible framing is that the QUBO/Ising *mapping* is the contribution and
SA is one solver for it. That makes the scaling sweep more important, since it is
where enumeration and CBC should break and SA should not.

**Part C — 19 unique portfolios across 20 seeds.** Energy mean -17.9997, sd
0.0028, best -18.0062 (equals exact, so SA reaches it roughly 1 seed in 20).
Budget 20/20, violations 0/20. The landscape is nearly flat near the optimum.

**Part D — beta sweep is clean.** Edge +0.138 -> +0.005, variance 0.600 -> 0.058,
monotone, stabilising by beta=1.2. Best-of-seeds removed the earlier noise.

### KNOWN BUG in the saved CSV

`benchmark_200_K3.csv` marks `PuLP+CBC (QUBO)` as `proved_optimal=True` at
energy -17.954681, which is worse than the exact optimum — impossible. It ran
300.90 s against a 300 s cap, i.e. it timed out; PuLP reports status `Optimal`
even when CBC stops early with a best-found solution, so the
`status == "Optimal"` test in `_pulp_solve` is wrong under a time limit. Energy
and gap are correct; only the flag is wrong. Fix before this row goes in a table.

---

## Open items

1. ~~Read the K=3 results.~~ DONE — see above. SA gap is 0.044%, not zero.
2. **Scaling sweep** — the advisor's central ask, still not run. Partial data
   before it was stopped: CBC-linearized 0.25 s → 2.55 s → 56.59 s at n=12/25/50,
   while CBC-native stayed at 0.26 s. Find where CBC-native actually breaks
   before claiming classical intractability.
3. **K=9 run?** Matches the intro's ρ=0.05 rule but has no exact ground truth.
4. **Paper rewrite** — Data/Model/Results against the new numbers, plus the
   claims that no longer hold (see session memory `paper-claims-to-revise`).
5. **QAOA on simulator** — `qiskit` 2.3.0 and `qiskit_aer` 0.17.2 are already
   installed, so this needs no funding and would justify "quantum" in the title.
