# main_200.tex revision notes

Working notes for revising `main_200.tex`, the 167-contract version of the
paper. Started 2026-09-24 and updated 2026-09-25, so the work can resume if the
conversation context is lost.

- `main_200.tex` (project root) is the live paper. `main.tex` is the old
  12-contract draft and is superseded; `code/Ising/docs/paper/main.tex` is an
  even older duplicate.
- All experiment code is in `code/Ising/exp_200_real/`; shared solver code is
  in `code/Ising/common/`.
- Ground rule (Adrian): build and run are separate steps. Do not run
  experiments, reruns or sweeps without an explicit go-ahead.

---

## 1. Status at a glance

| Item | Status |
|---|---|
| Audit of main_200.tex against code, data and theory | DONE (§2) |
| Text fixes that needed no rerun | DONE |
| Code fixes (exact solver, gamma*K^2, proved flag, structure overrides, strike parser) | DONE (§5) |
| Edge freeze (`edges_frozen.csv`) | DONE |
| Rerun with fixes, incoherent forecast | DONE; results kept in `incoherent_K3/` |
| Coherent-forecast projection (`make_coherent`) | DONE |
| Many-forecast ablation (`ablation_many.py`), 100 draws x 3 modes x 9 betas | DONE (§3) |
| Main-table rerun with coherent forecast (`--coherence cap`) | DONE (§6) |
| Paper rewrite: new exclusivity story, combined table, coherent numbers | DONE 2026-09-24 (§4) |
| Compile main_200.tex and check layout | NOT DONE (no LaTeX on this machine) |
| `code/Ising/STATUS.md` update | NOT DONE (still has old numbers and the false "~1 in 20" claim) |
| Scaling sweep (`scaling.py`), the TA's central ask | NOT RUN; script needs fixes first (§7) |
| Optional: penalty-rescaling test for SA | NOT DONE |
| Git commit / push | PENDING decisions (§8) |

---

## 2. Audit findings (2026-09-24) and what was done about each

Verified correct as written: universe counts (62,945 contracts / 7,896 events /
19 categories; MX flag on 2,785 events / 18,544 contracts); about 40% of MX
events fail the coherence check (42.5% raw); 167 contracts / 32 events;
T = 89, 1/sqrt(T-3) = 0.108, raw dispersion 0.121, delta = 0.771; category
table; the Fed Sept / rate-cut-count pair is the top shrunk correlation; all
refs and citations resolve.

Found wrong, all now fixed in the paper and/or code:

| Finding | Resolution |
|---|---|
| Energies omitted the constant gamma*K^2 = 18, so "0.044% gap" was meaningless | Code adds it; paper reports E and gap E - E* |
| Ablation and sensitivity tables were SA best-of-5, not optimal | Solved exactly (`solve_exact_or_sa`) |
| "SA reaches the optimum ~1 in 20 seeds" | False (0/20); paper corrected |
| "Four independent solvers agree" | Only the two exact methods agree; reworded |
| Six "unstructured" events actually had structure (BTC, Greenland, TrumpOut, Aliens nested; Venezuela exclusive but unflagged) | Overrides + strike parser; verified against Kalshi `rules_primary` |
| "788 entries fixed by structure" | Now 560 (511 exclusive + 49 nested) |
| PSD projection distorts the "exact" within-event blocks (99/560 pairs >10%, 13 sign flips) | Disclosed in the paper |
| Covariance violates Fréchet bounds (228 of 13,861 pairs) | Disclosed in the paper (bounds stated, not enforced) |
| "Whole events", "sixty observations", "full set of open markets" | Reworded (partial events, history span, first 8,000 events) |
| Martingale section: [.,.] notation, pricing measure, stationarity and one-factor assumptions unstated, "correct quantity not a proxy" | Rewritten |
| Shrinkage called "selected by the data"; Ledoit-Wolf would give 0.52 | Described as a heuristic |
| "Most negative covariance in the matrix" | False (election D/R pairs -0.249); now "lower Fréchet bound at those prices" |
| "False hedge" / "logically impossible" / "variance reduction that cannot exist" | Wrong framing; replaced (§3) |
| Scope-of-limitations paragraph still said 12 contracts; Table I C(1000,50) | Fixed (9.5e84) |

---

## 3. The key finding: what the exclusivity term actually does

The old story was wrong. Buying YES on two mutually exclusive contracts is a
legitimate bet on their union, and its low variance is real (Fed "cuts 0" 0.912
+ "cuts 1" 0.090 pays $1 if the Fed cuts 0 or 1 times; the pair variance is about
0). What was phantom was the synthetic edge: independent noise made the
forecast believe P(0 or 1 cuts) = 1.061.

`ablation_many.py`: 100 forecast draws per mode, lambda = 0 vs lambda >= 1,
solved exactly (fast enumerator checked against the full QUBO at every beta).
Share of draws whose penalty-OFF optimum holds an exclusive combination:

| beta | none | cap (default) | match (strict) |
|---|---|---|---|
| 0 | 3% | 0% | 0% |
| 0.05 | 23% | 6% | 2% |
| 0.1 | 52% | 60% | 12% |
| 0.15 | 73% | 96% | 86% |
| 0.2 | 84% | 99% | 99% |
| 0.25 | 91% | 100% | 100% |
| 0.5 (paper) | 100% | 100% | 100% |
| 1 | 100% | 100% | 100% |
| 2 | 100% | 100% | 100% |

With the penalty ON, violations are 0% everywhere.

- Under `match`, every draw at beta = 0.5 buys both sides of a two-outcome
  election market (price sum 1.000, zero variance, zero edge): synthetic cash.
  Under `cap` the most common holding is two F1 drivers (KXF1-26 GR + KA).
- At beta = 2 the penalty-off optimum has negative edge in 85% of `cap` draws
  (100% of `match`); for the paper draw it holds all three Fed September
  outcomes (price sum 1.02), edge -0.020, variance 0.001.
- Penalty-off edge is lower than the valid optimum's in 65% of `cap` draws at
  beta = 0.5 and 98% at beta = 2 (hence "often" in the paper).

Claim now in the paper: holding several outcomes of one event is a bet on their
union, a near-riskless synthetic position when they cover the event. Once risk
enters the objective, an optimizer without the constraint fills its limited
positions with such combinations; the penalty removes them. Specific to event
contracts. The penalty form is standard (Lucas 2014); the constraint is new to
portfolio optimization (Adrian wants the novelty claim kept in that form).

---

## 4. Paper state (main_200.tex after the 2026-09-24 rewrite)

Backups: `paper_backups/main_200.before_coherent_rewrite.tex` and
`paper_backups/main_200.after_coherent_rewrite.tex`.

- Abstract, Intro "This work", Model note after Eq. disjoint_cov, Results and
  Conclusion carry the new exclusivity story (§3). No "false hedge",
  "impossible" or "contradictory" phrasing remains.
- Model / Problem setup has a paragraph on forecast coherence: the raw draw is
  incoherent (1.06 Fed example); the projection (cap; nested ladders made
  monotone by isotonic fit) and the stricter `match` variant; coherence cut
  the valid-optimum edge from 0.115 to 0.005.
- Data section: Venezuela added to the exclusive set (prices sum to 1.16;
  exclusivity is settlement logic, not quotes); five events reclassified as
  nested; 26 exclusive / 6 nested events, 511 pairs, 560 fixed entries, 13,301
  estimated; categories table caption dated 2026-09-08.
- Results:
  - `tab:ablation` is now ONE full-width `table*` replacing the old ablation and
    sensitivity tables: per beta in {0, .05, .1, .15, .2, .25, .5, 1, 2},
    penalty on edge/var | penalty off edge/var + exclusive holding | share of
    100 draws (cap); `match` and `none` shares are in the caption.
  - Benchmark (coherent numbers) plus a paragraph on why E* > 0: the fixed-K
    mandate forces three positions even when none is worth holding.
  - "Simulated Annealing Across Seeds" (label `sec:robustness`) replaces
    "Robustness and Risk Aversion"; `sec:sensitivity` and `tab:sensitivity` no
    longer exist.
- Verified after the rewrite: all refs, labels and citations resolve; braces
  and environments balance. NOT compiled.

---

## 5. Code changes (all in `code/Ising/exp_200_real/` unless noted)

| File | Change |
|---|---|
| `select_contracts.py` | `STRUCTURE_OVERRIDES` (Venezuela -> disjoint; Greenland, TrumpOut, Aliens -> nested); `_strike` parses marker-less decimal strikes (BTC) |
| `benchmark_200.py` | exact ablation and sensitivity (`solve_exact_or_sa`); `energy_offset` (gamma*K^2); `gap_abs` = E - E*; `proved_optimal` False when CBC uses its whole time limit; `add_edges` reads frozen draws by ticker; `make_coherent` + `--coherence {none,cap,match}` (default `cap`) |
| `edges_frozen.csv` (new) | raw model_prob per ticker from the 2026-09-08 run, bit-exact |
| `ablation_many.py` (new) | exact many-forecast ablation; `--check` verifies the fast solver against the full QUBO; `--beta`, `--seeds`, `--modes` |
| `beta_table.csv` (new) | combined beta table assembled from `ablation_many_b{beta}.csv` |

Result folders: `old_K3/` (pre-fix, 2026-09-08), `incoherent_K3/` (post-fix,
incoherent forecast), top level (post-fix, coherent `cap`, used by the paper).
Logs: `select.log`, `build_correlation.log`, `bench200_K3.log`,
`make_tables.log`, `plot_correlation.log`, `ablation_many*.log`.

Known leftover: the Part C line "Energy: mean=..." in `bench200_K3.log` comes
from `common/benchmark.py` and omits gamma*K^2; `robustness_200_K3.csv` has the
corrected values.

---

## 6. Numbers now in the paper (coherent forecast, `cap`)

- Universe and data: as in §2; snapshot 2026-09-08, history 2026-06-11 to
  2026-09-08; median contract moves on 49 of 89 days.
- Ablation, paper draw, beta = 0.5: penalty off -> ATP-MUS + Michigan Senate D
  (0.63) + R (0.37), edge +0.0018, var 0.0196; penalty on (lambda 1-20 identical)
  -> ATP-MUS + OSCARPIC-SOC + WNBA-WSH (all priced 0.02), edge +0.0051, var 0.0586.
- Combined table, penalty on / off (paper draw): beta 0 .0915/.6239 both;
  .05 and .1 .0915/.6032 both; .15 on .0731/.4604, off .0434/.2491 (MI D+R);
  .2-1 on .0051/.0586, off .0018/.0196 (MI D+R); 2 on .0033/.0571, off
  -.0200/.0011 (Fed Sept, all three).
- Benchmark: E* = +0.024132 (enumeration 5.01 s; CBC native 217.82 s, proved);
  SA +0.026021 (gap 0.0019, about 8% of E*, 35.67 s); CBC QUBO +0.028765 (gap
  0.0046, 301.37 s, timed out); SciPy +18 (empty, 5.36 s).
- SA across seeds: 20/20 budget, 20/20 no violation, 19/20 distinct, mean
  +0.0259, sd 0.0006, best +0.0245, none optimal, all 0.0004-0.0031 above E*.

---

## 7. Scaling sweep (`scaling.py`): NOT done, needs fixes first

Partial results from 2026-09-08 (`scaling_results.csv`, `scaling_run.log`),
all with the old code: n = 12, 25, 49 at K = 3 (all solvers agree; CBC on the
QUBO 0.3 -> 3.5 -> 59 s, CBC native about 0.1 s); n = 99, K = 5 only SA
finished before the run was stopped.

Problems to fix before running (same classes as fixed in the benchmark):
1. Edge is additive U(-0.08, 0.08) with no coherence; the paper uses +/-10% of
   price, projected.
2. Energies omit gamma*K^2.
3. `_pulp_solve` marks timed-out CBC runs as proved optimal.
4. Exclusivity comes from the exchange flag only (no overrides).
5. SA read count drops with n (1000 / 200 / 50), which confounds timing.

Design decisions for Adrian:
- Covariance above ~170 contracts: no price history, so structural prior plus
  event formulas only (fine for timing; must be stated).
- Ground truth: enumeration only while C(n,K) <= 5e6 (about n = 60-100 at
  rho = 0.05); beyond that gaps are against the best solution found.
- Time budget: 300 s per solver per size; n up to 1000 may take an hour or more.

---

## 8. Git (2026-09-25)

- Only `code/Ising/` is a git repo (remote `origin` =
  https://github.com/rvraghav2000/Ising.git, branch `main`, last commit
  bea7eaf). The project root, which holds `main_200.tex`, `main.tex`,
  `REVISION_NOTES.md` and `paper_backups/`, is NOT under version control.
- The repo's working tree has an uncommitted reorganisation from an earlier
  session: 23 old top-level files show as deleted and `common/`, `docs/`,
  `exp_12_mock/`, `exp_12_real/`, `exp_200_real/`, `STATUS.md` as untracked.
  A commit now would include that reorganisation, not only today's changes.
- `exp_200_real/kalshi_universe.csv` is 10 MB; the whole folder is 14 MB.
- `.gitignore` already excludes `__pycache__/` and `paper/`.

---

## 9. Remaining work, in order

1. Resolve §8 and commit/push.
2. Compile main_200.tex; check the full-width table and the longer Data section.
3. Update `code/Ising/STATUS.md`.
4. Fix `scaling.py` (§7), agree the design decisions, then run the sweep.
5. Optional: penalty-rescaling test for SA.
