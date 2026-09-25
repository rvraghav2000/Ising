# benchmark_200.py - ablation, solver comparison and SA robustness on the
# 200-contract Kalshi selection with the empirically estimated covariance.
#
# Reuses the solver routines from benchmark.py; only the data loading and the
# budget K differ.  Nothing in benchmark.py or kalshi_data.py is modified.

# --- path bootstrap (added by repo reorganisation) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..', 'common'))
# --- end path bootstrap ---

import os
import sys
import time
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

import quantum_portfolio_optimizer as qpo
import benchmark as bench
from quantum_portfolio_optimizer import build_qubo, qubo_to_ising

SEED = 42
EDGE_PCT = 0.10      # edge = +/- 10% OF THE LISTED PRICE (relative, not absolute)

SEP = "=" * 110


def _cap_cbc_runtime(limit: float):
    """
    Give every CBC solve a wall-clock ceiling.

    benchmark.py constructs PULP_CBC_CMD(msg=0) with no time limit, which is
    harmless at n=12 but at n=167 the linearized QUBO has ~13,900 auxiliary
    binaries and can run indefinitely.  Wrapping the constructor applies the
    limit without editing benchmark.py.
    """
    import pulp
    original = pulp.PULP_CBC_CMD

    def limited(*args, **kwargs):
        kwargs.setdefault("timeLimit", limit)
        return original(*args, **kwargs)

    pulp.PULP_CBC_CMD = limited


EDGES_FROZEN = "edges_frozen.csv"
COHERENCE_MODES = ("none", "cap", "match")
COHERENCE = "cap"


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """least-squares non-decreasing fit (pool adjacent violators)"""
    blocks = [[float(v), 1] for v in y]          # [mean, weight]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m0, w0 = blocks[i]
            m1, w1 = blocks.pop(i + 1)
            blocks[i] = [(m0 * w0 + m1 * w1) / (w0 + w1), w0 + w1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([np.full(w, m) for m, w in blocks])


def make_coherent(df: pd.DataFrame, model_prob: np.ndarray,
                  mode: str = COHERENCE) -> np.ndarray:
    """
    Project a synthetic forecast onto the event logic.

    Independent noise per contract ignores the structure of an event, so the
    forecast can believe that "Fed cuts 0 times" and "cuts once" together have
    probability 1.06.  A mean-variance optimiser then finds a riskless position
    (the union of the pair) carrying edge that no coherent forecaster could hold.

      none   raw independent noise (the original generator)
      cap    disjoint events: rescale proportionally only where the kept
             contracts' probabilities sum above 1, so the forecast obeys the
             probability axioms and nothing more
      match  disjoint events: rescale so the kept contracts sum to the market's
             own total for them (capped at 1): the forecaster agrees with the
             market on the event's total mass and disagrees only on how it is
             split -- the strictest stress test, no union can carry edge

    In both coherent modes nested ladders are made monotone: ordered by market
    price (the containment order), the forecast is replaced by its isotonic fit,
    so "by 2027" is never more likely than "by 2029".
    """
    if mode not in COHERENCE_MODES:
        raise ValueError(f"coherence mode must be one of {COHERENCE_MODES}")
    mp = np.asarray(model_prob, float).copy()
    if mode == "none":
        return mp
    p = df["price"].to_numpy(float)
    for _, grp in df.groupby("event_ticker"):
        idx = grp.index.to_numpy()
        struct = grp["structure"].iloc[0]
        if struct == "disjoint":
            total = mp[idx].sum()
            target = min(p[idx].sum(), 1.0) if mode == "match" else min(total, 1.0)
            if total > 0:
                mp[idx] *= target / total
        elif struct == "nested":
            order = idx[np.argsort(p[idx], kind="stable")]
            mp[order] = _isotonic_increasing(mp[order])
    return mp


def add_edges(df: pd.DataFrame, pct: float = EDGE_PCT, seed: int = SEED,
              frozen: str = EDGES_FROZEN, coherence: str = COHERENCE):
    """
    Simulate a forecasting advantage as a relative perturbation of the market
    price: model_prob = p * (1 + u), u ~ U(-pct, pct).  A 10% edge on a 0.90
    contract is therefore 0.09, on a 0.05 contract 0.005 -- proportional to the
    price rather than a flat additive shift.

    The draw is positional, so reordering the contract file would hand each
    contract a different u.  To keep the synthetic edge attached to the
    contract, model_prob is looked up by ticker in `frozen` (written from the
    2026-09-08 K=3 run); only tickers missing from it take the seeded draw.
    The frozen values are the raw draws; `coherence` is applied afterwards
    (see make_coherent).
    """
    rng = np.random.RandomState(seed)
    p = df["price"].to_numpy(float)
    u = rng.uniform(-pct, pct, size=len(df))
    model_prob = np.clip(p * (1.0 + u), 0.01, 0.99)

    if frozen and os.path.exists(frozen):
        fz = pd.read_csv(frozen, float_precision="round_trip")
        lookup = dict(zip(fz["ticker"], fz["model_prob"]))
        hit = df["ticker"].map(lookup)
        model_prob = np.where(hit.notna(), hit.to_numpy(float), model_prob)
        n_new = int(hit.isna().sum())
        print(f"  edges: {len(df) - n_new} from {frozen}, {n_new} drawn fresh")

    model_prob = make_coherent(df, model_prob, coherence)
    print(f"  forecast coherence: {coherence}")

    df = df.copy()
    df["model_prob"] = model_prob
    df["edge"] = df["model_prob"] - p
    return df


def load_200(contracts="contracts_used.csv", cov_csv="covariance_empirical.csv",
             coherence=COHERENCE):
    df = pd.read_csv(contracts)
    S = pd.read_csv(cov_csv, index_col=0)
    assert list(S.index) == df["ticker"].tolist(), "covariance not aligned to contracts"
    df = add_edges(df, coherence=coherence)

    pairs = []
    for ev, grp in df[df["structure"] == "disjoint"].groupby("event_ticker"):
        pairs.extend(combinations(sorted(grp.index.tolist()), 2))

    return df, S.to_numpy(float), df["edge"].to_numpy(float), pairs


ENUM_CAP = 5_000_000        # C(n,K) ceiling for exact enumeration
MILP_TIME_LIMIT = 300.0     # seconds, per CBC solve


def enum_cardinality(Q, n, K):
    """
    Exact optimum over the feasible set {x : sum x = K}.

    This replaces the 2^n brute force in benchmark.py, which is fine at n=12 but
    at n=167 would enumerate 1.9e50 states.  Enumerating C(n,K) instead gives the
    same ground truth over the feasible region at a cost of C(167,3)=762,000.
    """
    best_e, best_x = np.inf, None
    for combo in combinations(range(n), K):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        e = float(x @ Q @ x)
        if e < best_e:
            best_e, best_x = e, x
    return {i: int(v) for i, v in enumerate(best_x)}, best_e


def energy_offset(K, gamma=None):
    """
    Constant dropped by build_qubo.

    Expanding gamma * (sum x - K)^2 leaves a constant gamma * K^2 that does not
    affect the minimiser, so build_qubo omits it and x^T Q x is the objective
    minus gamma * K^2.  Adding it back makes reported energies equal the
    objective -edge + beta * variance (+ penalties), so gaps are meaningful.
    """
    g = qpo.GAMMA if gamma is None else gamma
    return g * K ** 2


def solve_exact_or_sa(Q, n, K, num_reads=5000, seeds=(1, 7, 13, 29, 42)):
    """
    Exact optimum over {x : sum x = K} when C(n,K) is small enough to enumerate,
    otherwise best-of-seeds annealing.  The tables that carry the paper's claims
    (ablation, sensitivity) must report the model's optimum, not a sampler's.
    """
    from math import comb
    if comb(n, K) <= ENUM_CAP:
        sol, e = enum_cardinality(Q, n, K)
        return sol, e, "exact"
    sol, e = solve_best_of_seeds(Q, n, num_reads=num_reads, seeds=seeds)
    return sol, e, "SA best-of-%d" % len(seeds)


def solve_best_of_seeds(Q, n, num_reads=5000, seeds=(1, 7, 13, 29, 42)):
    """
    Best result across several annealing seeds.

    A single SA run at this size is not reliable: the ablation returned five
    different portfolios for lambda values that define an identical optimisation
    problem.  Taking the best over seeds removes that solver noise so a table
    reports the model's behaviour rather than the sampler's luck.
    """
    from neal import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    h, J, offset = qubo_to_ising(Q)
    best_x, best_e = None, np.inf
    for s in seeds:
        res = sampler.sample_ising(h, J, num_reads=num_reads, seed=int(s))
        smp = res.first.sample
        x = np.array([int((smp[i] + 1) / 2) for i in range(n)], float)
        e = float(x @ Q @ x)
        if e < best_e:
            best_e, best_x = e, x
    return {i: int(v) for i, v in enumerate(best_x)}, best_e


def run_ablation_200(df, cov, pairs, edges, K, lambdas=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
                     num_reads=5000, seeds=(1, 7, 13, 29, 42)):
    """
    Exclusivity ablation, solved exactly where C(n,K) allows.

    When no pair is violated the lambda term contributes exactly zero, so every
    lambda >= 1 defines an identical optimisation problem and must yield an
    identical portfolio.  Annealing (even best-of-5 seeds) did not: at K=3 it
    returned different, suboptimal portfolios per row and understated the
    lambda=0 edge by half.  Exact enumeration removes the sampler entirely.
    """
    print("\n" + SEP)
    print("  PART A: EXCLUSIVITY ABLATION")
    print(SEP)
    print(f"  {'lambda':>7} {'Edge':>10} {'Variance':>10} {'Risk-Adj':>10} "
          f"{'Viol':>5} {'Budget':>7}  Portfolio")
    print("-" * 110)

    rows = []
    for lam in lambdas:
        Q = build_qubo(edges, cov, pairs, beta=qpo.BETA, gamma=qpo.GAMMA,
                       lam=float(lam), k=K)
        sol, e, method = solve_exact_or_sa(Q, len(df), K, num_reads=num_reads,
                                           seeds=seeds)
        m = bench.evaluate_portfolio(sol, df, cov, pairs)
        rows.append(dict(lam=float(lam), method=method,
                         energy=round(e + energy_offset(K), 6),
                         total_edge=round(m["total_edge"], 6),
                         port_variance=round(m["port_variance"], 6),
                         risk_adjusted=round(m["risk_adjusted"], 6),
                         n_selected=m["n_selected"], budget_ok=m["budget_ok"],
                         n_violations=m["n_violations"],
                         tickers=", ".join(m["tickers"])))
        flag = " ***" if m["n_violations"] else ""
        tick = ", ".join(m["tickers"])
        print(f"  {lam:>7.1f} {m['total_edge']:>+10.4f} {m['port_variance']:>10.4f} "
              f"{m['risk_adjusted']:>+10.4f} {m['n_violations']:>5} "
              f"{str(m['budget_ok']):>7}  {tick[:52]}{flag}")
        sys.stdout.flush()
    print("-" * 110)

    zero = rows[0]
    if zero["n_violations"]:
        print(f"\n  KEY FINDING: at lambda=0 the optimiser selects "
              f"{zero['n_violations']} logically contradictory pair(s), and does so "
              f"at variance {zero['port_variance']:.4f} --")
        lowest = min(r["port_variance"] for r in rows[1:])
        print(f"  lower than every valid portfolio (best valid variance {lowest:.4f}). "
              f"The contradiction is selected as a hedge, not for edge.")
    else:
        print("\n  KEY FINDING: no violation at lambda=0 in this instance.")
    print(SEP)
    return rows


def run_sensitivity_200(df, cov, pairs, edges, K, betas=None, num_reads=5000,
                        seeds=(1, 7, 13, 29, 42)):
    """
    Risk-aversion sweep on the 200-contract selection.

    sensitivity_analysis.py imports load_kalshi_data() directly, so it can only
    ever run on the 12-contract cache.  This is the same sweep against the
    empirical covariance, solved exactly where C(n,K) allows so the rows
    reflect beta rather than sampler noise.
    """
    if betas is None:
        betas = np.linspace(0.0, 2.0, 6)

    print("\n" + SEP)
    print("  PART D: RISK-AVERSION SWEEP (beta)")
    print(SEP)
    print(f"  {'beta':>6} {'Edge':>10} {'Variance':>10} {'Risk-Adj':>10} "
          f"{'Viol':>5}  Portfolio")
    print("-" * 110)

    rows = []
    for b in betas:
        Q = build_qubo(edges, cov, pairs, beta=float(b), gamma=qpo.GAMMA,
                       lam=qpo.LAMBDA, k=K)
        sol, e, method = solve_exact_or_sa(Q, len(df), K, num_reads=num_reads,
                                           seeds=seeds)
        m = bench.evaluate_portfolio(sol, df, cov, pairs, beta=float(b))
        rows.append(dict(beta=round(float(b), 3), method=method,
                         energy=round(e + energy_offset(K), 6),
                         total_edge=round(m["total_edge"], 6),
                         port_variance=round(m["port_variance"], 6),
                         risk_adjusted=round(m["risk_adjusted"], 6),
                         n_selected=m["n_selected"], budget_ok=m["budget_ok"],
                         n_violations=m["n_violations"],
                         tickers=", ".join(m["tickers"])))
        tick = ", ".join(m["tickers"])
        print(f"  {b:>6.2f} {m['total_edge']:>+10.4f} {m['port_variance']:>10.4f} "
              f"{m['risk_adjusted']:>+10.4f} {m['n_violations']:>5}  "
              f"{tick[:58]}")
        sys.stdout.flush()
    print("-" * 110)
    print(SEP)
    return rows


def run_solver_benchmark_200(df, cov, pairs, edges, K):
    """
    Solver comparison sized for n in the hundreds.

    Differences from benchmark.run_solver_benchmark:
      * no 2^n brute force -- exact ground truth comes from enum_cardinality
      * every CBC solve carries a wall-clock limit so a hard instance reports a
        timeout instead of hanging the suite
    """
    from math import comb
    n = len(df)
    Q = build_qubo(edges, cov, pairs, beta=qpo.BETA, gamma=qpo.GAMMA,
                   lam=qpo.LAMBDA, k=K)

    print("\n" + SEP)
    print("  PART B: SOLVER BENCHMARKING")
    print(SEP)

    iu = np.triu_indices(n, 1)
    print(f"  n={n}  K={K}  aux binaries for MILP encodings: "
          f"linearized={int(np.count_nonzero(Q[iu]))}, "
          f"native={int(np.count_nonzero(cov[iu]))}")

    solvers, exact_energy = {}, None

    # --- exact, over the feasible set ------------------------------------
    n_feas = comb(n, K)
    if n_feas <= ENUM_CAP:
        print(f"\n  [exact] enumerating C({n},{K}) = {n_feas:,} feasible portfolios...")
        t0 = time.perf_counter()
        sol, exact_energy = enum_cardinality(Q, n, K)
        dt = time.perf_counter() - t0
        solvers["Exact C(n,K)"] = (sol, dt, True)
        print(f"          E={exact_energy + energy_offset(K):.6f}  t={dt:.2f}s")
    else:
        print(f"\n  [exact] C({n},{K}) = {n_feas:.3g} exceeds cap -- skipped")

    # --- simulated annealing ---------------------------------------------
    print("\n  [SA] simulated annealing (dwave-neal)...")
    t0 = time.perf_counter()
    h, J, off = qubo_to_ising(Q)
    sa_sol, _ = qpo.solve_ising(h, J, off, num_reads=5000)
    dt = time.perf_counter() - t0
    solvers["SA (Ising)"] = (sa_sol, dt, False)
    print(f"       t={dt:.2f}s")

    # --- continuous relaxation + rounding ---------------------------------
    print("\n  [SciPy] continuous relaxation + rounding...")
    t0 = time.perf_counter()
    sp_sol, _ = bench.scipy_relaxed_solve(Q, n)
    dt = time.perf_counter() - t0
    solvers["SciPy relax+round"] = (sp_sol, dt, False)
    print(f"          t={dt:.2f}s")

    # --- CBC, both encodings, time-limited --------------------------------
    import pulp
    for label, builder in [
        ("PuLP+CBC (QUBO)", lambda: bench.pulp_linearized_solve(Q, n)),
        ("PuLP+CBC (native)",
         lambda: bench.pulp_constrained_solve(edges, cov, pairs,
                                              beta=qpo.BETA, k=K)),
    ]:
        print(f"\n  [{label}] ...")
        t0 = time.perf_counter()
        try:
            sol, _ = builder()
            dt = time.perf_counter() - t0
            # PuLP reports "Optimal" even when CBC stops at its time limit with
            # a best-found incumbent, so a run that used the whole budget is
            # not a proof of optimality.
            solvers[label] = (sol, dt, dt < MILP_TIME_LIMIT)
            print(f"          t={dt:.2f}s")
        except Exception as exc:
            dt = time.perf_counter() - t0
            print(f"          FAILED after {dt:.1f}s: {type(exc).__name__}: {exc}")

    # --- table -------------------------------------------------------------
    print("\n" + SEP)
    print(f"  {'Solver':<22} {'Energy':>12} {'Gap':>10} {'Edge':>9} {'Var':>9} "
          f"{'#Sel':>5} {'Budget':>7} {'Viol':>5} {'Time (s)':>10}")
    print("-" * 110)

    # energies include the gamma*K^2 constant, so they equal the objective;
    # the optimum is then near zero and a relative gap is ill-conditioned, so
    # the absolute gap E - E* is the primary measure
    off = energy_offset(K)
    rows = []
    ref = exact_energy if exact_energy is not None else min(
        float(np.array([solvers[k][0].get(i, 0) for i in range(n)], float)
              @ Q @ np.array([solvers[k][0].get(i, 0) for i in range(n)], float))
        for k in solvers)
    ref += off

    for name, (sol, dt, proved) in solvers.items():
        m = bench.evaluate_portfolio(sol, df, cov, pairs)
        e = m["qubo_energy"] + off
        gap = e - ref
        gap_pct = abs(gap) / abs(ref) * 100 if ref else float("nan")
        rows.append(dict(solver=name, energy=round(e, 6),
                         qubo_energy_no_const=round(m["qubo_energy"], 6),
                         gap_abs=round(gap, 6), gap_pct=round(gap_pct, 4),
                         total_edge=round(m["total_edge"], 6),
                         port_variance=round(m["port_variance"], 6),
                         risk_adjusted=round(m["risk_adjusted"], 6),
                         n_selected=m["n_selected"], budget_ok=m["budget_ok"],
                         n_violations=m["n_violations"], proved_optimal=proved,
                         time_s=round(dt, 4), tickers=", ".join(m["tickers"])))
        print(f"  {name:<22} {e:>12.6f} {gap:>+10.6f} {m['total_edge']:>+9.4f} "
              f"{m['port_variance']:>9.4f} {m['n_selected']:>5} "
              f"{str(m['budget_ok']):>7} {m['n_violations']:>5} {dt:>10.2f}")
    print("-" * 110)
    if exact_energy is not None:
        print(f"  energy includes gamma*K^2; gap is E - E* against the exact "
              f"optimum over C({n},{K})")
    else:
        print("  no exact optimum available; gap is against the best solver found")

    print("\n  SELECTED PORTFOLIOS:")
    for r in rows:
        print(f"    {r['solver']:<22} {r['tickers']}")
    print(SEP)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--coherence", choices=COHERENCE_MODES, default=COHERENCE,
                    help="forecast coherence projection (see make_coherent)")
    a = ap.parse_args()

    _cap_cbc_runtime(MILP_TIME_LIMIT)
    df, cov, edges, pairs = load_200(coherence=a.coherence)
    n = len(df)

    # propagate the budget into both modules (their functions read module-level K)
    qpo.K = a.K
    bench.K = a.K
    K = a.K

    print("\n" + SEP)
    print(f"  200-CONTRACT BENCHMARK  |  empirical covariance  |  K={K}")
    print(SEP)
    print(f"  contracts={n}  events={df['event_ticker'].nunique()}  "
          f"exclusivity pairs={len(pairs)}")
    print(f"  beta={qpo.BETA}  gamma={qpo.GAMMA}  lambda={qpo.LAMBDA}  "
          f"edge=+/-{EDGE_PCT:.0%} of price  coherence={a.coherence}")
    print(f"  min eigenvalue of Sigma: {np.linalg.eigvalsh(cov).min():+.2e}")
    print(f"  edge: mean={edges.mean():+.4f} sd={edges.std():.4f} "
          f"min={edges.min():+.4f} max={edges.max():+.4f}")

    # ---- Part A: exclusivity ablation ----------------------------------
    ablation = run_ablation_200(df, cov, pairs, edges, K)

    # ---- Part B: solver benchmark (exact ground truth included) ---------
    benchmark = run_solver_benchmark_200(df, cov, pairs, edges, K)

    # ---- Part C: SA robustness ------------------------------------------
    robust = bench.run_sa_robustness(df, cov, pairs, edges, n_trials=a.trials)
    # benchmark.py reports x^T Q x; shift to the objective like Parts A, B, D
    for key in ("mean_energy", "min_energy", "max_energy"):
        robust[key] += energy_offset(K)
    exact_rows = [r for r in benchmark if r["solver"] == "Exact C(n,K)"]
    if exact_rows:
        robust["exact_energy"] = exact_rows[0]["energy"]
        robust["any_seed_reached_exact"] = bool(
            robust["min_energy"] <= exact_rows[0]["energy"] + 1e-6)
        print(f"  (energies incl. gamma*K^2)  best seed {robust['min_energy']:.6f}  "
              f"exact {exact_rows[0]['energy']:.6f}  "
              f"reached: {robust['any_seed_reached_exact']}")

    # ---- Part D: risk-aversion sweep ------------------------------------
    sensitivity = run_sensitivity_200(df, cov, pairs, edges, K)

    tag = a.tag or f"K{K}"
    pd.DataFrame(ablation).to_csv(f"ablation_200_{tag}.csv", index=False)
    pd.DataFrame(benchmark).to_csv(f"benchmark_200_{tag}.csv", index=False)
    pd.DataFrame([robust]).to_csv(f"robustness_200_{tag}.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(f"sensitivity_200_{tag}.csv", index=False)
    df.to_csv(f"contracts_200_edges_{tag}.csv", index=False)

    print(f"\n  wrote ablation_200_{tag}.csv, benchmark_200_{tag}.csv, "
          f"robustness_200_{tag}.csv, sensitivity_200_{tag}.csv")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
