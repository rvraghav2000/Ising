# benchmark.py - runs ablation, solver comparison, and robustness tests

import sys
import time
import numpy as np
import pandas as pd
from itertools import product as iter_product, combinations
from typing import Dict, List, Tuple

from quantum_portfolio_optimizer import (
    generate_kalshi_contracts,
    compute_model_probabilities,
    build_covariance_matrix,
    define_exclusivity_pairs,
    build_qubo,
    qubo_to_ising,
    solve_ising,
    BETA, GAMMA, LAMBDA, K, SEED,
)

SEP  = "=" * 110
THIN = "-" * 110

# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def evaluate_portfolio(
    solution: dict,
    df: pd.DataFrame,
    cov: np.ndarray,
    exclusivity_pairs: List[Tuple[int, int]],
    beta: float = BETA,
) -> dict:
    """compute metrics for a given portfolio"""
    selected = [i for i, v in solution.items() if v == 1]
    n_sel = len(selected)

    total_edge = sum(df.iloc[i]["edge"] for i in selected)

    # portfolio variance  x^T Sigma x
    x = np.array([solution.get(i, 0) for i in range(len(df))])
    port_var = float(x @ cov @ x)

    # risk-adjusted objective  (edge - beta * variance)
    risk_adj = total_edge - beta * port_var

    # exclusivity violations
    violations = []
    for (i, j) in exclusivity_pairs:
        if solution.get(i, 0) == 1 and solution.get(j, 0) == 1:
            violations.append((df.iloc[i]["ticker"], df.iloc[j]["ticker"]))

    # full QUBO energy
    Q = build_qubo(
        df["edge"].values, cov, exclusivity_pairs,
        beta=beta, gamma=GAMMA, lam=LAMBDA, k=K,
    )
    qubo_energy = float(x @ Q @ x)

    return {
        "selected": selected,
        "n_selected": n_sel,
        "tickers": [df.iloc[i]["ticker"] for i in selected],
        "total_edge": total_edge,
        "port_variance": port_var,
        "risk_adjusted": risk_adj,
        "qubo_energy": qubo_energy,
        "budget_ok": n_sel == K,
        "n_violations": len(violations),
        "violations": violations,
    }


# ─────────────────────────────────────────────────────────────
# PART A: Exclusivity Constraint Ablation
# ─────────────────────────────────────────────────────────────

def run_exclusivity_ablation(df, cov, exclusivity_pairs, edges):
    """Compare SA portfolios WITH vs WITHOUT exclusivity penalty."""
    print()
    print(SEP)
    print("  PART A: EXCLUSIVITY CONSTRAINT ABLATION STUDY")
    print("  Proving that lambda * x_i * x_j improves portfolio quality")
    print(SEP)

    lambdas_test = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []

    for lam in lambdas_test:
        Q = build_qubo(edges, cov, exclusivity_pairs,
                       beta=BETA, gamma=GAMMA, lam=lam, k=K)
        h, J, offset = qubo_to_ising(Q)
        sol, energy = solve_ising(h, J, offset, num_reads=5000)

        metrics = evaluate_portfolio(sol, df, cov, exclusivity_pairs)
        results.append({
            "lambda": lam,
            "label": "NO CONSTRAINT" if lam == 0.0 else f"lambda={lam}",
            **metrics,
        })

    # Print comparison table
    print()
    print(f"  {'Lambda':<12} {'#Sel':<6} {'Edge':>9} {'Variance':>10} "
          f"{'Risk-Adj':>10} {'Violations':>11} {'Budget OK':>10}  Tickers")
    print(THIN)
    for r in results:
        tag = "***" if r["n_violations"] > 0 else "   "
        tickers_str = ", ".join(r["tickers"])
        if len(tickers_str) > 55:
            tickers_str = tickers_str[:52] + "..."
        print(f"  {r['lambda']:<12.1f} {r['n_selected']:<6} {r['total_edge']:>+9.4f} "
              f"{r['port_variance']:>10.4f} {r['risk_adjusted']:>+10.4f} "
              f"{r['n_violations']:>11} {str(r['budget_ok']):>10}  "
              f"{tickers_str} {tag}")
    print(THIN)

    # highlight key finding
    no_constraint = results[0]
    best_constrained = min(
        [r for r in results if r["lambda"] > 0],
        key=lambda r: -r["risk_adjusted"] if r["n_violations"] == 0 else 999
    )

    print()
    print("  KEY FINDING:")
    if no_constraint["n_violations"] > 0:
        print(f"  Without exclusivity (lambda=0): {no_constraint['n_violations']} violation(s) detected")
        print(f"    Violated pairs: {no_constraint['violations']}")
        print(f"    This portfolio holds contradictory positions (e.g. betting both")
        print(f"    FOR and AGAINST the same event), which is financially irrational.")
    else:
        print(f"  Without exclusivity (lambda=0): No violations (lucky run), but no guarantee.")

    if best_constrained["n_violations"] == 0:
        print(f"  With exclusivity (lambda={best_constrained['lambda']}): "
              f"0 violations, risk-adj edge = {best_constrained['risk_adjusted']:+.4f}")
        print(f"    The constraint enforces logically consistent portfolios.")
    print(SEP)

    return results


# ─────────────────────────────────────────────────────────────
# PART B: Classical Solver Benchmarking
# ─────────────────────────────────────────────────────────────

def brute_force_solve(Q: np.ndarray, n: int) -> Tuple[dict, float]:
    """Enumerate all 2^n binary vectors, return the one with lowest Q energy."""
    best_energy = np.inf
    best_x = None
    for bits in iter_product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = bits
    solution = {i: int(v) for i, v in enumerate(best_x)}
    return solution, best_energy


def scipy_relaxed_solve(Q: np.ndarray, n: int) -> Tuple[dict, float]:
    """relax to continuous [0,1], optimize, then round to binary"""
    from scipy.optimize import minimize

    def obj(x):
        return float(x @ Q @ x)

    def grad(x):
        return (Q + Q.T) @ x

    best_energy = np.inf
    best_sol = None

    # try multiple random starts
    for trial in range(50):
        x0 = np.random.RandomState(SEED + trial).rand(n)
        bounds = [(0, 1)] * n
        res = minimize(obj, x0, jac=grad, method="L-BFGS-B", bounds=bounds)

        # round the continuous solution to binary
        x_rounded = np.round(res.x).astype(int)
        energy = float(x_rounded @ Q @ x_rounded)
        if energy < best_energy:
            best_energy = energy
            best_sol = x_rounded

    solution = {i: int(v) for i, v in enumerate(best_sol)}
    return solution, best_energy


def pulp_linearized_solve(Q: np.ndarray, n: int) -> Tuple[dict, float]:
    """solve QUBO with PuLP by linearizing the quadratic terms"""
    import pulp

    prob = pulp.LpProblem("QUBO_linearized", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    z = {}
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                z_ij = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
                z[(i, j)] = z_ij
                prob += z_ij <= x[i]
                prob += z_ij <= x[j]
                prob += z_ij >= x[i] + x[j] - 1

    obj = pulp.lpSum(Q[i, i] * x[i] for i in range(n))
    obj += pulp.lpSum(Q[i, j] * z[(i, j)] for (i, j) in z)
    prob += obj

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"PuLP QUBO: non-optimal status {prob.status}")

    solution = {i: int(round(x[i].varValue)) for i in range(n)}
    xvec = np.array([solution[i] for i in range(n)], dtype=float)
    energy = float(xvec @ Q @ xvec)
    return solution, energy


def pulp_constrained_solve(
    edges: np.ndarray,
    cov: np.ndarray,
    exclusivity_pairs: List[Tuple[int, int]],
    beta: float = BETA,
    k: int = K,
) -> Tuple[dict, float]:
    """solve with hard constraints instead of penalties"""
    import pulp

    n = len(edges)
    prob = pulp.LpProblem("Portfolio_constrained", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    z = {}
    for i in range(n):
        for j in range(i + 1, n):
            if cov[i, j] != 0.0:
                z_ij = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
                z[(i, j)] = z_ij
                prob += z_ij <= x[i]
                prob += z_ij <= x[j]
                prob += z_ij >= x[i] + x[j] - 1

    edge_term = pulp.lpSum(-edges[i] * x[i] for i in range(n))
    var_term = pulp.lpSum(beta * cov[i, i] * x[i] for i in range(n))
    var_term += pulp.lpSum(2 * beta * cov[i, j] * z[(i, j)] for (i, j) in z)
    prob += edge_term + var_term

    prob += pulp.lpSum(x) == k, "budget"
    for idx, (i, j) in enumerate(exclusivity_pairs):
        prob += x[i] + x[j] <= 1, f"excl_{idx}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"PuLP constrained: non-optimal status {prob.status}")

    solution = {i: int(round(x[i].varValue)) for i in range(n)}
    xvec = np.array([solution[i] for i in range(n)], dtype=float)
    Q_full = build_qubo(edges, cov, exclusivity_pairs,
                        beta=beta, gamma=GAMMA, lam=LAMBDA, k=k)
    energy = float(xvec @ Q_full @ xvec)
    return solution, energy


def run_solver_benchmark(df, cov, exclusivity_pairs, edges):
    """compare SA against brute force, scipy, and pulp"""
    print()
    print(SEP)
    print("  PART B: CLASSICAL SOLVER BENCHMARKING")
    print("  SA (Ising) vs Brute-Force vs SciPy vs PuLP+CBC")
    print(SEP)

    n = len(df)
    Q = build_qubo(edges, cov, exclusivity_pairs,
                   beta=BETA, gamma=GAMMA, lam=LAMBDA, k=K)

    solvers = {}

    # 1. Simulated Annealing (our method)
    print(f"\n  [1/5] Simulated Annealing (dwave-neal)...")
    t0 = time.perf_counter()
    h, J, offset = qubo_to_ising(Q)
    sa_sol, sa_energy = solve_ising(h, J, offset, num_reads=5000)
    sa_time = time.perf_counter() - t0
    solvers["SA (Ising)"] = (sa_sol, sa_time)
    print(f"        Done in {sa_time:.4f}s  |  Energy: {sa_energy:.4f}")

    # 2. Brute-force exact
    print(f"\n  [2/5] Brute-Force Exact (2^{n} = {2**n} combos)...")
    t0 = time.perf_counter()
    bf_sol, bf_energy = brute_force_solve(Q, n)
    bf_time = time.perf_counter() - t0
    solvers["Brute-Force"] = (bf_sol, bf_time)
    print(f"        Done in {bf_time:.4f}s  |  Energy: {bf_energy:.4f}")

    # 3. SciPy relaxation
    print("\n  [3/5] SciPy Continuous Relaxation + Rounding...")
    t0 = time.perf_counter()
    sp_sol, sp_energy = scipy_relaxed_solve(Q, n)
    sp_time = time.perf_counter() - t0
    solvers["SciPy Relax"] = (sp_sol, sp_time)
    print(f"        Done in {sp_time:.4f}s  |  Energy: {sp_energy:.4f}")

    # 4. PuLP + CBC on linearized QUBO
    try:
        print("\n  [4/5] PuLP + CBC (linearized QUBO, penalty-encoded)...")
        t0 = time.perf_counter()
        pulp_sol, pulp_energy = pulp_linearized_solve(Q, n)
        pulp_time = time.perf_counter() - t0
        solvers["PuLP+CBC (QUBO)"] = (pulp_sol, pulp_time)
        print(f"        Done in {pulp_time:.4f}s  |  Energy: {pulp_energy:.4f}")
    except Exception as e:
        print(f"        PuLP QUBO failed: {e}")

    # 5. PuLP + CBC constrained (native formulation)
    try:
        print("\n  [5/5] PuLP + CBC (hard constraints, native formulation)...")
        t0 = time.perf_counter()
        pulpc_sol, pulpc_energy = pulp_constrained_solve(
            edges, cov, exclusivity_pairs, beta=BETA, k=K
        )
        pulpc_time = time.perf_counter() - t0
        solvers["PuLP+CBC (Native)"] = (pulpc_sol, pulpc_time)
        print(f"        Done in {pulpc_time:.4f}s  |  QUBO Energy: {pulpc_energy:.4f}")
    except Exception as e:
        print(f"        PuLP Constrained failed: {e}")

    # ── Results Table ──
    print()
    print(SEP)
    print("  BENCHMARK RESULTS")
    print(THIN)
    print(f"  {'Solver':<22} {'Energy':>10} {'Edge':>9} {'Variance':>10} "
          f"{'Risk-Adj':>10} {'#Sel':>5} {'Budget':>7} {'Excl Viol':>10} "
          f"{'Time (s)':>10}")
    print(THIN)

    benchmark_rows = []
    for name, (sol, elapsed) in solvers.items():
        m = evaluate_portfolio(sol, df, cov, exclusivity_pairs)
        opt_tag = ""
        if name == "Brute-Force":
            opt_tag = " *"
        print(f"  {name:<22} {m['qubo_energy']:>10.4f} {m['total_edge']:>+9.4f} "
              f"{m['port_variance']:>10.4f} {m['risk_adjusted']:>+10.4f} "
              f"{m['n_selected']:>5} {str(m['budget_ok']):>7} {m['n_violations']:>10} "
              f"{elapsed:>10.4f}{opt_tag}")

        benchmark_rows.append({
            "solver": name,
            "qubo_energy": round(m["qubo_energy"], 6),
            "total_edge": round(m["total_edge"], 6),
            "port_variance": round(m["port_variance"], 6),
            "risk_adjusted": round(m["risk_adjusted"], 6),
            "n_selected": m["n_selected"],
            "budget_ok": m["budget_ok"],
            "n_violations": m["n_violations"],
            "time_s": round(elapsed, 6),
            "tickers": ", ".join(m["tickers"]),
        })

    print(THIN)
    print("  * = provably optimal (brute-force)")

    # ── Optimality gap ──
    bf_row = next(r for r in benchmark_rows if r["solver"] == "Brute-Force")
    bf_energy = bf_row["qubo_energy"]
    print()
    print("  OPTIMALITY GAPS (vs brute-force):")
    for row in benchmark_rows:
        if row["solver"] == "Brute-Force":
            continue
        if bf_energy != 0:
            gap = abs(row["qubo_energy"] - bf_energy) / abs(bf_energy) * 100
        else:
            gap = 0.0
        match_str = "OPTIMAL" if gap < 0.01 else f"gap = {gap:>6.2f}%"
        print(f"    {row['solver']:<22} {match_str}")

    # print selected portfolios per solver
    print()
    print("  SELECTED PORTFOLIOS:")
    for row in benchmark_rows:
        print(f"    {row['solver']:<22} {row['tickers']}")
    print(SEP)

    return benchmark_rows


# ─────────────────────────────────────────────────────────────
# PART C: Multi-seed robustness test for SA
# ─────────────────────────────────────────────────────────────

def run_sa_robustness(df, cov, exclusivity_pairs, edges, n_trials=20):
    """Run SA multiple times with different seeds, report consistency."""
    print()
    print(SEP)
    print(f"  PART C: SA ROBUSTNESS TEST ({n_trials} independent runs)")
    print(SEP)

    n = len(df)
    Q = build_qubo(edges, cov, exclusivity_pairs,
                   beta=BETA, gamma=GAMMA, lam=LAMBDA, k=K)

    from neal import SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()

    energies = []
    violations_list = []
    budget_ok_list = []
    portfolios = []
    times = []

    for trial in range(n_trials):
        t0 = time.perf_counter()
        h, J, offset = qubo_to_ising(Q)
        response = sampler.sample_ising(h, J, num_reads=5000, seed=trial * 7 + 1)
        best = response.first
        sol = {k: int((v + 1) / 2) for k, v in best.sample.items()}
        energy = best.energy + offset
        elapsed = time.perf_counter() - t0

        m = evaluate_portfolio(sol, df, cov, exclusivity_pairs)
        energies.append(m["qubo_energy"])
        violations_list.append(m["n_violations"])
        budget_ok_list.append(m["budget_ok"])
        portfolios.append(tuple(sorted(m["selected"])))
        times.append(elapsed)

    energies = np.array(energies)
    print(f"\n  Energy:  mean={energies.mean():.4f}  std={energies.std():.4f}  "
          f"min={energies.min():.4f}  max={energies.max():.4f}")
    print(f"  Budget satisfied:   {sum(budget_ok_list)}/{n_trials}")
    print(f"  Zero violations:    {sum(1 for v in violations_list if v == 0)}/{n_trials}")
    print(f"  Avg time per run:   {np.mean(times):.4f}s")

    unique_portfolios = set(portfolios)
    print(f"  Unique portfolios:  {len(unique_portfolios)} / {n_trials}")
    for p in unique_portfolios:
        count = portfolios.count(p)
        tickers = [df.iloc[i]["ticker"] for i in p]
        print(f"    {count:>3}x  {', '.join(tickers)}")
    print(SEP)

    return {
        "mean_energy": float(energies.mean()),
        "std_energy": float(energies.std()),
        "min_energy": float(energies.min()),
        "max_energy": float(energies.max()),
        "budget_rate": sum(budget_ok_list) / n_trials,
        "violation_free_rate": sum(1 for v in violations_list if v == 0) / n_trials,
    }


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────

def load_synthetic_data():
    """load the fake contract data"""
    np.random.seed(SEED)
    df = generate_kalshi_contracts()
    df = compute_model_probabilities(df)
    cov = build_covariance_matrix(df)
    edges = df["edge"].values
    exclusivity_pairs = define_exclusivity_pairs(df)
    return df, cov, edges, exclusivity_pairs, "SYNTHETIC"


def load_real_kalshi_data():
    """load real kalshi data from API or cache"""
    # kalshi_data.py lives with the 12-contract real experiment
    import os as _os, sys as _sys
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'exp_12_real')
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    from kalshi_data import load_kalshi_data, compute_model_probabilities as kmp
    np.random.seed(SEED)
    df, cov, exclusivity_pairs = load_kalshi_data(use_cache=True)
    if "edge" not in df.columns:
        df = kmp(df)
    edges = df["edge"].values
    return df, cov, edges, exclusivity_pairs, "REAL KALSHI"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    use_real = "--real" in sys.argv

    if use_real:
        df, cov, edges, exclusivity_pairs, data_label = load_real_kalshi_data()
    else:
        df, cov, edges, exclusivity_pairs, data_label = load_synthetic_data()

    print("\n" + SEP)
    print(f"  QUANTUM PORTFOLIO OPTIMIZER — BENCHMARKING SUITE")
    print(f"  Data source: {data_label}")
    print(SEP)

    print(f"\n  Universe: {len(df)} contracts  |  Budget K={K}  |  "
          f"Exclusivity pairs: {len(exclusivity_pairs)}")
    print(f"  Parameters: beta={BETA}, gamma={GAMMA}, lambda={LAMBDA}")

    # show all contracts
    print(f"\n  {'#':<4} {'Ticker':<42} {'Price':>7} {'Model P':>9} {'Edge':>9} {'Category'}")
    print(THIN)
    for idx, row in df.iterrows():
        print(f"  {idx:<4} {row['ticker']:<42} {row['price']:>7.4f} "
              f"{row['model_prob']:>9.4f} {row['edge']:>+9.4f} {row.get('category', '?')}")
    print(THIN)

    if exclusivity_pairs:
        print("\n  Exclusivity pairs:")
        for i, j in exclusivity_pairs:
            print(f"    {df.iloc[i]['ticker']}  <->  {df.iloc[j]['ticker']}")

    # Part A
    ablation_results = run_exclusivity_ablation(df, cov, exclusivity_pairs, edges)

    # Part B
    benchmark_results = run_solver_benchmark(df, cov, exclusivity_pairs, edges)

    # Part C
    robustness = run_sa_robustness(df, cov, exclusivity_pairs, edges, n_trials=20)

    # ── Save all results ──
    suffix = "_real" if use_real else "_synthetic"

    ablation_df = pd.DataFrame([{
        "lambda": r["lambda"],
        "n_selected": r["n_selected"],
        "total_edge": round(r["total_edge"], 6),
        "port_variance": round(r["port_variance"], 6),
        "risk_adjusted": round(r["risk_adjusted"], 6),
        "n_violations": r["n_violations"],
        "budget_ok": r["budget_ok"],
        "tickers": ", ".join(r["tickers"]),
    } for r in ablation_results])
    ablation_df.to_csv(f"ablation_results{suffix}.csv", index=False)

    benchmark_df = pd.DataFrame(benchmark_results)
    benchmark_df.to_csv(f"benchmark_results{suffix}.csv", index=False)

    print()
    print(SEP)
    print(f"  Results saved to:")
    print(f"    ablation_results{suffix}.csv   (Part A: exclusivity constraint study)")
    print(f"    benchmark_results{suffix}.csv  (Part B: solver comparison)")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
