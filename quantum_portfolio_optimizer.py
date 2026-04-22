#!/usr/bin/env python3
"""
Quantum Portfolio Optimizer for Kalshi
=======================================

Fetches (simulated) real-time market data from Kalshi-style binary event contracts,
constructs a QUBO formulation incorporating expected edge, covariance risk,
budget constraints, and exclusivity penalties, then solves via Simulated Annealing
(dwave-neal) after converting to an Ising Hamiltonian.

Author : Quantum Finance Pipeline
Date   : 2026-04-21
"""

import numpy as np
import pandas as pd
import dimod
from neal import SimulatedAnnealingSampler
from itertools import combinations
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration Constants
# ──────────────────────────────────────────────────────────────────────────────
BETA   = 0.5    # Risk-aversion coefficient for covariance penalty
GAMMA  = 2.0    # Budget constraint penalty multiplier
LAMBDA = 5.0    # Exclusivity penalty multiplier
K      = 3      # Target portfolio size (number of contracts to select)
SEED   = 42     # Reproducibility

np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Data Ingestion (Analyst Agent) — Simulated Kalshi Contracts
# ──────────────────────────────────────────────────────────────────────────────

def generate_kalshi_contracts() -> pd.DataFrame:
    """
    Simulate fetching 12 active Kalshi contracts across 4 categories.
    In production, this would call the Kalshi REST API via kalshi-python SDK.
    """
    contracts = [
        # ── Category: Fed Rates ──
        {"ticker": "FED-HIKE-25BPS",     "name": "Fed Hikes ≥25bps in June",      "category": "Fed Rates",  "price": 0.62},
        {"ticker": "FED-HIKE-50BPS",     "name": "Fed Hikes ≥50bps in June",      "category": "Fed Rates",  "price": 0.18},
        {"ticker": "FED-CUT-25BPS",      "name": "Fed Cuts ≥25bps in June",       "category": "Fed Rates",  "price": 0.15},

        # ── Category: Inflation / CPI ──
        {"ticker": "CPI-YOY-GT-3.5",     "name": "CPI YoY > 3.5% in May",        "category": "Inflation",  "price": 0.35},
        {"ticker": "CPI-YOY-GT-4.0",     "name": "CPI YoY > 4.0% in May",        "category": "Inflation",  "price": 0.12},
        {"ticker": "CPI-MOM-GT-0.3",     "name": "CPI MoM > 0.3% in May",        "category": "Inflation",  "price": 0.45},

        # ── Category: S&P 500 Levels ──
        {"ticker": "SP500-GT-5200",      "name": "S&P 500 > 5200 on Jun 30",     "category": "S&P 500",    "price": 0.55},
        {"ticker": "SP500-GT-5400",      "name": "S&P 500 > 5400 on Jun 30",     "category": "S&P 500",    "price": 0.30},
        {"ticker": "SP500-LT-5100",      "name": "S&P 500 < 5100 on Jun 30",     "category": "S&P 500",    "price": 0.22},

        # ── Category: Weather / Miscellaneous ──
        {"ticker": "NYC-RAIN-JUN",       "name": "Rain in NYC on Jun 1",          "category": "Weather",    "price": 0.40},
        {"ticker": "TEMP-NYC-GT-90F",    "name": "NYC Temp > 90°F in June",       "category": "Weather",    "price": 0.28},
        {"ticker": "HRCN-ATL-JUN",       "name": "Atlantic Hurricane in June",    "category": "Weather",    "price": 0.08},
    ]
    return pd.DataFrame(contracts)


def compute_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mock Edge Logic:
    Generate a dummy 'model probability' by adding ±5% random variance to the
    market price. The 'edge' is defined as (model_prob - market_price).
    """
    noise = np.random.uniform(-0.05, 0.05, size=len(df))
    df["model_prob"] = np.clip(df["price"] + noise, 0.01, 0.99)
    df["edge"]       = df["model_prob"] - df["price"]
    return df


def build_covariance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristic Covariance Matrix:
      - Same category  → ρ = 0.8 (Highly correlated)
      - Cross category → ρ = 0.1 (Loosely correlated)
      - MUTUALLY EXCLUSIVE → True negative covariance: Cov(A,B) = -P(A)P(B)
    """
    n = len(df)
    categories = df["category"].values
    prices     = df["price"].values
    tickers    = df["ticker"].values

    # Explicitly identify pairs that cannot physically happen at the same time
    mutually_exclusive = [
        {"SP500-GT-5200", "SP500-LT-5100"},
        {"FED-HIKE-50BPS", "FED-CUT-25BPS"}
    ]

    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sigma[i, j] = prices[i] * (1 - prices[i])       # Bernoulli variance
            else:
                pair_set = {tickers[i], tickers[j]}
                if pair_set in mutually_exclusive:
                    # P(A and B) = 0. Therefore, Cov(A,B) = E[AB] - E[A]E[B] = 0 - P(A)P(B)
                    sigma[i, j] = - (prices[i] * prices[j])
                else:
                    rho = 0.8 if categories[i] == categories[j] else 0.1
                    sigma[i, j] = rho * np.sqrt(prices[i] * (1 - prices[i]) *
                                                 prices[j] * (1 - prices[j]))
    return sigma


# ──────────────────────────────────────────────────────────────────────────────
# 3. QUBO Formulation (Engineer Agent)
# ──────────────────────────────────────────────────────────────────────────────

def define_exclusivity_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Identify mutually exclusive contract pairs based on domain logic.
    Returns a list of (i, j) index tuples.
    """
    tickers = df["ticker"].tolist()
    exclusivity_rules = [
        # S&P above 5200 and below 5100 are contradictory
        ("SP500-GT-5200", "SP500-LT-5100"),
        # Fed hike ≥50bps and Fed cut are contradictory
        ("FED-HIKE-50BPS", "FED-CUT-25BPS"),
        # CPI YoY > 4.0% implies CPI YoY > 3.5%, but we treat the extreme
        # combo as undesirable from a redundancy standpoint
        ("CPI-YOY-GT-3.5", "CPI-YOY-GT-4.0"),
    ]
    pairs = []
    for t1, t2 in exclusivity_rules:
        if t1 in tickers and t2 in tickers:
            pairs.append((tickers.index(t1), tickers.index(t2)))
    return pairs


def build_qubo(edges: np.ndarray,
               cov: np.ndarray,
               exclusivity_pairs: List[Tuple[int, int]],
               beta: float  = BETA,
               gamma: float = GAMMA,
               lam: float   = LAMBDA,
               k: int       = K) -> np.ndarray:
    """
    Construct the QUBO matrix Q ∈ ℝⁿˣⁿ such that:

        min  xᵀ Q x

    Components:
      1. Reward (diagonal):    Q_ii  -= edge_i
      2. Risk   (off-diag):    Q_ij  += β · Σ_ij
      3. Budget penalty:       γ (Σ x_i - K)²  expanded into linear + quadratic
      4. Exclusivity penalty:  λ · x_i · x_j   for each exclusive pair
    """
    n = len(edges)
    Q = np.zeros((n, n))

    # ── 1. Reward: negative edge on diagonal ──
    for i in range(n):
        Q[i, i] -= edges[i]

    # ── 2. Risk: covariance penalty ──
    # Add individual asset variance to the diagonal
    for i in range(n):
        Q[i, i] += beta * cov[i, i]
        
    # Add cross-covariance to the off-diagonal (multiplied by 2 since Q is upper triangular)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * beta * cov[i, j]

    # ── 3. Budget constraint:  γ (Σ x_i - K)²  ──
    #    Expand:  γ [ Σ x_i² + 2 Σ_{i<j} x_i x_j - 2K Σ x_i + K² ]
    #    Since x_i ∈ {0,1}, x_i² = x_i  →  linear terms on diagonal.
    #    The constant K² is irrelevant for optimization.
    for i in range(n):
        Q[i, i] += gamma * (1 - 2 * k)       # from  γ(x_i - 2K x_i)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * gamma              # from  γ · 2 x_i x_j

    # ── 4. Exclusivity penalty ──
    for (i, j) in exclusivity_pairs:
        ii, jj = min(i, j), max(i, j)
        Q[ii, jj] += lam

    return Q


# ──────────────────────────────────────────────────────────────────────────────
# 4. QUBO → Ising Conversion & Solving
# ──────────────────────────────────────────────────────────────────────────────

def qubo_to_ising(Q: np.ndarray):
    """
    Convert QUBO matrix to an Ising Hamiltonian using dimod.

    The mapping is:  x_i = (1 + s_i) / 2   where s_i ∈ {-1, +1}.

    Returns:
        h  : dict of linear biases  {i: h_i}
        J  : dict of quadratic biases {(i,j): J_ij}
        offset : energy offset constant
    """
    n = Q.shape[0]
    # Build a BinaryQuadraticModel in BINARY vartype, then convert
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_variable(i, Q[i, i])
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                bqm.add_interaction(i, j, Q[i, j])

    # Convert to Ising (SPIN vartype)
    ising_bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    h = {v: ising_bqm.linear[v] for v in ising_bqm.variables}
    J = {(u, v): bias for (u, v), bias in ising_bqm.quadratic.items()}
    offset = ising_bqm.offset

    return h, J, offset


def solve_ising(h: dict, J: dict, offset: float,
                num_reads: int = 5000) -> Tuple[dict, float]:
    """
    Solve the Ising problem using dwave-neal Simulated Annealing.
    Returns the best sample (spin → binary mapped) and its energy.
    """
    sampler   = SimulatedAnnealingSampler()
    response  = sampler.sample_ising(h, J,
                                     num_reads=num_reads,
                                     seed=SEED)
    best      = response.first
    # Map spins back to binary:  x_i = (1 + s_i) / 2
    binary_solution = {k: int((v + 1) / 2) for k, v in best.sample.items()}
    energy          = best.energy + offset   # include the offset for true QUBO energy
    return binary_solution, energy


# ──────────────────────────────────────────────────────────────────────────────
# 5. Validation & Reporting
# ──────────────────────────────────────────────────────────────────────────────

def validate_solution(solution: dict,
                      df: pd.DataFrame,
                      exclusivity_pairs: List[Tuple[int, int]],
                      k: int = K) -> dict:
    """
    Verify:
      1. Exactly K contracts selected.
      2. No exclusivity rules violated.
    """
    selected  = [i for i, v in solution.items() if v == 1]
    n_selected = len(selected)

    violations = []
    for (i, j) in exclusivity_pairs:
        if solution.get(i, 0) == 1 and solution.get(j, 0) == 1:
            violations.append((df.iloc[i]["ticker"], df.iloc[j]["ticker"]))

    return {
        "selected_indices": selected,
        "num_selected": n_selected,
        "budget_ok": n_selected == k,
        "exclusivity_violations": violations,
        "exclusivity_ok": len(violations) == 0,
    }


def print_summary(df: pd.DataFrame, solution: dict, validation: dict, energy: float):
    """
    Print a rich summary table and validation report.
    """
    SEP = "═" * 100
    THIN = "─" * 100

    print()
    print(SEP)
    print("  ⚛  QUANTUM PORTFOLIO OPTIMIZER  —  Kalshi Binary Event Contracts")
    print(SEP)
    print()

    # ── Summary Table ──
    header = f"{'#':<4} {'Ticker':<22} {'Contract Name':<35} {'Price':>7} {'Model P':>9} {'Edge':>9} {'Select':>8}"
    print(header)
    print(THIN)

    total_edge = 0.0
    for idx, row in df.iterrows():
        sel = solution.get(idx, 0)
        marker = "  ✔" if sel == 1 else "  ·"
        edge_str = f"{row['edge']:+.4f}"
        if sel == 1:
            total_edge += row["edge"]
        print(f"{idx:<4} {row['ticker']:<22} {row['name']:<35} {row['price']:>7.2f} {row['model_prob']:>9.4f} {edge_str:>9} {marker:>8}")

    print(THIN)
    print()

    # ── Portfolio Summary ──
    print(f"  📊  Selected Contracts : {validation['num_selected']} / {K}  "
          f"{'✅ Budget OK' if validation['budget_ok'] else '❌ BUDGET VIOLATION'}")
    print(f"  💰  Total Expected Edge: {total_edge:+.4f}")
    print(f"  ⚡  QUBO Energy        : {energy:.4f}")
    print()

    # ── Exclusivity Check ──
    if validation["exclusivity_ok"]:
        print("  🔒  Exclusivity Check  : ✅  No violations detected.")
    else:
        print("  🔒  Exclusivity Check  : ❌  VIOLATIONS FOUND:")
        for t1, t2 in validation["exclusivity_violations"]:
            print(f"        ⚠  {t1}  ↔  {t2}")
    print()

    # ── Selected Portfolio Detail ──
    print("  📋  SELECTED PORTFOLIO:")
    print(THIN)
    for i in validation["selected_indices"]:
        row = df.iloc[i]
        print(f"      • [{row['ticker']}]  {row['name']}")
        print(f"        Market Price: {row['price']:.2f}  |  Model P: {row['model_prob']:.4f}  |  Edge: {row['edge']:+.4f}")
    print()
    print(SEP)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n🚀  Initializing Quantum Portfolio Optimizer …\n")

    # ── Step 1: Data Ingestion ──
    print("  [1/5]  Fetching Kalshi contract data …")
    df = generate_kalshi_contracts()
    df = compute_model_probabilities(df)
    print(f"         → {len(df)} contracts loaded across {df['category'].nunique()} categories.\n")

    # ── Step 2: Covariance Matrix ──
    print("  [2/5]  Constructing heuristic covariance matrix …")
    cov = build_covariance_matrix(df)
    print(f"         → Σ shape: {cov.shape}   |  max off-diag: {np.max(np.abs(cov - np.diag(np.diag(cov)))):.4f}\n")

    # ── Step 3: QUBO Construction ──
    print("  [3/5]  Building QUBO matrix …")
    edges = df["edge"].values
    exclusivity_pairs = define_exclusivity_pairs(df)
    print(f"         → Exclusivity pairs: {len(exclusivity_pairs)}")
    for (i, j) in exclusivity_pairs:
        print(f"           ⊘  {df.iloc[i]['ticker']}  ↔  {df.iloc[j]['ticker']}")

    Q = build_qubo(edges, cov, exclusivity_pairs)
    print(f"         → Q shape: {Q.shape}   |  Q range: [{Q.min():.4f}, {Q.max():.4f}]\n")

    # ── Step 4: QUBO → Ising → Solve ──
    print("  [4/5]  Converting QUBO → Ising Hamiltonian …")
    h, J_couplings, offset = qubo_to_ising(Q)
    print(f"         → h terms: {len(h)}   |   J terms: {len(J_couplings)}   |   offset: {offset:.4f}")

    print("         Solving via dwave-neal Simulated Annealing (5000 reads) …")
    solution, energy = solve_ising(h, J_couplings, offset, num_reads=5000)
    print(f"         → Solution found.  QUBO Energy: {energy:.4f}\n")

    # ── Step 5: Validation & Output ──
    print("  [5/5]  Validating solution …")
    validation = validate_solution(solution, df, exclusivity_pairs)
    print_summary(df, solution, validation, energy)


if __name__ == "__main__":
    main()
