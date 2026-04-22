#!/usr/bin/env python3
"""
Sensitivity Analysis for Quantum Portfolio Optimizer
====================================================

Runs the Quantum Optimizer repeatedly with varying 'beta' (risk aversion)
values. For each beta, prints a FULL table of every available contract with
its selection status, so you can see exactly what was on the table and what
the solver picked.
"""

import numpy as np
import pandas as pd
from quantum_portfolio_optimizer import (
    generate_kalshi_contracts,
    compute_model_probabilities,
    build_covariance_matrix,
    define_exclusivity_pairs,
    build_qubo,
    qubo_to_ising,
    solve_ising,
)

SEP  = "=" * 108
THIN = "-" * 108

CATEGORY_EMOJI = {
    "Fed Rates":  "[FED]",
    "Inflation":  "[CPI]",
    "S&P 500":    "[SPX]",
    "Weather":    "[WX] ",
}


def print_master_table(df: pd.DataFrame, exclusivity_pairs):
    """Print every contract that is available to the optimizer."""
    print()
    print(SEP)
    print("  ALL AVAILABLE CONTRACTS  (the full universe of picks)")
    print(SEP)
    hdr = (f"{'#':<4} {'Cat':<7} {'Ticker':<22} {'Contract Name':<36} "
           f"{'Price':>7} {'Model P':>9} {'Edge':>9}  {'Exclusive With'}")
    print(hdr)
    print(THIN)

    excl_map: dict = {i: [] for i in range(len(df))}
    for (i, j) in exclusivity_pairs:
        excl_map[i].append(df.iloc[j]["ticker"])
        excl_map[j].append(df.iloc[i]["ticker"])

    for idx, row in df.iterrows():
        tag      = CATEGORY_EMOJI.get(row["category"], "[?]  ")
        excl_str = ", ".join(excl_map[idx]) if excl_map[idx] else "none"
        edge_str = f"{row['edge']:+.4f}"
        print(f"{idx:<4} {tag} {row['ticker']:<22} "
              f"{row['name']:<36} {row['price']:>7.2f} {row['model_prob']:>9.4f} "
              f"{edge_str:>9}  {excl_str}")

    print(THIN)
    print(f"  Total contracts: {len(df)}  |  "
          f"Categories: {', '.join(df['category'].unique())}")
    print(f"  Exclusivity rules: {len(exclusivity_pairs)} pair(s) identified")
    print(SEP)
    print()


def print_beta_table(df: pd.DataFrame, solution: dict, beta: float,
                     total_edge: float, energy: float):
    """For a given beta, print every contract with PICKED / skipped status."""
    label = ""
    if beta == 0.0:
        label = "(no risk penalty — pure edge maximiser)"
    elif beta >= 2.0:
        label = "(maximum risk aversion)"

    print()
    print(f"  Beta = {beta:.1f}  {label}")
    print(THIN)
    hdr = (f"{'#':<4} {'Ticker':<22} {'Category':<12} {'Price':>7} "
           f"{'Model P':>9} {'Edge':>9}  {'Status'}")
    print(hdr)
    print(THIN)

    for idx, row in df.iterrows():
        sel    = solution.get(idx, 0)
        status = ">>  PICKED  <<" if sel == 1 else "    skipped"
        edge_str = f"{row['edge']:+.4f}"
        print(f"{idx:<4} {row['ticker']:<22} {row['category']:<12} "
              f"{row['price']:>7.2f} {row['model_prob']:>9.4f} "
              f"{edge_str:>9}  {status}")

    print(THIN)
    selected_names = [df.iloc[i]["name"] for i, v in solution.items() if v == 1]
    print(f"  Total Edge : {total_edge:+.4f}   |   QUBO Energy: {energy:.4f}")
    print(f"  Portfolio  : {' | '.join(selected_names)}")
    print()


def run_analysis():
    print()
    print("  Quantum Portfolio Sensitivity Analyzer")
    print()

    # -- 1. Build the universe --
    df = generate_kalshi_contracts()
    df = compute_model_probabilities(df)
    cov = build_covariance_matrix(df)
    edges = df["edge"].values
    exclusivity_pairs = define_exclusivity_pairs(df)

    # -- 2. Show every available contract once --
    print_master_table(df, exclusivity_pairs)

    # -- 3. Sweep Beta --
    betas = np.linspace(0.0, 2.0, 6)   # 6 representative risk levels
    sweep_results = []

    print(SEP)
    print("  BETA SWEEP  --  How portfolio changes with risk aversion")
    print(SEP)

    for beta in betas:
        Q = build_qubo(edges, cov, exclusivity_pairs, beta=float(beta))
        h, J, offset = qubo_to_ising(Q)
        solution, energy = solve_ising(h, J, offset, num_reads=2000)

        selected_idx     = [i for i, v in solution.items() if v == 1]
        selected_tickers = [df.iloc[i]["ticker"] for i in selected_idx]
        total_edge       = float(sum(df.iloc[i]["edge"] for i in selected_idx))

        print_beta_table(df, solution, float(beta), total_edge, energy)

        sweep_results.append({
            "beta":         round(float(beta), 2),
            "total_edge":   round(total_edge, 6),
            "num_selected": len(selected_idx),
            "tickers":      ", ".join(selected_tickers),
        })

    # -- 4. Final summary table --
    res_df = pd.DataFrame(sweep_results)
    print(SEP)
    print("  SWEEP SUMMARY")
    print(THIN)
    print(res_df.to_string(index=False))
    print(THIN)

    res_df.to_csv("sensitivity_results.csv", index=False)
    print("\n  Results saved to sensitivity_results.csv")
    print(SEP)
    print()


if __name__ == "__main__":
    run_analysis()
