import numpy as np
import pandas as pd
import dimod
from neal import SimulatedAnnealingSampler
from itertools import combinations
from typing import Dict, List, Tuple

BETA   = 0.5
GAMMA  = 2.0
LAMBDA = 5.0
K      = 3
SEED   = 42

np.random.seed(SEED)

def generate_kalshi_contracts() -> pd.DataFrame:
    # fake market data
    contracts = [
        {"ticker": "FED-HIKE-25BPS",     "name": "Fed Hikes >= 25bps in June",      "category": "Fed Rates",  "price": 0.62},
        {"ticker": "FED-HIKE-50BPS",     "name": "Fed Hikes >= 50bps in June",      "category": "Fed Rates",  "price": 0.18},
        {"ticker": "FED-CUT-25BPS",      "name": "Fed Cuts >= 25bps in June",       "category": "Fed Rates",  "price": 0.15},
        {"ticker": "CPI-YOY-GT-3.5",     "name": "CPI YoY > 3.5% in May",        "category": "Inflation",  "price": 0.35},
        {"ticker": "CPI-YOY-GT-4.0",     "name": "CPI YoY > 4.0% in May",        "category": "Inflation",  "price": 0.12},
        {"ticker": "CPI-MOM-GT-0.3",     "name": "CPI MoM > 0.3% in May",        "category": "Inflation",  "price": 0.45},
        {"ticker": "SP500-GT-5200",      "name": "S&P 500 > 5200 on Jun 30",     "category": "S&P 500",    "price": 0.55},
        {"ticker": "SP500-GT-5400",      "name": "S&P 500 > 5400 on Jun 30",     "category": "S&P 500",    "price": 0.30},
        {"ticker": "SP500-LT-5100",      "name": "S&P 500 < 5100 on Jun 30",     "category": "S&P 500",    "price": 0.22},
        {"ticker": "NYC-RAIN-JUN",       "name": "Rain in NYC on Jun 1",          "category": "Weather",    "price": 0.40},
        {"ticker": "TEMP-NYC-GT-90F",    "name": "NYC Temp > 90F in June",       "category": "Weather",    "price": 0.28},
        {"ticker": "HRCN-ATL-JUN",       "name": "Atlantic Hurricane in June",    "category": "Weather",    "price": 0.08},
    ]
    return pd.DataFrame(contracts)


def compute_model_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    # calc random variance
    noise = np.random.uniform(-0.05, 0.05, size=len(df))
    df["model_prob"] = np.clip(df["price"] + noise, 0.01, 0.99)
    df["edge"]       = df["model_prob"] - df["price"]
    return df


    # Math:
    # Var(A) = P(A)(1 - P(A))
    # Cov(A, B) = -P(A)P(B) if mutually exclusive
    # Cov(A, B) = rho * sqrt(Var(A) * Var(B)) otherwise
def build_covariance_matrix(df: pd.DataFrame) -> np.ndarray:
    # builds covariance matrix
    n = len(df)
    categories = df["category"].values
    prices     = df["price"].values
    tickers    = df["ticker"].values

    # disjoint events
    mutually_exclusive = [
        {"SP500-GT-5200", "SP500-LT-5100"},
        {"FED-HIKE-50BPS", "FED-CUT-25BPS"}
    ]

    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sigma[i, j] = prices[i] * (1 - prices[i])
            else:
                pair_set = {tickers[i], tickers[j]}
                if pair_set in mutually_exclusive:
                    sigma[i, j] = - (prices[i] * prices[j])
                else:
                    rho = 0.8 if categories[i] == categories[j] else 0.1
                    sigma[i, j] = rho * np.sqrt(prices[i] * (1 - prices[i]) *
                                                 prices[j] * (1 - prices[j]))
    return sigma

def define_exclusivity_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    # get conflicting contracts
    tickers = df["ticker"].tolist()
    exclusivity_rules = [
        ("SP500-GT-5200", "SP500-LT-5100"),
        ("FED-HIKE-50BPS", "FED-CUT-25BPS"),
        ("CPI-YOY-GT-3.5", "CPI-YOY-GT-4.0"),
    ]
    pairs = []
    for t1, t2 in exclusivity_rules:
        if t1 in tickers and t2 in tickers:
            pairs.append((tickers.index(t1), tickers.index(t2)))
    return pairs


    # Math:
    # Objective: min x^T * Q * x
    # Reward: Q_ii -= edge_i
    # Risk: Q_ij += beta * Cov_ij
    # Budget: gamma * (sum(x_i) - K)^2
    # Exclusivity: lambda * x_i * x_j
def build_qubo(edges: np.ndarray,
               cov: np.ndarray,
               exclusivity_pairs: List[Tuple[int, int]],
               beta: float  = BETA,
               gamma: float = GAMMA,
               lam: float   = LAMBDA,
               k: int       = K) -> np.ndarray:
    # build the matrix
    n = len(edges)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, i] -= edges[i]

    for i in range(n):
        Q[i, i] += beta * cov[i, i]
        
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * beta * cov[i, j]

    for i in range(n):
        Q[i, i] += gamma * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * gamma

    for (i, j) in exclusivity_pairs:
        ii, jj = min(i, j), max(i, j)
        Q[ii, jj] += lam

    return Q

    # Math:
    # Mapping: x_i = (1 + s_i) / 2
    # s_i in {-1, 1}
def qubo_to_ising(Q: np.ndarray):
    # converts qubo to ising
    n = Q.shape[0]
    # build bqm
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_variable(i, Q[i, i])
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                bqm.add_interaction(i, j, Q[i, j])

    # convert to spin vars
    ising_bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
    h = {v: ising_bqm.linear[v] for v in ising_bqm.variables}
    J = {(u, v): bias for (u, v), bias in ising_bqm.quadratic.items()}
    offset = ising_bqm.offset

    return h, J, offset


def solve_ising(h: dict, J: dict, offset: float,
                num_reads: int = 5000) -> Tuple[dict, float]:
    # run simulated annealing
    sampler   = SimulatedAnnealingSampler()
    response  = sampler.sample_ising(h, J,
                                     num_reads=num_reads,
                                     seed=SEED)
    best      = response.first
    binary_solution = {k: int((v + 1) / 2) for k, v in best.sample.items()}
    energy          = best.energy + offset
    return binary_solution, energy




def validate_solution(solution: dict,
                      df: pd.DataFrame,
                      exclusivity_pairs: List[Tuple[int, int]],
                      k: int = K) -> dict:
    # check if solution breaks rules
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
    print("\n--- QUANTUM PORTFOLIO OPTIMIZER ---")

    header = f"{'#':<4} {'Ticker':<22} {'Contract Name':<35} {'Price':>7} {'Model P':>9} {'Edge':>9} {'Select':>8}"
    print(header)

    total_edge = 0.0
    for idx, row in df.iterrows():
        sel = solution.get(idx, 0)
        marker = "  Y" if sel == 1 else "  N"
        edge_str = f"{row['edge']:+.4f}"
        if sel == 1:
            total_edge += row["edge"]
        print(f"{idx:<4} {row['ticker']:<22} {row['name']:<35} {row['price']:>7.2f} {row['model_prob']:>9.4f} {edge_str:>9} {marker:>8}")

    print()
    print(f"Selected Contracts: {validation['num_selected']} / {K} "
          f"{'(Budget OK)' if validation['budget_ok'] else '(BUDGET VIOLATION)'}")
    print(f"Total Expected Edge: {total_edge:+.4f}")
    print(f"QUBO Energy: {energy:.4f}")
    print()

    if validation["exclusivity_ok"]:
        print("Exclusivity Check: No violations detected.")
    else:
        print("Exclusivity Check: VIOLATIONS FOUND:")
        for t1, t2 in validation["exclusivity_violations"]:
            print(f"  {t1}  <->  {t2}")
    print()

    print("SELECTED PORTFOLIO:")
    for i in validation["selected_indices"]:
        row = df.iloc[i]
        print(f"  [{row['ticker']}]  {row['name']}")
        print(f"  Market Price: {row['price']:.2f} | Model P: {row['model_prob']:.4f} | Edge: {row['edge']:+.4f}")
    print()


def main():
    print("\nrunning optimizer...\n")

    print("[1/5] Fetching data...")
    df = generate_kalshi_contracts()
    df = compute_model_probabilities(df)
    print(f"Loaded {len(df)} contracts.\n")

    print("[2/5] Constructing covariance matrix...")
    cov = build_covariance_matrix(df)
    print(f"Matrix shape: {cov.shape}\n")

    print("[3/5] Building QUBO matrix...")
    edges = df["edge"].values
    exclusivity_pairs = define_exclusivity_pairs(df)
    Q = build_qubo(edges, cov, exclusivity_pairs)
    print(f"Q shape: {Q.shape}\n")

    print("[4/5] Converting QUBO to Ising...")
    h, J_couplings, offset = qubo_to_ising(Q)
    print("Solving via dwave-neal...")
    solution, energy = solve_ising(h, J_couplings, offset, num_reads=5000)
    print(f"Solution found. Energy: {energy:.4f}\n")

    print("[5/5] Validating solution...")
    validation = validate_solution(solution, df, exclusivity_pairs)
    print_summary(df, solution, validation, energy)


if __name__ == "__main__":
    main()
