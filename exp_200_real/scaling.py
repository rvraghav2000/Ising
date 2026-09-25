# scaling.py - how solver cost grows with universe size n.
#
# Instances are sampled whole-event-at-a-time from the cached Kalshi universe
# so that event structure (and therefore exclusivity) survives sampling.
# Budget follows a fixed participation ratio K = ceil(rho * n), which is the
# regime in which the feasible set grows exponentially; at constant K the
# problem is only O(n^K) and exhaustive search never becomes hard.

# --- path bootstrap (added by repo reorganisation) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..', 'common'))
# --- end path bootstrap ---

import sys
import time
import argparse
from itertools import combinations, product

import numpy as np
import pandas as pd

from quantum_portfolio_optimizer import build_qubo, qubo_to_ising, BETA, GAMMA, LAMBDA

RHO = 0.05          # participation ratio -> K = max(K_MIN, ceil(RHO * n))
K_MIN = 3           # floor, so small-n instances match the n=12 baseline
NOISE = 0.08        # edge = price + U(-NOISE, NOISE)
SEED = 42

# per-solver wall-clock ceilings (seconds)
TIME_LIMIT_MILP = 300.0
BRUTE_2N_MAX_N = 20          # 2^n enumeration ceiling
ENUM_CARD_MAX = 5_000_000    # C(n,K) enumeration ceiling

# Linearizing a dense QUBO needs one auxiliary binary and three constraints per
# nonzero off-diagonal.  Past this many terms the MILP model cannot even be
# *constructed* in reasonable time or memory, independent of solve time -- which
# is itself the result: the linearization, not the search, is the bottleneck.
MAX_MILP_TERMS = 60_000


# ---------------------------------------------------------------- instances
COHERENCE_TOL = 0.10   # |sum of YES prices - 1| allowed inside an MX event


def coherent_events(univ: pd.DataFrame, tol: float = COHERENCE_TOL):
    """
    Keep only events we can price-check.

    In a mutually-exclusive event exactly one contract settles YES, so the YES
    prices must sum to 1 up to fees and the bid-ask spread.  Many listed
    contracts have never traded, and for those the bid/ask midpoint is a poor
    probability estimate -- roughly 40% of flagged events violate the sum rule
    badly.  Dropping them is a no-arbitrage data-quality filter, not a
    convenience: an event whose YES prices sum to 3.9 carries no usable
    covariance structure.  Non-exclusive events admit no such check and are
    kept as-is.
    """
    mx = univ[univ["mutually_exclusive"]]
    psum = mx.groupby("event_ticker")["price"].sum()
    bad = set(psum[(psum - 1.0).abs() > tol].index)
    return univ[~univ["event_ticker"].isin(bad)]


def build_instance(univ: pd.DataFrame, n: int, seed: int = SEED):
    """sample whole events until we have n contracts"""
    rng = np.random.RandomState(seed)
    univ = coherent_events(univ)

    sizes = univ.groupby("event_ticker").size()
    usable = sizes[(sizes >= 2) & (sizes <= 15)].index.to_numpy()
    rng.shuffle(usable)

    picked, total = [], 0
    for ev in usable:
        block = univ[univ["event_ticker"] == ev]
        if total + len(block) > n:
            continue
        picked.append(block)
        total += len(block)
        if total == n:
            break

    df = pd.concat(picked).reset_index(drop=True)
    if len(df) > n:
        df = df.iloc[:n].reset_index(drop=True)

    noise = rng.uniform(-NOISE, NOISE, size=len(df))
    df["model_prob"] = np.clip(df["price"] + noise, 0.01, 0.99)
    df["edge"] = df["model_prob"] - df["price"]
    return df


def build_covariance(df: pd.DataFrame) -> np.ndarray:
    """Bernoulli diagonal; disjoint pairs exact; else correlation by structure"""
    n = len(df)
    p = df["price"].to_numpy(float)
    ev = df["event_ticker"].to_numpy()
    cat = df["category"].to_numpy()
    mx = df["mutually_exclusive"].to_numpy(bool)

    sd = np.sqrt(p * (1 - p))
    sigma = np.outer(sd, sd)

    same_ev = ev[:, None] == ev[None, :]
    same_cat = cat[:, None] == cat[None, :]
    both_mx = mx[:, None] & mx[None, :]

    rho = np.where(same_ev, 0.4, np.where(same_cat, 0.2, 0.0))
    sigma = sigma * rho

    # contracts in the same mutually-exclusive event cannot both settle YES,
    # so their covariance is exactly -p_i p_j
    disjoint = same_ev & both_mx
    sigma = np.where(disjoint, -np.outer(p, p), sigma)

    np.fill_diagonal(sigma, p * (1 - p))

    lo = np.linalg.eigvalsh(sigma).min()
    if lo < 0:
        sigma = sigma + (-lo + 1e-8) * np.eye(n)
    return sigma


def exclusivity_pairs(df: pd.DataFrame):
    """every pair inside an exchange-flagged mutually-exclusive event"""
    pairs = []
    for _, idx in df.groupby("event_ticker").groups.items():
        idx = list(idx)
        if not bool(df.loc[idx[0], "mutually_exclusive"]):
            continue
        pairs.extend(combinations(sorted(idx), 2))
    return pairs


# ------------------------------------------------------------------ solvers
def energy_of(Q, x):
    return float(x @ Q @ x)


def solve_brute_2n(Q, n, K):
    best_e, best_x = np.inf, None
    for bits in product([0, 1], repeat=n):
        x = np.asarray(bits, float)
        e = energy_of(Q, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e, True


def solve_enum_cardinality(Q, n, K):
    """exact over the feasible set {x : sum x = K}"""
    best_e, best_x = np.inf, None
    for combo in combinations(range(n), K):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        e = energy_of(Q, x)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e, True


def solve_sa(Q, n, K, num_reads):
    from neal import SimulatedAnnealingSampler
    h, J, offset = qubo_to_ising(Q)
    res = SimulatedAnnealingSampler().sample_ising(
        h, J, num_reads=num_reads, seed=SEED)
    best = res.first
    x = np.array([int((best.sample[i] + 1) / 2) for i in range(n)], float)
    return x, energy_of(Q, x), False


def _pulp_solve(prob, xs, n, Q, time_limit):
    import pulp
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    x = np.array([float(round(v.varValue or 0)) for v in xs])
    return x, energy_of(Q, x), status == "Optimal"


def solve_cbc_linearized(Q, n, K, time_limit=TIME_LIMIT_MILP):
    import pulp
    prob = pulp.LpProblem("qubo_lin", pulp.LpMinimize)
    xs = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    terms = [Q[i, i] * xs[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] == 0.0:
                continue
            z = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
            prob += z <= xs[i]
            prob += z <= xs[j]
            prob += z >= xs[i] + xs[j] - 1
            terms.append(Q[i, j] * z)
    prob += pulp.lpSum(terms)
    return _pulp_solve(prob, xs, n, Q, time_limit)


def solve_cbc_native(Q, n, K, cov, edges, pairs, time_limit=TIME_LIMIT_MILP):
    """hard budget + exclusivity constraints instead of penalties"""
    import pulp
    prob = pulp.LpProblem("native", pulp.LpMinimize)
    xs = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    terms = [(-edges[i] + BETA * cov[i, i]) * xs[i] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if cov[i, j] == 0.0:
                continue
            z = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
            prob += z <= xs[i]
            prob += z <= xs[j]
            prob += z >= xs[i] + xs[j] - 1
            terms.append(2 * BETA * cov[i, j] * z)
    prob += pulp.lpSum(terms)
    prob += pulp.lpSum(xs) == K
    for a, b in pairs:
        prob += xs[a] + xs[b] <= 1
    return _pulp_solve(prob, xs, n, Q, time_limit)


# --------------------------------------------------------------------- main
def run(ns, out="scaling_results.csv"):
    from fetch_universe import load_universe
    univ = load_universe(use_cache=True)

    rows = []
    for n in ns:
        K = max(K_MIN, int(np.ceil(RHO * n)))
        df = build_instance(univ, n, seed=SEED)
        n_act = len(df)
        cov = build_covariance(df)
        edges = df["edge"].to_numpy(float)
        pairs = exclusivity_pairs(df)
        Q = build_qubo(edges, cov, pairs, beta=BETA, gamma=GAMMA, lam=LAMBDA, k=K)

        from math import comb  # noqa
        n_cand = comb(n_act, K)
        print(f"\n=== n={n_act}  K={K}  excl_pairs={len(pairs)}  "
              f"C(n,K)={float(n_cand):.3g}")
        sys.stdout.flush()

        num_reads = 1000 if n_act <= 100 else (200 if n_act <= 500 else 50)

        # auxiliary-variable count for the two MILP encodings
        iu = np.triu_indices(n_act, 1)
        nz_Q = int(np.count_nonzero(Q[iu]))
        nz_cov = int(np.count_nonzero(cov[iu]))
        print(f"    (aux binaries: linearized={nz_Q}, native={nz_cov}; "
              f"cap={MAX_MILP_TERMS})")

        jobs = [("SA", lambda: solve_sa(Q, n_act, K, num_reads))]
        if n_act <= BRUTE_2N_MAX_N:
            jobs.append(("BruteForce2n", lambda: solve_brute_2n(Q, n_act, K)))
        if comb(n_act, K) <= ENUM_CARD_MAX:
            jobs.append(("EnumCardinality", lambda: solve_enum_cardinality(Q, n_act, K)))
        skipped = []
        if nz_Q <= MAX_MILP_TERMS:
            jobs.append(("CBC-linearized", lambda: solve_cbc_linearized(Q, n_act, K)))
        else:
            skipped.append(("CBC-linearized", nz_Q))
        if nz_cov <= MAX_MILP_TERMS:
            jobs.append(("CBC-native",
                         lambda: solve_cbc_native(Q, n_act, K, cov, edges, pairs)))
        else:
            skipped.append(("CBC-native", nz_cov))

        for name, nz in skipped:
            rows.append(dict(n=n_act, K=K, solver=name, energy=np.nan,
                             time_s=np.nan, n_selected=-1, budget_ok=False,
                             violations=-1, proved_optimal=False,
                             excl_pairs=len(pairs), num_reads="",
                             aux_vars=nz, status="model_too_large"))
            print(f"    {name:16s} SKIPPED: {nz} auxiliary binaries "
                  f"exceeds cap {MAX_MILP_TERMS}")

        for name, fn in jobs:
            t0 = time.perf_counter()
            try:
                x, e, proved = fn()
                dt = time.perf_counter() - t0
                sel = int(x.sum())
                viol = sum(1 for a, b in pairs if x[a] > 0.5 and x[b] > 0.5)
                rows.append(dict(n=n_act, K=K, solver=name, energy=round(e, 6),
                                 time_s=round(dt, 4), n_selected=sel,
                                 budget_ok=(sel == K), violations=viol,
                                 proved_optimal=proved, excl_pairs=len(pairs),
                                 num_reads=(num_reads if name == "SA" else ""),
                                 aux_vars=(nz_Q if name == "CBC-linearized"
                                           else (nz_cov if name == "CBC-native" else "")),
                                 status="ok"))
                print(f"    {name:16s} E={e:14.5f}  t={dt:9.3f}s  "
                      f"sel={sel:3d}/{K}  viol={viol}  proved={proved}")
            except Exception as exc:
                dt = time.perf_counter() - t0
                rows.append(dict(n=n_act, K=K, solver=name, energy=np.nan,
                                 time_s=round(dt, 4), n_selected=-1,
                                 budget_ok=False, violations=-1,
                                 proved_optimal=False, excl_pairs=len(pairs),
                                 num_reads="", aux_vars="",
                                 status=f"FAIL:{type(exc).__name__}"))
                print(f"    {name:16s} FAILED after {dt:.1f}s: {type(exc).__name__}: {exc}")
            sys.stdout.flush()
            pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nwrote {out}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=str, default="12,25,50,100")
    ap.add_argument("--out", type=str, default="scaling_results.csv")
    a = ap.parse_args()
    run([int(v) for v in a.ns.split(",")], out=a.out)
