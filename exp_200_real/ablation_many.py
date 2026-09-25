# ablation_many.py - exclusivity ablation repeated over many synthetic forecasts.
#
# The single-draw ablation in benchmark_200.py rests on one forecast (seed 42)
# and one contradictory pair.  A reviewer can fairly ask whether the effect is
# an artifact of that draw, or of the forecast being incoherent (independent
# noise lets "Fed cuts 0 times" + "cuts once" have probability 1.06, so the
# union of the pair looks like riskless edge).  This script answers both:
#
#   for each coherence mode in {none, cap, match} (see benchmark_200.make_coherent)
#     for each of N forecast draws
#       solve lambda = 0 and lambda >= 1 EXACTLY over all C(n,K) portfolios
#       record whether the penalty-free optimum holds an exclusive pair, and
#       the edge / variance of both optima
#
# lambda >= 1 is solved as "best portfolio holding no exclusive pair": the
# penalty term is zero on that set and >= 1 elsewhere, which exceeds any
# objective difference at these scales (checked by --check against the QUBO).
#
# Covariance and prices are fixed across draws, so every portfolio's variance
# and exclusivity flag are computed once; each draw only re-scores the edge.

# --- path bootstrap ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.join(_HERE, '..', 'common'))
# --- end path bootstrap ---

import argparse
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

import quantum_portfolio_optimizer as qpo
import benchmark_200 as b200


def draw_model_prob(df: pd.DataFrame, seed: int, pct: float = b200.EDGE_PCT):
    """raw forecast for one draw; u is assigned in sorted-ticker order so the
    draw does not depend on row order"""
    rng = np.random.RandomState(seed)
    order = np.argsort(df["ticker"].to_numpy())
    u = np.empty(len(df))
    u[order] = rng.uniform(-pct, pct, size=len(df))
    p = df["price"].to_numpy(float)
    return np.clip(p * (1.0 + u), 0.01, 0.99)


def frozen_model_prob(df: pd.DataFrame):
    """the raw draw used for the paper's tables (edges_frozen.csv)"""
    fz = pd.read_csv(b200.EDGES_FROZEN, float_precision="round_trip")
    lookup = dict(zip(fz["ticker"], fz["model_prob"]))
    return df["ticker"].map(lookup).to_numpy(float)


class Enumerator:
    """all C(n,K) portfolios with their variance and exclusivity flag"""

    def __init__(self, cov, pairs, n, K):
        if comb(n, K) > b200.ENUM_CAP:
            raise ValueError(f"C({n},{K}) = {comb(n, K):.3g} exceeds ENUM_CAP")
        count = comb(n, K)
        flat = np.fromiter((i for c in combinations(range(n), K) for i in c),
                           dtype=np.int32, count=count * K)
        self.C = flat.reshape(count, K)

        excl = np.zeros((n, n), bool)
        for i, j in pairs:
            excl[i, j] = excl[j, i] = True

        self.var = np.zeros(count)
        self.n_viol = np.zeros(count, np.int8)
        for a in range(K):
            self.var += cov[self.C[:, a], self.C[:, a]]
            for b in range(a + 1, K):
                self.var += 2.0 * cov[self.C[:, a], self.C[:, b]]
                self.n_viol += excl[self.C[:, a], self.C[:, b]]
        self.feasible = self.n_viol == 0

    def solve(self, edges, beta):
        """(index of lambda=0 optimum, index of lambda>=1 optimum)"""
        edge = edges[self.C].sum(axis=1)
        obj = -edge + beta * self.var
        i0 = int(np.argmin(obj))
        f = np.flatnonzero(self.feasible)
        i1 = int(f[np.argmin(obj[f])])
        return i0, i1, edge, obj


def violated_pairs(sel, pairs_set, tick):
    return [f"{tick[i]}+{tick[j]}" for a, i in enumerate(sel) for j in sel[a + 1:]
            if (min(i, j), max(i, j)) in pairs_set]


def self_check(df, cov, pairs, K, beta, enum):
    """compare the fast solver with enum_cardinality on the full QUBO"""
    from quantum_portfolio_optimizer import build_qubo
    mp = b200.make_coherent(df, frozen_model_prob(df), "none")
    edges = mp - df["price"].to_numpy(float)
    i0, i1, _, obj = enum.solve(edges, beta)
    ok = True
    for lam, idx in [(0.0, i0), (1.0, i1)]:
        Q = build_qubo(edges, cov, pairs, beta=beta, gamma=qpo.GAMMA, lam=lam, k=K)
        sol, e = b200.enum_cardinality(Q, len(df), K)
        ref = sorted(i for i, v in sol.items() if v)
        got = sorted(enum.C[idx].tolist())
        same = ref == got and abs(e + b200.energy_offset(K) - obj[idx]) < 1e-9
        ok &= same
        print(f"  check lambda={lam:g}: QUBO {ref}  fast {got}  "
              f"E_qubo+gK^2={e + b200.energy_offset(K):+.6f}  obj={obj[idx]:+.6f}  "
              f"{'OK' if same else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--beta", type=float, default=qpo.BETA)
    ap.add_argument("--seeds", type=int, default=100, help="forecast draws per mode")
    ap.add_argument("--modes", default="none,cap,match")
    ap.add_argument("--check", action="store_true",
                    help="verify the fast solver against the QUBO before running")
    ap.add_argument("--out", default="ablation_many.csv")
    a = ap.parse_args()

    modes = a.modes.split(",")
    for m in modes:
        if m not in b200.COHERENCE_MODES:
            raise SystemExit(f"unknown mode {m}")

    df = pd.read_csv("contracts_used.csv")
    S = pd.read_csv("covariance_empirical.csv", index_col=0)
    assert list(S.index) == df["ticker"].tolist(), "covariance not aligned"
    cov = S.to_numpy(float)
    p = df["price"].to_numpy(float)
    tick = df["ticker"].tolist()
    pairs = []
    for _, grp in df[df["structure"] == "disjoint"].groupby("event_ticker"):
        pairs.extend(combinations(sorted(grp.index.tolist()), 2))
    pairs_set = set(pairs)
    n, K = len(df), a.K

    print(f"\n=== ABLATION OVER FORECAST DRAWS ===")
    print(f"  n={n}  K={K}  beta={a.beta}  exclusive pairs={len(pairs)}  "
          f"portfolios={comb(n, K):,}  draws/mode={a.seeds}  modes={modes}")
    enum = Enumerator(cov, pairs, n, K)
    print(f"  enumerated; {int(enum.feasible.sum()):,} portfolios hold no exclusive pair")

    if a.check and not self_check(df, cov, pairs, K, a.beta, enum):
        raise SystemExit("fast solver disagrees with the QUBO -- not running")

    rows = []
    for mode in modes:
        draws = [("frozen", frozen_model_prob(df))] + \
                [(s, draw_model_prob(df, s)) for s in range(a.seeds)]
        for seed, raw in draws:
            mp = b200.make_coherent(df, raw, mode)
            edges = mp - p
            i0, i1, edge, obj = enum.solve(edges, a.beta)
            s0, s1 = enum.C[i0].tolist(), enum.C[i1].tolist()
            vp = violated_pairs(s0, pairs_set, tick)
            rows.append(dict(
                mode=mode, seed=seed,
                violation=bool(enum.n_viol[i0]), n_violated_pairs=int(enum.n_viol[i0]),
                edge0=edge[i0], var0=enum.var[i0], obj0=obj[i0],
                edge1=edge[i1], var1=enum.var[i1], obj1=obj[i1],
                var_ratio=enum.var[i0] / enum.var[i1] if enum.var[i1] > 0 else np.nan,
                edge_diff=edge[i0] - edge[i1],
                violated=";".join(vp),
                portfolio0=", ".join(tick[i] for i in s0),
                portfolio1=", ".join(tick[i] for i in s1)))

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    print(f"\n  {'mode':<7} {'draws':>5} {'viol@0':>8} {'var0/var1':>10} "
          f"{'edge0-edge1':>12}   paper draw")
    print("  " + "-" * 70)
    summ = []
    for mode, g in out.groupby("mode", sort=False):
        r = g[g["seed"] != "frozen"]
        v = r[r["violation"]]
        fz = g[g["seed"] == "frozen"].iloc[0]
        s = dict(mode=mode, draws=len(r), frac_violation=r["violation"].mean(),
                 median_var_ratio_when_violated=v["var_ratio"].median() if len(v) else np.nan,
                 median_edge_diff_when_violated=v["edge_diff"].median() if len(v) else np.nan,
                 paper_draw_violation=bool(fz["violation"]))
        summ.append(s)
        print(f"  {mode:<7} {len(r):>5} {s['frac_violation']:>7.0%} "
              f"{s['median_var_ratio_when_violated']:>10.3f} "
              f"{s['median_edge_diff_when_violated']:>+12.4f}   "
              f"{'violation' if fz['violation'] else 'clean'}")
    pd.DataFrame(summ).to_csv(a.out.replace(".csv", "_summary.csv"), index=False)

    print("\n  most frequently violated events (all modes):")
    ev = out.loc[out["violation"], "violated"].str.split(";").explode()
    ev = ev.str.split("+").str[0].str.rsplit("-", n=1).str[0]
    for name, k in ev.value_counts().head(8).items():
        print(f"    {k:4d}  {name}")
    print(f"\n  wrote {a.out} and {a.out.replace('.csv', '_summary.csv')}\n")


if __name__ == "__main__":
    main()
