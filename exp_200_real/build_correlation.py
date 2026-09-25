"""
build_correlation.py - estimate the covariance structure from Kalshi price
history instead of asserting it from a hand-written table.

Why price increments carry the answer
-------------------------------------
A YES price is the market's running estimate of the outcome:
P_t = E[X | F_t], with P_T = X in {0,1}.  That makes P a martingale, and for
two martingales  P^i P^j - <P^i,P^j>  is itself a martingale.  Taking
expectations at settlement,

    Cov(X_i, X_j) = E[ <P^i, P^j>_T ]

so the covariance of the *outcomes* is the expected quadratic covariation of
the *price paths*.  The diagonal is a consistency check: E[<P^i>_T] = p(1-p),
exactly the Bernoulli variance the model already uses.  Realized co-movement of
prices is therefore the right thing to measure, not a proxy for it.

Estimating it
-------------
We only observe part of [0,T], so we normalise: realized *correlation* over the
observed window, rescaled by the terminal Bernoulli standard deviation.

Contracts inside one event are driven by one underlying quantity, and in a
winner-take-all event they move in opposite directions -- averaging their
correlations against another event would cancel to nothing.  So each event gets
a factor (first principal component of its contracts' increments) and a signed
loading per contract; cross-event correlation is estimated between factors.
That turns ~19,900 unknown contract pairs into ~500 event pairs, which is what
90 days of history can actually support.

Within an event no estimation is needed: event structure fixes the covariance
analytically (disjoint -> -p_i p_j, nested -> min(p_i,p_j) - p_i p_j).
"""

import argparse
import numpy as np
import pandas as pd

MIN_OBS = 60          # candles required before a contract is usable
MIN_EVENT_SIZE = 2


# ----------------------------------------------------------------- utilities
def nearest_psd(sigma: np.ndarray) -> np.ndarray:
    """
    Project onto the positive-semidefinite cone, preserving the diagonal.

    x^T Sigma x is a portfolio's variance, so a negative eigenvalue means some
    portfolio has negative variance -- and since the objective *minimises*
    beta * x^T Sigma x, the optimiser is actively rewarded for finding it.
    Clipping the offending eigenvalues fixes only those directions, unlike
    adding a constant to the whole diagonal, which inflates every variance.
    """
    sym = (sigma + sigma.T) / 2.0
    w, V = np.linalg.eigh(sym)
    if w.min() >= 0:
        return sym
    out = (V * np.clip(w, 0.0, None)) @ V.T
    # restore the exact Bernoulli variances on the diagonal
    d = np.sqrt(np.diag(sym) / np.clip(np.diag(out), 1e-12, None))
    out = out * np.outer(d, d)
    return (out + out.T) / 2.0


def corr_to_cov(corr: np.ndarray, p: np.ndarray) -> np.ndarray:
    sd = np.sqrt(p * (1.0 - p))
    return corr * np.outer(sd, sd)


# ------------------------------------------------------------------ pipeline
def load_panel(contracts: pd.DataFrame, hist: pd.DataFrame):
    """wide price panel (T x n), aligned to the contracts we keep"""
    hist = hist.dropna(subset=["close"])
    wide = hist.pivot_table(index="date", columns="ticker",
                            values="close", aggfunc="last").sort_index()
    wide = wide.ffill()

    good = [t for t in wide.columns
            if wide[t].notna().sum() >= MIN_OBS and wide[t].std(skipna=True) > 1e-9]
    wide = wide[good]

    c = contracts[contracts["ticker"].isin(good)].copy()
    # keep only events that still have enough contracts to define a factor
    sizes = c.groupby("event_ticker").size()
    keep = sizes[sizes >= MIN_EVENT_SIZE].index
    c = c[c["event_ticker"].isin(keep)].reset_index(drop=True)
    wide = wide[c["ticker"].tolist()]
    return c, wide


def event_factors(c: pd.DataFrame, dP: pd.DataFrame):
    """
    One factor per event = first PC of that event's price increments.
    Returns the factor panel (T x E) and a signed loading per contract.
    """
    factors, loadings = {}, {}
    for ev, grp in c.groupby("event_ticker"):
        cols = grp["ticker"].tolist()
        A = dP[cols].to_numpy(float)
        A = np.nan_to_num(A - np.nanmean(A, axis=0), nan=0.0)

        sd = A.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Z = A / sd

        # first principal component
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        f = U[:, 0] * S[0]
        w = Vt[0]

        # orient the factor so it points with the event's largest contract
        if w[np.argmax(np.abs(w))] < 0:
            f, w = -f, -w

        factors[ev] = f
        for t, wi in zip(cols, w):
            # loading = correlation of this contract's increments with the factor
            zi = Z[:, cols.index(t)]
            denom = (np.std(zi) * np.std(f))
            loadings[t] = float(np.corrcoef(zi, f)[0, 1]) if denom > 1e-12 else 0.0
    F = pd.DataFrame(factors, index=dP.index)
    return F, loadings


def shrink_event_correlations(R, events, cat_of, T):
    """
    Ledoit-Wolf style shrinkage of the estimated event-pair correlations toward
    a structural prior.

        R_shrunk = (1 - delta) * R_hat + delta * Prior

    The prior is a *rule* fixed in advance -- same category 0.2, cross category
    0.0 -- not a per-pair judgement made after seeing the estimates, which would
    be cherry-picking.  delta is set by comparing how far the estimates spread
    against how much of that spread is pure sampling noise: with T daily
    increments each correlation carries sampling variance ~(1-rho^2)^2/(T-3), so
    when that noise accounts for most of the observed dispersion, delta -> 1 and
    the prior does the work.  This is the standard remedy when T < n.
    """
    E = len(events)
    iu = np.triu_indices(E, 1)
    off = R[iu]

    prior = np.zeros_like(R)
    for a in range(E):
        for b in range(E):
            if a != b and cat_of.get(events[a]) == cat_of.get(events[b]):
                prior[a, b] = 0.2
    np.fill_diagonal(prior, 1.0)

    noise_var = ((1.0 - off ** 2) ** 2 / max(T - 3, 1)).mean()
    spread = max(off.var(), 1e-12)
    delta = float(np.clip(noise_var / spread, 0.0, 1.0))

    out = (1.0 - delta) * R + delta * prior
    np.fill_diagonal(out, 1.0)
    return out, delta, prior


def structural_block(grp: pd.DataFrame) -> np.ndarray:
    """within-event covariance implied by the event's logical structure"""
    p = grp["price"].to_numpy(float)
    k = len(p)
    struct = grp["structure"].iloc[0]
    if struct == "disjoint":
        blk = -np.outer(p, p)                      # cannot both settle YES
    elif struct == "nested":
        blk = np.minimum.outer(p, p) - np.outer(p, p)
    else:
        blk = np.full((k, k), np.nan)              # no analytic form: estimate
    np.fill_diagonal(blk, p * (1.0 - p))
    return blk


def build(contracts_csv="contracts_200.csv", hist_csv="price_history.csv",
          shrink=True):
    contracts = pd.read_csv(contracts_csv)
    hist = pd.read_csv(hist_csv)

    c, wide = load_panel(contracts, hist)
    dP = wide.diff().iloc[1:]
    n, T = len(c), len(dP)
    print(f"  usable: {n} contracts, {c['event_ticker'].nunique()} events, "
          f"T={T} daily increments")

    # ---- 1. event factors and cross-event correlation --------------------
    F, loadings = event_factors(c, dP)
    events = list(F.columns)
    R_raw = F.corr().to_numpy(float)
    R_raw = np.nan_to_num(R_raw, nan=0.0)
    np.fill_diagonal(R_raw, 1.0)
    print(f"  event factors: {len(events)}  -> {len(events)*(len(events)-1)//2} "
          f"event-pair correlations estimated")

    cat_of = c.groupby("event_ticker")["category"].first().to_dict()
    R_shr, delta, _prior = shrink_event_correlations(R_raw, events, cat_of, T)
    iu = np.triu_indices(len(events), 1)
    print(f"  shrinkage toward structural prior: delta={delta:.3f} "
          f"({100*delta:.0f}% prior / {100*(1-delta):.0f}% data)")
    print(f"    raw    sd={R_raw[iu].std():.3f} range "
          f"[{R_raw[iu].min():+.3f},{R_raw[iu].max():+.3f}]")
    print(f"    shrunk sd={R_shr[iu].std():.3f} range "
          f"[{R_shr[iu].min():+.3f},{R_shr[iu].max():+.3f}]")
    R_ev = R_shr if shrink else R_raw

    # ---- 2. expand to a contract-level correlation ------------------------
    ev_idx = {e: i for i, e in enumerate(events)}
    ei = np.array([ev_idx[e] for e in c["event_ticker"]])
    lam = np.array([loadings.get(t, 0.0) for t in c["ticker"]], float)

    corr = R_ev[np.ix_(ei, ei)] * np.outer(lam, lam)
    np.fill_diagonal(corr, 1.0)

    p = c["price"].to_numpy(float)
    sigma = corr_to_cov(corr, p)

    # ---- 3. overwrite within-event blocks with the analytic form ----------
    realized = dP.corr().to_numpy(float)
    realized = np.nan_to_num(realized, nan=0.0)
    n_struct = 0
    for ev, grp in c.groupby("event_ticker", sort=False):
        idx = grp.index.to_numpy()
        blk = structural_block(grp)
        if np.isnan(blk).any():                     # unstructured -> empirical
            sub = corr_to_cov(realized[np.ix_(idx, idx)], p[idx])
            np.fill_diagonal(sub, p[idx] * (1 - p[idx]))
            blk = sub
        else:
            n_struct += 1
        sigma[np.ix_(idx, idx)] = blk
    print(f"  within-event blocks: {n_struct} analytic, "
          f"{c['event_ticker'].nunique() - n_struct} empirical")

    # ---- 4. enforce PSD ---------------------------------------------------
    w_before = np.linalg.eigvalsh(sigma).min()
    sigma_psd = nearest_psd(sigma)
    w_after = np.linalg.eigvalsh(sigma_psd).min()
    print(f"  min eigenvalue: {w_before:+.6f} -> {w_after:+.6f}")

    ev_raw = pd.DataFrame(R_raw, index=events, columns=events)
    ev_shr = pd.DataFrame(R_shr, index=events, columns=events)
    return c, sigma_psd, ev_raw, ev_shr, delta, loadings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", default="contracts_200.csv")
    ap.add_argument("--history", default="price_history.csv")
    ap.add_argument("--out", default="covariance_empirical.csv")
    ap.add_argument("--out-events", default="event_correlation.csv")
    ap.add_argument("--no-shrink", action="store_true",
                    help="use the raw estimate for the covariance instead of the shrunk one")
    a = ap.parse_args()

    print("\n=== EMPIRICAL COVARIANCE FROM PRICE HISTORY ===\n")
    c, sigma, ev_raw, ev_shr, delta, loadings = build(a.contracts, a.history,
                                                       shrink=not a.no_shrink)

    pd.DataFrame(sigma, index=c["ticker"], columns=c["ticker"]).to_csv(a.out)
    ev_raw.to_csv(a.out_events)
    ev_shr.to_csv(a.out_events.replace(".csv", "_shrunk.csv"))
    c.assign(loading=[loadings.get(t, np.nan) for t in c["ticker"]]) \
     .to_csv("contracts_used.csv", index=False)

    print(f"\n  wrote {a.out}  ({sigma.shape[0]}x{sigma.shape[1]})")
    print(f"  wrote {a.out_events} and *_shrunk.csv  ({ev_raw.shape[0]} events)")
    print(f"  covariance built from the {'SHRUNK' if not a.no_shrink else 'RAW'} "
          f"event correlations (delta={delta:.3f})")
    print(f"  wrote contracts_used.csv (with factor loadings)")

    iu = np.triu_indices(len(ev_raw), 1)
    off = ev_raw.to_numpy()[iu]
    print(f"\n  estimated event-pair correlations: n={len(off)}")
    print(f"    mean={off.mean():+.3f}  sd={off.std():.3f}  "
          f"min={off.min():+.3f}  max={off.max():+.3f}")
    print(f"    |rho| > 0.3 : {(np.abs(off) > 0.3).sum()}")
    print(f"    |rho| > 0.5 : {(np.abs(off) > 0.5).sum()}")
    print()


if __name__ == "__main__":
    main()
