# kalshi_data.py - fetches real prediction market data from Kalshi API

import requests
import numpy as np
import pandas as pd
import time
from typing import List, Tuple

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# hand-picked contracts across 4 categories for our 12-asset portfolio
# we intentionally include contracts from the SAME event to test exclusivity
SELECTED_TICKERS = [
    # --- CPI (Inflation) --- same event: KXCPI-26MAY
    "KXCPI-26MAY-T0.6",      # CPI MoM > 0.6%  (high inflation)
    "KXCPI-26MAY-T0.4",      # CPI MoM > 0.4%  (moderate inflation)
    "KXCPI-26MAY-T-0.2",     # CPI MoM > -0.2% (almost always yes)
    # --- GDP --- same event: KXGDP-26JUL30
    "KXGDP-26JUL30-T2.5",    # GDP > 2.5%
    "KXGDP-26JUL30-T1.0",    # GDP > 1.0%
    # --- Weather (NYC High Temp) --- same event: KXHIGHNY-26MAY08
    "KXHIGHNY-26MAY08-T71",  # NYC high > 71F
    "KXHIGHNY-26MAY08-T64",  # NYC high > 64F
    "KXHIGHNY-26MAY08-B64.5",# NYC high 64-65F bracket
    # --- PPI (Producer Prices) --- same event: KXUSPPIYOY-26MAY13
    "KXUSPPIYOY-26MAY13-T5.2", # PPI YoY > 5.2%
    "KXUSPPIYOY-26MAY13-T4.6", # PPI YoY > 4.6%
    "KXUSPPIYOY-26MAY13-T4.2", # PPI YoY > 4.2%
    "KXUSPPIYOY-26MAY13-T3.0", # PPI YoY > 3.0%
]


def fetch_market(ticker: str) -> dict:
    """get one market from API"""
    r = requests.get(f"{BASE_URL}/markets/{ticker}")
    r.raise_for_status()
    return r.json().get("market", {})


def fetch_all_markets(tickers: List[str] = None) -> pd.DataFrame:
    """grab all the contracts we need"""
    if tickers is None:
        tickers = SELECTED_TICKERS

    print(f"  Fetching {len(tickers)} markets from Kalshi API...")
    contracts = []
    for i, ticker in enumerate(tickers):
        try:
            m = fetch_market(ticker)
            # price = last traded price, or midpoint of bid/ask
            price_str = m.get("last_price_dollars", "0")
            price = float(price_str) if price_str else 0.0
            
            if price == 0:
                yes_bid = float(m.get("yes_bid_dollars", "0") or "0")
                yes_ask = float(m.get("yes_ask_dollars", "0") or "0")
                price = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else 0.5

            contracts.append({
                "ticker": ticker,
                "name": m.get("title", ticker),
                "category": _infer_category(ticker),
                "event_ticker": m.get("event_ticker", ""),
                "price": price,
                "volume": float(m.get("volume_fp", "0") or "0"),
                "status": m.get("status", "?"),
            })
            print(f"    [{i+1}/{len(tickers)}] {ticker:40s} price={price:.4f}")
            time.sleep(0.1)  # rate limiting
        except Exception as e:
            print(f"    [{i+1}/{len(tickers)}] {ticker:40s} FAILED: {e}")

    df = pd.DataFrame(contracts)
    return df


def _infer_category(ticker: str) -> str:
    """figure out category from ticker name"""
    if ticker.startswith("KXCPI"):
        return "Inflation (CPI)"
    elif ticker.startswith("KXGDP"):
        return "GDP Growth"
    elif ticker.startswith("KXHIGHNY"):
        return "Weather (NYC)"
    elif ticker.startswith("KXUSPPIYOY"):
        return "Producer Prices (PPI)"
    elif ticker.startswith("KXBTC"):
        return "Crypto (BTC)"
    elif ticker.startswith("KXFEDRATE"):
        return "Fed Rates"
    return "Other"


def _extract_strike(ticker: str) -> float:
    """pull strike value from ticker string"""
    for marker in ["-T", "-B"]:
        if marker in ticker:
            try:
                return float(ticker.split(marker)[-1])
            except ValueError:
                pass
    return float('nan')


def _is_threshold(ticker: str) -> bool:
    return "-T" in ticker


def _is_bracket(ticker: str) -> bool:
    return "-B" in ticker


# cross-category correlations from macro econ reasoning
_CROSS_CORR = {
    frozenset({"Inflation (CPI)", "Producer Prices (PPI)"}):  0.6,
    frozenset({"Inflation (CPI)", "GDP Growth"}):            -0.3,
    frozenset({"Producer Prices (PPI)", "GDP Growth"}):      -0.2,
}


def _cross_category_rho(cat_a: str, cat_b: str) -> float:
    """lookup correlation between two categories"""
    return _CROSS_CORR.get(frozenset({cat_a, cat_b}), 0.0)


def build_covariance_matrix(df: pd.DataFrame) -> np.ndarray:
    """build cov matrix using bernoulli variance + domain knowledge correlations"""
    n = len(df)
    tickers = df["ticker"].tolist()
    events  = df["event_ticker"].tolist()
    prices  = df["price"].values
    cats    = [_infer_category(t) for t in tickers]

    sigma = np.zeros((n, n))

    for i in range(n):
        # diagonal = bernoulli variance
        sigma[i, i] = prices[i] * (1 - prices[i])

        for j in range(i + 1, n):
            pi, pj = prices[i], prices[j]

            if events[i] == events[j]:
                # same event
                if _is_threshold(tickers[i]) and _is_threshold(tickers[j]):
                    # nested thresholds: P(both YES) = min(pi, pj)
                    p_high = min(pi, pj)   # the harder threshold
                    p_low  = max(pi, pj)   # the easier threshold
                    # Cov = E[XiXj] - E[Xi]E[Xj] = p_high - pi*pj
                    cov_ij = p_high - pi * pj
                elif _is_bracket(tickers[i]) and _is_bracket(tickers[j]):
                    # two brackets in the same event are mutually exclusive
                    # Cov = -pi*pj  (can't both settle YES)
                    cov_ij = -pi * pj
                else:
                    # bracket vs threshold — moderate correlation
                    rho = 0.4
                    cov_ij = rho * np.sqrt(pi * (1 - pi) * pj * (1 - pj))

            elif cats[i] == cats[j]:
                # same macro category, different event
                rho = 0.2
                cov_ij = rho * np.sqrt(pi * (1 - pi) * pj * (1 - pj))

            else:
                # cross-category — use macro-economic correlation structure
                rho = _cross_category_rho(cats[i], cats[j])
                cov_ij = rho * np.sqrt(pi * (1 - pi) * pj * (1 - pj))

            sigma[i, j] = cov_ij
            sigma[j, i] = cov_ij

    # ensure positive semi-definite (nudge tiny negative eigenvalues)
    eigvals = np.linalg.eigvalsh(sigma)
    if eigvals.min() < 0:
        sigma += (-eigvals.min() + 1e-8) * np.eye(n)

    return sigma


def detect_exclusivity(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """find pairs of contracts that shouldn't both be held"""
    tickers = df["ticker"].tolist()
    events = df["event_ticker"].tolist()
    prices = df["price"].values

    pairs = []
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            if events[i] != events[j]:
                continue  # different events can't be exclusive

            ti, tj = tickers[i], tickers[j]

            # Rule 1: bracket markets with overlapping ranges
            if "-B" in ti and "-B" in tj:
                pairs.append((i, j))
                continue

            # Rule 2: threshold markets — if one is very high strike and 
            # other is very low strike in same event, they're quasi-exclusive
            # (buying YES on both is irrational hedging)
            if "-T" in ti and "-T" in tj:
                # extract strike values
                try:
                    strike_i = float(ti.split("-T")[-1])
                    strike_j = float(tj.split("-T")[-1])
                except ValueError:
                    continue

                # if strikes are far apart (one high, one low), 
                # and prices suggest near-impossibility of both being "barely" true
                # mark as exclusive for portfolio purposes
                if abs(strike_i - strike_j) > 0 and (prices[i] + prices[j]) < 0.5:
                    pairs.append((i, j))

    # also detect bracket vs threshold conflicts
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] != events[j]:
                continue
            ti, tj = tickers[i], tickers[j]
            if ("-B" in ti and "-T" in tj) or ("-T" in ti and "-B" in tj):
                # bracket and threshold in same event often conflict
                b_ticker = ti if "-B" in ti else tj
                t_ticker = tj if "-B" in ti else ti
                b_idx = i if "-B" in ti else j
                t_idx = j if "-B" in ti else i

                # if bracket range doesn't include threshold, they're exclusive
                try:
                    b_val = float(b_ticker.split("-B")[-1])
                    t_val = float(t_ticker.split("-T")[-1])
                    # bracket is typically a narrow range around b_val
                    # if threshold is well above or below bracket range
                    if abs(b_val - t_val) > 2:
                        if (i, j) not in pairs:
                            pairs.append((i, j))
                except ValueError:
                    continue

    return pairs


def compute_model_probabilities(df: pd.DataFrame, noise_scale: float = 0.08) -> pd.DataFrame:
    """add fake model probs = price + random noise to simulate having an edge"""
    np.random.seed(42)
    noise = np.random.uniform(-noise_scale, noise_scale, size=len(df))
    df["model_prob"] = np.clip(df["price"] + noise, 0.01, 0.99)
    df["edge"] = df["model_prob"] - df["price"]
    return df


def load_kalshi_data(use_cache: bool = True):
    """load data, build cov matrix, find exclusivity pairs"""
    cache_file = "kalshi_cache.csv"
    
    if use_cache:
        try:
            df = pd.read_csv(cache_file)
            print(f"  Loaded cached data from {cache_file}")
            if "edge" not in df.columns:
                df = compute_model_probabilities(df)
            cov = build_covariance_matrix(df)
            excl = detect_exclusivity(df)
            return df, cov, excl
        except FileNotFoundError:
            pass

    # fetch fresh from API
    df = fetch_all_markets()
    
    # check if any contracts were unavailable and try alternates
    if len(df) < 12:
        print(f"\n  WARNING: Only got {len(df)} contracts. Trying alternates...")
        # try fetching more from same series
        alt_tickers = _get_alternate_tickers(df)
        for ticker in alt_tickers:
            if len(df) >= 12:
                break
            if ticker not in df["ticker"].values:
                try:
                    m = fetch_market(ticker)
                    price = float(m.get("last_price_dollars", "0") or "0")
                    if price > 0:
                        df = pd.concat([df, pd.DataFrame([{
                            "ticker": ticker,
                            "name": m.get("title", ticker),
                            "category": _infer_category(ticker),
                            "event_ticker": m.get("event_ticker", ""),
                            "price": price,
                            "volume": float(m.get("volume_fp", "0") or "0"),
                            "status": m.get("status", "?"),
                        }])], ignore_index=True)
                except Exception:
                    pass
    
    df = compute_model_probabilities(df)
    
    # cache
    df.to_csv(cache_file, index=False)
    print(f"\n  Cached {len(df)} contracts to {cache_file}")
    
    # build covariance from domain knowledge
    cov = build_covariance_matrix(df)
    
    # detect exclusivity from event structure
    excl = detect_exclusivity(df)
    
    return df, cov, excl


def _get_alternate_tickers(df: pd.DataFrame) -> List[str]:
    """try to find backup tickers if some are unavailable"""
    alts = []
    try:
        # fetch more CPI markets
        r = requests.get(f"{BASE_URL}/markets", 
                        params={"series_ticker": "KXCPI", "status": "open", "limit": 10})
        for m in r.json().get("markets", []):
            alts.append(m["ticker"])
        
        # fetch more GDP markets
        r = requests.get(f"{BASE_URL}/markets",
                        params={"series_ticker": "KXGDP", "status": "open", "limit": 10})
        for m in r.json().get("markets", []):
            alts.append(m["ticker"])
    except Exception:
        pass
    return alts


if __name__ == "__main__":
    print("\n=== KALSHI REAL DATA LOADER ===\n")
    df, cov, excl = load_kalshi_data(use_cache=False)
    
    print(f"\n--- {len(df)} Contracts Loaded ---")
    for _, row in df.iterrows():
        print(f"  {row['ticker']:40s} price={row['price']:.4f} edge={row['edge']:+.4f} "
              f"cat={row['category']}  event={row['event_ticker']}")
    
    print(f"\n--- Covariance Matrix ({cov.shape}) ---")
    print(np.array2string(cov, precision=6, suppress_small=True))
    
    print(f"\n--- Exclusivity Pairs ({len(excl)}) ---")
    for i, j in excl:
        print(f"  {df.iloc[i]['ticker']}  <->  {df.iloc[j]['ticker']}")
    
    print()
