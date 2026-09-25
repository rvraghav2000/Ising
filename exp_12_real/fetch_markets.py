"""Quick script to explore Kalshi API and find good markets for our portfolio optimizer."""
import requests
import json

BASE = "https://external-api.kalshi.com/trade-api/v2"

# Series that should have multi-strike events
series_tickers = [
    "KXFEDRATE", "KXCPI", "KXGDP", "KXHIGHNY", 
    "KXBTC", "KXSP500", "KXUNRATE", "KXINR",
    "KXAAAGASD", "KXNATGASD", "KXCORNMON",
    "KXUSPPIYOY", "FED", "INR",
]

for st in series_tickers:
    r = requests.get(f"{BASE}/markets", params={"series_ticker": st, "status": "open", "limit": 15})
    ms = r.json().get("markets", [])
    if ms:
        print(f"\n=== {st} ({len(ms)} open markets) ===")
        for m in ms[:8]:
            price = m.get("last_price_dollars", "?")
            vol = m.get("volume_fp", "0")
            evt = m.get("event_ticker", "?")
            ticker = m.get("ticker", "?")
            title = m.get("title", "?")[:55]
            print(f"  {ticker:45s} price={price:>8s} vol={vol:>10s} evt={evt}")
    else:
        pass  # skip empty
