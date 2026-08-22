# /// script
# requires-python = ">=3.10"
# dependencies = ["arch>=6.0", "pandas>=2.0", "numpy>=1.24", "yfinance>=0.2"]
# ///
"""
vol_target.py — turn a volatility forecast into a position size.

The idea in one line:
    size = (target_vol / forecast_vol) ** k      (clipped to [min_size, max_leverage])

Storm coming -> smaller position. Calm ahead -> bigger position.
Same trades, different sizes. This is the "how much" answer.

The exponent k controls how hard the rule reacts:
    k = 1.0  linear — the classic vol-targeting rule
    k > 1.0  convex — amplifies both directions: more size when the forecast
             is below target, and a sharper cut when it is above
    k < 1.0  damped — sizing moves less than the vol does

k > 1 is NOT free upside. It buys larger positions in calm regimes by paying
with a steeper cut in storms, and it raises turnover. Test it with compare.py
against k = 1.0 before trusting it.

Usage:
  uv run vol_target.py --csv prices.csv --profile high-beta
  uv run vol_target.py --ticker IONQ --profile quantum --json
"""

import argparse
import json

import numpy as np
import pandas as pd

from profiles import add_profile_args, resolve

# Defaults for direct library callers. The CLI resolves these from --profile.
MAX_LEVERAGE = 2.0
MIN_SIZE = 0.01
RESPONSE_K = 1.35
TARGET_VOL = 50.0


def size_from_vol(forecast_vol_ann: float, target_vol_ann: float = TARGET_VOL,
                  max_leverage: float = MAX_LEVERAGE, min_size: float = MIN_SIZE,
                  response_k: float = RESPONSE_K) -> float:
    """Position size multiplier from an annualized vol forecast (%)."""
    if forecast_vol_ann is None or forecast_vol_ann <= 0 or np.isnan(forecast_vol_ann):
        return min_size
    raw = (target_vol_ann / forecast_vol_ann) ** response_k
    return float(np.clip(raw, min_size, max_leverage))


def size_series(fcast_vol_ann: pd.Series, target_vol_ann: float = TARGET_VOL,
                max_leverage: float = MAX_LEVERAGE, min_size: float = MIN_SIZE,
                response_k: float = RESPONSE_K) -> pd.Series:
    """Vectorized version for backtests."""
    ratio = target_vol_ann / fcast_vol_ann.where(fcast_vol_ann > 0)
    return (ratio ** response_k).clip(lower=min_size, upper=max_leverage).fillna(min_size)


def main():
    from garch_forecast import load_prices, walkforward_garch, HONESTY_NOTE

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ticker")
    ap.add_argument("--json", action="store_true")
    add_profile_args(ap)
    args = ap.parse_args()
    cfg = resolve(args)

    prices = load_prices(csv=args.csv, ticker=args.ticker)
    res = walkforward_garch(prices, periods_per_year=cfg["periods_per_year"],
                            min_train=cfg["min_train"], refit_every=cfg["refit_every"],
                            regime_lookback=cfg["regime_lookback"])
    latest = res.dropna(subset=["fcast_vol"]).iloc[-1]
    mult = size_from_vol(float(latest["fcast_vol_ann"]), cfg["target_vol"],
                         cfg["max_leverage"], cfg["min_size"], cfg["response_k"])

    payload = {
        "profile": args.profile,
        "as_of": str(latest["date"].date()),
        "forecast_vol_annualized_pct": round(float(latest["fcast_vol_ann"]), 1),
        "target_vol_pct": cfg["target_vol"],
        "regime": str(latest["regime"]),
        "response_k": cfg["response_k"],
        "position_size_multiplier": round(mult, 3),
        "read_as": f"run {round(mult, 3)}x your baseline position size",
        "caps": {"max": cfg["max_leverage"], "min": cfg["min_size"]},
        "note": HONESTY_NOTE,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n  [{args.profile}] as of {payload['as_of']}: forecast vol "
              f"{payload['forecast_vol_annualized_pct']}% vs target {cfg['target_vol']}% "
              f"({payload['regime'].upper()})")
        print(f"  → position size: {payload['position_size_multiplier']}x baseline\n")


if __name__ == "__main__":
    main()
