# /// script
# requires-python = ">=3.10"
# dependencies = ["arch>=6.0", "pandas>=2.0", "numpy>=1.24", "matplotlib>=3.7", "yfinance>=0.2"]
# ///
"""
garch_forecast.py — walk-forward GARCH(1,1) volatility forecasting.

What this does:
  Fits a GARCH(1,1) model walk-forward (no lookahead) and produces a
  1-day-ahead volatility forecast for every day in the sample.

What this does NOT do:
  Predict direction. GARCH forecasts the MAGNITUDE of moves, not which
  way they go. Every output of this module carries that note on purpose.

Usage:
  uv run garch_forecast.py --csv prices.csv            # date + close columns
  uv run garch_forecast.py --ticker BTC-USD            # via yfinance (needs internet)
  uv run garch_forecast.py --csv prices.csv --json     # machine-readable output
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from profiles import add_profile_args, resolve

warnings.filterwarnings("ignore")

TRADING_DAYS_CRYPTO = 365
TRADING_DAYS_EQUITY = 252
MIN_TRAIN = 250          # days of history before first forecast
REFIT_EVERY = 10         # re-estimate params every N days (walk-forward)
REGIME_LOOKBACK = 180    # window for vol percentile / regime classification
CALM_PCTILE = 25         # below this percentile -> calm
STORM_PCTILE = 80        # above this percentile -> storm

HONESTY_NOTE = "GARCH forecasts magnitude (volatility), not direction. It tells you how violent tomorrow is likely to be — not which way it goes."


def load_prices(csv=None, ticker=None):
    """Load a price series from CSV (date + close) or yfinance."""
    if csv:
        df = pd.read_csv(csv)
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = next((cols[k] for k in ("date", "time", "timestamp") if k in cols), df.columns[0])
        px_col = next((cols[k] for k in ("close", "price", "priceusd", "adj close", "adj_close") if k in cols), df.columns[1])
        out = df[[date_col, px_col]].copy()
        out.columns = ["date", "close"]
    elif ticker:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("yfinance not installed. Use --csv, or: uv pip install yfinance")
        data = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        out = data.reset_index()[["Date", "Close"]]
        out.columns = ["date", "close"]
    else:
        sys.exit("Provide --csv or --ticker")

    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna().sort_values("date").reset_index(drop=True)
    out = out[out["close"] > 0]
    return out


def walkforward_garch(prices: pd.DataFrame, periods_per_year: int = TRADING_DAYS_EQUITY,
                      min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY,
                      regime_lookback: int = REGIME_LOOKBACK) -> pd.DataFrame:
    """
    Walk-forward GARCH(1,1). For each day t >= min_train, forecast the vol of
    day t+1 using ONLY data available at the close of day t.

    Params are re-estimated every `refit_every` days on an expanding window.
    Between refits, the GARCH recursion is rolled forward with the last
    fitted params — still zero lookahead, because params were estimated on
    strictly prior data.

    Returns a DataFrame indexed like `prices` with:
      ret          — daily % return
      fcast_vol    — 1-day-ahead conditional vol forecast (daily, %)
      fcast_vol_ann— annualized forecast vol (%)
      vol_pctile   — percentile of today's forecast vs trailing REGIME_LOOKBACK
      regime       — calm / normal / storm
    """
    from arch import arch_model

    px = prices["close"].to_numpy(dtype=float)
    rets = 100.0 * np.diff(px) / px[:-1]           # daily % returns, scaled for arch
    n = len(rets)
    if n < min_train + 10:
        sys.exit(f"Need at least {min_train + 10} days of prices; got {n + 1}.")

    fcast_var = np.full(n, np.nan)                  # forecast of NEXT day's variance, made at t
    omega = alpha = beta = mu = None
    sigma2 = None

    for t in range(min_train, n):
        if (t - min_train) % refit_every == 0:
            # o=1 makes this GJR-GARCH: a separate coefficient on NEGATIVE
            # shocks, so selloffs raise the vol forecast more than rallies of
            # the same size. skewt allows a fat, asymmetric tail.
            am = arch_model(rets[:t], vol="GARCH", p=1, o=1, q=1,
                            mean="Constant", dist="skewt")
            res = am.fit(disp="off", show_warning=False)
            p = res.params
            mu, omega = p["mu"], p["omega"]
            alpha, beta, gamma = p["alpha[1]"], p["beta[1]"], p["gamma[1]"]
            sigma2 = float(res.conditional_volatility[-1] ** 2)
        # roll the recursion one step with today's observed residual.
        # GJR: sigma2_next = omega + (alpha + gamma*1[eps<0])*eps^2 + beta*sigma2
        # The indicator is what makes the forecast asymmetric — without it the
        # gamma term fitted above would be estimated and then thrown away.
        eps = rets[t] - mu if t > 0 else 0.0
        leverage = gamma if eps < 0 else 0.0
        sigma2 = omega + (alpha + leverage) * eps ** 2 + beta * sigma2
        fcast_var[t] = sigma2                        # made at close of t, for day t+1

    out = prices.iloc[1:].copy().reset_index(drop=True)
    out["ret"] = rets
    out["fcast_vol"] = np.sqrt(fcast_var)                                # daily %
    out["fcast_vol_ann"] = out["fcast_vol"] * np.sqrt(periods_per_year)  # annualized %
    pct = out["fcast_vol"].rolling(regime_lookback, min_periods=60).apply(
        lambda w: (w.iloc[:-1] < w.iloc[-1]).mean() * 100 if len(w) > 1 else np.nan, raw=False)
    out["vol_pctile"] = pct
    out["regime"] = pd.cut(out["vol_pctile"],
                           bins=[-1, CALM_PCTILE, STORM_PCTILE, 101],
                           labels=["calm", "normal", "storm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ticker")
    ap.add_argument("--json", action="store_true", help="print latest forecast as JSON")
    ap.add_argument("--out-csv", help="write full walk-forward series to CSV")
    add_profile_args(ap)
    args = ap.parse_args()
    cfg = resolve(args)

    prices = load_prices(csv=args.csv, ticker=args.ticker)
    res = walkforward_garch(prices, periods_per_year=cfg["periods_per_year"],
                            min_train=cfg["min_train"], refit_every=cfg["refit_every"],
                            regime_lookback=cfg["regime_lookback"])
    latest = res.dropna(subset=["fcast_vol"]).iloc[-1]

    payload = {
        "asset": args.ticker or args.csv,
        "profile": args.profile,
        "model": "GJR-GARCH(1,1,1), skew-t",
        "as_of": str(latest["date"].date()),
        "forecast_vol_daily_pct": round(float(latest["fcast_vol"]), 3),
        "forecast_vol_annualized_pct": round(float(latest["fcast_vol_ann"]), 1),
        "vol_percentile": round(float(latest["vol_pctile"]), 1) if pd.notna(latest["vol_pctile"]) else None,
        "regime": str(latest["regime"]),
        "note": HONESTY_NOTE,
    }
    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        payload["series_csv"] = args.out_csv
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n  {payload['asset']} [{args.profile}] — as of {payload['as_of']}")
        print(f"  1-day vol forecast : {payload['forecast_vol_daily_pct']}% daily "
              f"({payload['forecast_vol_annualized_pct']}% annualized)")
        print(f"  vol percentile     : {payload['vol_percentile']}")
        print(f"  regime             : {payload['regime'].upper()}")
        print(f"\n  ⚠ {HONESTY_NOTE}\n")


if __name__ == "__main__":
    main()
