---
name: garch-method
description: Volatility forecasting and position sizing via walk-forward GJR-GARCH(1,1,1), tuned for a high-beta thematic universe (AI/LLM, autonomous driving, AR, chips, quantum, cybersecurity, HPC, CPO, eVTOL, IoT, space, rare earth, biotech, blockchain, fintech, data-center build-out). Use whenever the user asks about volatility forecasts, position sizing, "how much should I put on", vol targeting, risk throttling, storm/calm regimes, or wants to test whether vol-targeted sizing improves an existing strategy. Benchmarks against ARKK, not SPY. Works on any ticker (yfinance) or any CSV with date + close columns. Answers "how much" — never "which way".
---

# GARCH Method — volatility forecasting + position sizing (high-beta build)

This skill answers the question retail never asks and every fund asks daily: **how much?**

It does NOT predict direction. GARCH forecasts the *magnitude* of moves — how violent tomorrow is likely to be, not which way it goes. Say this to the user whenever presenting results.

## Profiles — always pass one

This build is tuned for high-beta thematic growth. Sizing presets live in
`scripts/profiles.py` and are selected with `--profile`:

| profile | target vol | k | band | periods/yr | use for |
|---|---|---|---|---|---|
| `high-beta` *(default)* | 50% | 1.35 | 0.01–2.0 | 252 | ARKK-like thematic growth |
| `chips` | 45% | 1.25 | 0.01–2.0 | 252 | AI chips, HPC, CPO, data-center |
| `quantum` | 75% | 1.50 | 0.01–2.0 | 252 | quantum, eVTOL, space, rare earth |
| `biotech` | 65% | 1.20 | 0.01–2.0 | 252 | binary-catalyst names |
| `crypto` | 55% | 1.35 | 0.01–2.0 | 365 | blockchain / digital assets |
| `legacy` | 15% | 1.00 | 0.25–2.0 | 365 | upstream defaults, for reproducing old results |

Any field can be overridden: `--target-vol`, `--min-size`, `--max-leverage`,
`--response-k`, `--min-train`, `--refit-every`, `--regime-lookback`,
`--periods-per-year`.

## The model

**GJR-GARCH(1,1,1) with a skew-t innovation distribution.** The `o=1` term
estimates a separate coefficient on *negative* shocks, so a selloff raises the
vol forecast more than a rally of the same size — the leverage effect, which is
pronounced across this universe. Plain symmetric GARCH treats them identically
and de-risks too slowly into drawdowns.

Refits every 10 days (5 for `quantum`) on an expanding window; the recursion
rolls forward between refits using the fitted asymmetry term. Zero lookahead.

## The sizing rule

```
size = (target_vol / forecast_vol) ** k     clipped to [min_size, max_leverage]
```

`k` is the convexity exponent. `k = 1.0` is the classic linear rule. `k > 1`
amplifies the response in **both** directions: bigger in calm regimes, and a
sharper cut in storms. It is not free upside — it buys calm-regime size by
paying with steeper drawdown-side cuts and higher turnover. Always test a new
`k` with `compare.py` against `k = 1.0`.

## The three tools

All scripts live in `scripts/` and run with `uv run` (dependencies resolve automatically via inline metadata — nothing to pip-install).

### 1. `garch_forecast.py` — the forecast
Walk-forward GARCH(1,1), zero lookahead (params re-estimated every 21 days on an expanding window; the recursion rolls forward between refits using only past data).

```
uv run scripts/garch_forecast.py --ticker NVDA --profile chips --json
uv run scripts/garch_forecast.py --ticker IONQ --profile quantum --json
```

Output: 1-day-ahead vol forecast (daily + annualized), vol percentile vs trailing year, regime (calm / normal / storm).

### 2. `vol_target.py` — the size
The entire idea: `size = target_vol / forecast_vol`, capped at [0.25x, 2.0x].

```
uv run scripts/vol_target.py --ticker IONQ --profile quantum --json
```

Output: position size multiplier. "Run 0.6x your baseline" — that's the answer.

### 3. `compare.py` — the honest test
Runs the same signals twice — fixed size vs vol-targeted — and shows both equity curves plus stats side by side. Ships with an EMA 9/21 crossover demo; accepts any strategy via `--signals mine.csv` (columns: date, signal in {-1,0,1}).

```
uv run scripts/compare.py --ticker NVDA --profile chips --chart equity.png --json
uv run scripts/compare.py --csv prices.csv --signals mine.csv --profile high-beta
uv run scripts/compare.py --csv prices.csv --no-benchmark        # offline
```

Benchmarks buy-and-hold **ARKK** by default (override with `--benchmark`,
skip with `--no-benchmark`). SPY is not the relevant opportunity cost for a
high-beta thematic book.

Output: CAGR, ann vol, Sharpe, max drawdown, worst month, final equity — fixed,
vol-targeted, and the ARKK benchmark — plus the equity chart with storm regimes
shaded, plus `sizing_diagnostics` (mean/median multiplier, % of days pinned at
the floor or the cap).

## JSON contract

Every script supports `--json`. Core output shape:

```json
{
  "as_of": "2026-05-23",
  "forecast_vol_annualized_pct": 41.2,
  "vol_percentile_1y": 78.0,
  "regime": "storm",
  "position_size_multiplier": 0.6,
  "note": "GARCH forecasts magnitude (volatility), not direction."
}
```

## Three composition patterns

**A. Sizing layer** — bolt onto any existing strategy. Your strategy decides *if*; this skill decides *how much*. Take the strategy's signal, multiply by `position_size_multiplier`, done.

**B. Risk throttle** — standalone kill-switch. If `regime == "storm"`, cut all exposure to the multiplier regardless of what your signals say. Works with any agent that manages positions.

**C. Comparison harness** — before trusting any strategy, run it through `compare.py` and check whether vol targeting improves its Sharpe / drawdown. If sizing doesn't help, the strategy's edge may be too weak to survive real conditions.

Composes cleanly with regime-direction skills (e.g. Markov-style bull/bear classifiers): their output answers *which way*, this answers *how much*. Multiply the two.

## Defaults & conventions

- Pick the profile from the asset, not the habit. Equities default to 252
  periods/year; only `crypto` uses 365.
- Minimum history: `min_train + 10` observations (260 by default, 210 for
  `quantum`). Many quantum and recent-IPO names sit near this line — below it
  the script exits rather than guessing.
- Data: yfinance ticker (needs internet) or any CSV with date + close columns.
- **Check `sizing_diagnostics` every run.** If `pct_days_at_floor` or
  `pct_days_at_cap` is high, sizing is saturated and carries no information —
  retune `--target-vol` rather than trusting the output.

## Honesty rules (non-negotiable)

1. Never present GARCH output as a direction call.
2. Never hide the drawdown or worst-month numbers when reporting a comparison.
3. If vol targeting does NOT improve the user's strategy, say so plainly — that result is just as valuable.
4. State the profile and `k` alongside any number reported. A multiplier is
   meaningless without the band and exponent that produced it.
5. Never present a `k > 1` result as strictly better than `k = 1` without
   showing the drawdown side of the trade.
6. This is sizing machinery, not investment advice. Fewer than 260
   observations, a saturated multiplier, or a benchmark that failed to fetch
   all make the output less reliable — say so rather than reporting a clean
   number.
