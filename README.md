# GARCH Method — high-beta fork

Volatility forecasting and position sizing. It answers **how much**, never **which way**.

Forked from [milesdeutscher/garchmethod](https://github.com/milesdeutscher/garchmethod)
(MIT) and retuned for a high-beta thematic universe: AI agents / LLMs, autonomous
driving, AR, chips, quantum, cybersecurity, HPC, CPO, eVTOL, IoT, space, rare
earth, biotech, blockchain, fintech, and data-center build-out. Benchmarked
against **ARKK**, not SPY.

---

## What this fork changes

The upstream defaults (target 15% vol, 0.25x size floor) are calibrated for a
diversified equity book. On high-beta assets they **pin the size multiplier at
its floor permanently** — everything above ~60% annualized vol collapses to the
same 0.25x. The sizing signal dies while the script keeps printing a
confident-looking number. Measured on NVDA with upstream settings: **44.9% of
days stuck at the floor**.

| | Upstream | This fork |
|---|---|---|
| Model | GARCH(1,1), Student-t | **GJR-GARCH(1,1,1), skew-t** — separate coefficient on negative shocks |
| Size floor / cap | 0.25x / 2.0x | **0.01x / 2.0x** |
| Sizing rule | `target / forecast` | `(target / forecast) ** k` — convexity exponent |
| Target vol | 15% | 45–75%, by profile |
| Refit cadence | 21 days | 10 days (5 for `quantum`) |
| Min history | 500 bars | 250 bars (200 for `quantum`) |
| Regime bands | 33 / 67 pctile | 25 / 80 pctile |
| Benchmark | none | **ARKK** buy & hold |
| Saturation check | none | `sizing_diagnostics` + warning |

Three things worth calling out:

**GJR asymmetry.** Plain GARCH treats a +12% day and a −12% day as identical
evidence. Risk assets don't work that way — vol explodes on the downside far
more than the upside. The `o=1` term fits that separately. The hand-rolled
variance recursion between refits carries the gamma indicator too; without it
the asymmetry would be estimated and then silently discarded.

**Convexity exponent `k`.** `k=1` is the classic linear rule. `k>1` amplifies
in both directions — more size in calm regimes, sharper cuts in storms. It is
**not free upside**: it buys calm-regime size by paying with steeper
drawdown-side cuts and higher turnover. Always test against `--response-k 1.0`.

**`yfinance` dependency fix.** `--ticker` was documented everywhere upstream but
`yfinance` was never declared in any script's PEP 723 block, so it failed on
every clean machine. Fixed.

## Install

```
/plugin marketplace add SaberYasser/Garch-method
/plugin install garch-method@garchmethod
```

The marketplace is named `garchmethod` in its manifest even though the repo was
renamed to `Garch-method` — the plugin spec above is correct as written.

No API keys, no accounts. Dependencies resolve on first run via `uv`.

## Profiles — always pass one

| profile | target vol | k | band | periods/yr | for |
|---|---|---|---|---|---|
| `high-beta` *(default)* | 50% | 1.35 | 0.01–2.0 | 252 | ARKK-like thematic growth |
| `chips` | 45% | 1.25 | 0.01–2.0 | 252 | AI chips, HPC, CPO, data-center |
| `quantum` | 75% | 1.50 | 0.01–2.0 | 252 | quantum, eVTOL, space, rare earth |
| `biotech` | 65% | 1.20 | 0.01–2.0 | 252 | binary-catalyst names |
| `crypto` | 55% | 1.35 | 0.01–2.0 | 365 | blockchain / digital assets |
| `legacy` | 15% | 1.00 | 0.25–2.0 | 365 | reproduces upstream exactly |

```bash
uv run scripts/garch_forecast.py --ticker ETH-USD --profile crypto --json
uv run scripts/vol_target.py     --ticker IONQ    --profile quantum --json
uv run scripts/compare.py        --ticker NVDA    --profile chips --chart out.png
uv run scripts/compare.py --csv prices.csv --signals mine.csv --profile high-beta
```

Every field is overridable: `--target-vol`, `--min-size`, `--max-leverage`,
`--response-k`, `--min-train`, `--refit-every`, `--regime-lookback`,
`--periods-per-year`, `--benchmark`, `--no-benchmark`.

**Check `sizing_diagnostics` on every run.** If `pct_days_at_floor` or
`pct_days_at_cap` is high, sizing is saturated and the multiplier carries no
information — retune `--target-vol` instead of trusting the number.

---

## Limits — read before sizing real money

**It does not predict direction.** GARCH forecasts the magnitude of moves. A
"storm" reading says moves will be large, not that price will fall. Anyone
reading a direction into it is adding it themselves.

**One-day horizon, daily bars only.** Every forecast is for the *next bar*. It
cannot size an intraday trade, and it produces exactly one number per day. For
swing trading (multi-day holds) the horizon matches. **For day trading it does
not** — the most it honestly offers is a once-daily risk dial for the whole
session.

**Vol targeting does not create edge — it redistributes it.** If a strategy has
no edge, sizing it better yields a smoother path to the same nowhere. On the
ETH demo, vol targeting moved Sharpe 1.02 → 1.07. That 0.05 is well inside
noise for a single asset and should not be treated as a result.

**The backtest is frictionless.** No fees, no slippage, no borrow cost, no
funding, no bid-ask. It assumes every signal fills at the close. Vol targeting
*increases* turnover — you resize daily — so real-world costs bite it harder
than they bite fixed sizing. The gap between backtest and live is wider here
than the equity curve suggests.

**GARCH handles jumps badly.** It assumes volatility diffuses. Exchange hacks,
depegs, halts, FDA decisions, and gap-downs are discrete jumps. The model
under-reacts on the day, then keeps the forecast elevated far longer than
warranted — so you get sized down *after* the damage and stay small through the
recovery.

**Percentile regimes are relative, not absolute.** "Storm" means high versus
that asset's own trailing window. A storm on a sleepy name may be less violent
than a calm day on a microcap. Never compare regime labels across assets.

**Short histories are fragile.** Below `min_train + 10` bars the script exits
rather than guess. Just above it, GARCH parameters are estimated on thin data
and are noisy. Many quantum, eVTOL, and recent-listing names sit in that band.

**Single-asset, no portfolio view.** It sizes one ticker at a time with no
notion of correlation. In this universe nearly everything is a levered bet on
the same handful of factors — sizing ten names "correctly" one by one can still
leave you with one enormous concentrated position.

**Leverage at the 2x cap is the dangerous end.** The cap engages when forecast
vol drops below roughly target ÷ 2 — a *quiet* stretch. Quiet stretches in
fat-tailed assets are exactly where gap risk is priced cheapest and hurts most.

**Selection bias in the demos.** NVDA and ETH both had historic runs. Numbers
from those tickers flatter any long-biased strategy. Test on your own signals
and your own universe, including the names that didn't work.

**Not investment advice.** MIT-licensed, no warranty. This is sizing machinery.

---

## Next step: proper crypto support

The `crypto` profile works today (365 periods/year, 55% target) and ETH-USD runs
end to end. But the *data path* is still equity-shaped. Four concrete gaps, in
the order they should be fixed.

### 1. Drop the incomplete final bar

The highest-value fix and the smallest. `yfinance` returns a partial bar for the
current day. Crypto never closes, so the 00:00 UTC boundary is arbitrary and the
newest row is always mid-formation. Running on a Saturday afternoon produced an
`as_of` of that same day whose return covered ~21 of 24 hours — the newest and
most heavily weighted observation in the GARCH recursion was incomplete.

In `load_prices()`, after sorting:

```python
if drop_partial and out["date"].iloc[-1].date() >= pd.Timestamp.utcnow().date():
    out = out.iloc[:-1]
```

Expose as `--drop-partial-bar`, default **on** for `periods_per_year == 365`.

### 2. Replace yfinance with an exchange feed (`ccxt`)

`yfinance` gives one aggregated crypto series with no exchange granularity,
occasional gaps, and no order-book context. Add `ccxt` as an alternative source:

```python
# dependencies = [..., "ccxt>=4.0"]
import ccxt
ex = getattr(ccxt, exchange)()          # binance, coinbase, kraken
ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", limit=1500)
```

Add `--source {yfinance,ccxt}`, `--exchange`, `--symbol ETH/USDT`. This also
unlocks (3) and (4). Watch the quote currency: `ETH/USDT` and `ETH/USD` differ
by the stablecoin basis, which widens in exactly the stressed conditions the
model is meant to catch.

### 3. Intraday bars

Crypto trades 24/7, so daily bars discard most of the sample. Moving to 4h or 1h
bars gives 6× or 24× the observations — materially better GARCH estimates and a
horizon short enough to be useful intraday. This is also the only route to
making the tool relevant for day trading.

Requires making the period constants coherent rather than hardcoded:

```python
BARS_PER_YEAR = {"1d": 365, "4h": 365*6, "1h": 365*24}
```

`min_train`, `refit_every`, and `regime_lookback` are all currently expressed in
*bars* but reasoned about in *days*. Add a `--timeframe` flag and scale all three
from it, or the 250-bar minimum silently becomes ten days of hourly data.

### 4. Funding rates

For perpetuals, funding is a real carry cost that swings sign and spikes
precisely during storms. Sizing that ignores it understates the cost of holding
through a violent stretch. `ccxt` exposes `fetch_funding_rate_history()`.
Simplest correct treatment: subtract realized funding from the strategy return
series in `compare.py` before computing stats, so the vol-targeted and fixed
columns are both charged for the leverage they actually used.

### Also worth doing, lower priority

- **Jump filter.** Winsorize returns beyond ~6σ before fitting, and report them
  separately. Keeps one depeg from distorting parameters for months.
- **Realized-vol blend.** With intraday bars, compute realized vol from 5-minute
  returns and blend with the GARCH forecast. Realized vol reacts to jumps
  immediately where GARCH lags.
- **Portfolio layer.** Estimate a correlation matrix across held names and scale
  total book exposure, not just per-asset. In a universe this correlated, this
  matters more than any per-asset refinement above.

---

## Pine Script — Storm Gauge

`pine-script/storm-gauge.pine` — TradingView v5 indicator: vol bands, storm tint,
and a corner gauge with size in dollars. Uses *realized* vol (Pine-native math),
not the walk-forward GARCH forecast. Daily chart; 365 periods/year for crypto,
252 for stocks.

## Credit

- **Model family:** Robert Engle (ARCH, Nobel 2003), Tim Bollerslev (GARCH, 1986),
  Glosten-Jagannathan-Runkle (GJR, 1993).
- **Original skill, installer, and Pine:** [Miles Deutscher](https://github.com/milesdeutscher/garchmethod).
- **High-beta fork:** Yasser Saber.

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
