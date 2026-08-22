"""
profiles.py — sizing presets for high-beta thematic universes.

The upstream defaults (target 15% vol, floor 0.25x) are calibrated for a
diversified equity book. On high-beta growth and crypto they pin the size
multiplier at its floor permanently — every asset above ~60% annualized vol
collapses to the same number, and the sizing signal dies silently.

These profiles retune for a high-beta universe: AI agents / LLMs, autonomous
driving, AR, chips, quantum, cybersecurity, HPC, CPO, eVTOL, IoT, space,
rare earth, biotech, blockchain, fintech, and data-center build-out.

Benchmark is ARKK, not SPY — SPY is not the relevant opportunity cost for
this book.

Fields
------
target_vol       annualized vol (%) the sizing rule aims at
min_size         floor on the size multiplier
max_leverage     cap on the size multiplier
response_k       convexity exponent (see vol_target.size_from_vol)
periods_per_year 252 for equities, 365 for crypto
min_train        days of history before the first forecast
refit_every      days between GARCH parameter re-estimations
regime_lookback  window for the vol percentile / regime label
benchmark        ticker for the buy-and-hold comparison in compare.py
"""

DEFAULT_PROFILE = "high-beta"

PROFILES = {
    # ARKK-like thematic growth: the default for this universe.
    "high-beta": {
        "target_vol": 50.0, "min_size": 0.01, "max_leverage": 2.0,
        "response_k": 1.35, "periods_per_year": 252,
        "min_train": 250, "refit_every": 10, "regime_lookback": 180,
        "benchmark": "ARKK",
    },
    # AI chips, HPC, CPO, data-center build-out. Deep liquidity, big trends.
    "chips": {
        "target_vol": 45.0, "min_size": 0.01, "max_leverage": 2.0,
        "response_k": 1.25, "periods_per_year": 252,
        "min_train": 250, "refit_every": 10, "regime_lookback": 180,
        "benchmark": "ARKK",
    },
    # Quantum, eVTOL, space, rare earth: microcaps, gappy, short history.
    "quantum": {
        "target_vol": 75.0, "min_size": 0.01, "max_leverage": 2.0,
        "response_k": 1.5, "periods_per_year": 252,
        "min_train": 200, "refit_every": 5, "regime_lookback": 120,
        "benchmark": "ARKK",
    },
    # Biotech: binary catalyst risk, vol clustering breaks down around events.
    "biotech": {
        "target_vol": 65.0, "min_size": 0.01, "max_leverage": 2.0,
        "response_k": 1.2, "periods_per_year": 252,
        "min_train": 250, "refit_every": 10, "regime_lookback": 180,
        "benchmark": "ARKK",
    },
    # Blockchain / digital assets traded 24/7.
    "crypto": {
        "target_vol": 55.0, "min_size": 0.01, "max_leverage": 2.0,
        "response_k": 1.35, "periods_per_year": 365,
        "min_train": 250, "refit_every": 10, "regime_lookback": 180,
        "benchmark": "ARKK",
    },
    # Upstream behaviour, unchanged. Kept so results stay reproducible.
    "legacy": {
        "target_vol": 15.0, "min_size": 0.25, "max_leverage": 2.0,
        "response_k": 1.0, "periods_per_year": 365,
        "min_train": 500, "refit_every": 21, "regime_lookback": 365,
        "benchmark": None,
    },
}


def get_profile(name: str = DEFAULT_PROFILE) -> dict:
    """Look up a profile by name. Raises with the valid list on a typo."""
    key = (name or DEFAULT_PROFILE).lower().strip()
    if key not in PROFILES:
        raise SystemExit(
            f"Unknown profile {name!r}. Choose one of: {', '.join(sorted(PROFILES))}")
    return dict(PROFILES[key])


def add_profile_args(ap):
    """Attach the shared profile/override flags to an ArgumentParser."""
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    choices=sorted(PROFILES),
                    help=f"sizing preset (default {DEFAULT_PROFILE})")
    ap.add_argument("--target-vol", type=float, default=None,
                    help="override the profile's annualized target vol %%")
    ap.add_argument("--min-size", type=float, default=None,
                    help="override the profile's size floor")
    ap.add_argument("--max-leverage", type=float, default=None,
                    help="override the profile's size cap")
    ap.add_argument("--response-k", type=float, default=None,
                    help="convexity exponent; 1.0 = linear, >1 = amplified")
    ap.add_argument("--periods-per-year", type=int, default=None,
                    help="252 equities, 365 crypto")
    ap.add_argument("--min-train", type=int, default=None,
                    help="days of history before the first forecast")
    ap.add_argument("--refit-every", type=int, default=None,
                    help="days between GARCH refits")
    ap.add_argument("--regime-lookback", type=int, default=None,
                    help="window for the vol percentile")
    return ap


def resolve(args) -> dict:
    """Profile defaults, with any explicit CLI flag taking precedence."""
    cfg = get_profile(getattr(args, "profile", DEFAULT_PROFILE))
    for field in ("target_vol", "min_size", "max_leverage", "response_k",
                  "periods_per_year", "min_train", "refit_every",
                  "regime_lookback"):
        override = getattr(args, field, None)
        if override is not None:
            cfg[field] = override
    if cfg["min_size"] <= 0 or cfg["max_leverage"] <= cfg["min_size"]:
        raise SystemExit("Need 0 < min_size < max_leverage.")
    if cfg["response_k"] <= 0:
        raise SystemExit("response_k must be positive.")
    return cfg
