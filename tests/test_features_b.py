import numpy as np
import pandas as pd

from src.features.branch_a import build_features_a
from src.features.branch_b import build_features_b


def make_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    hi = pd.concat([open_, close], axis=1).max(axis=1) * (1 + rng.uniform(0, 0.005, n))
    lo = pd.concat([open_, close], axis=1).min(axis=1) * (1 - rng.uniform(0, 0.005, n))
    vol = pd.Series(rng.uniform(10, 100, n), index=idx)
    trades = (vol * 5).round()
    taker = vol * rng.uniform(0.3, 0.7, n)
    for s in (vol, trades, taker):          # simulate a gap of empty bars
        s.iloc[300:303] = 0
    return pd.DataFrame({
        "open": open_, "high": hi, "low": lo, "close": close, "volume": vol,
        "quote_volume": vol * close, "trades": trades, "taker_buy_base": taker,
    })


CUT = 1100


def test_b_has_no_lookahead_truncation():
    df, oth = make_frame(seed=0), make_frame(seed=1)
    full = build_features_b(df, oth)
    part = build_features_b(df.iloc[:CUT], oth.iloc[:CUT])
    pd.testing.assert_frame_equal(full.iloc[:CUT], part, check_freq=False)


def test_wrecking_future_of_both_coins_leaves_past_unchanged():
    df, oth = make_frame(seed=0), make_frame(seed=1)
    df2, oth2 = df.copy(), oth.copy()
    df2.iloc[CUT:, :] *= 3
    oth2.iloc[CUT:, :] *= 3
    a, b = build_features_b(df, oth), build_features_b(df2, oth2)
    pd.testing.assert_frame_equal(a.iloc[:CUT], b.iloc[:CUT], check_freq=False)


def test_other_coin_future_does_not_leak_into_past():
    df, oth = make_frame(seed=0), make_frame(seed=1)
    oth2 = oth.copy()
    oth2.iloc[CUT:, :] *= 3
    a, b = build_features_b(df, oth), build_features_b(df, oth2)
    pd.testing.assert_frame_equal(a.iloc[:CUT], b.iloc[:CUT], check_freq=False)


def test_b_contains_all_of_a():
    df, oth = make_frame(seed=0), make_frame(seed=1)
    assert set(build_features_a(df).columns) <= set(build_features_b(df, oth).columns)