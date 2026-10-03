import numpy as np
import pandas as pd

from src.features.branch_a import build_features_a


def make_ohlcv(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + rng.uniform(0, 0.005, n))
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - rng.uniform(0, 0.005, n))
    vol = pd.Series(rng.uniform(10, 100, n), index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_branch_a_has_no_lookahead():
    df = make_ohlcv()
    cut = 400
    full = build_features_a(df)
    part = build_features_a(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_freq=False)


def test_future_changes_do_not_alter_past_features():
    df = make_ohlcv()
    altered = df.copy()
    altered.iloc[500:, :] = altered.iloc[500:, :] * 3   # wreck the future
    a, b = build_features_a(df), build_features_a(altered)
    pd.testing.assert_frame_equal(a.iloc[:500], b.iloc[:500], check_freq=False)