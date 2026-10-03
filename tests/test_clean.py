import pandas as pd
import pytest

from src.data.clean import clean


def make_df():
    idx = pd.date_range("2021-01-01", periods=6, freq="1h", tz="UTC")
    idx = idx.delete(3)  # remove one bar -> gap at 03:00
    n = len(idx)
    return pd.DataFrame(
        {"open": range(1, n + 1), "high": range(2, n + 2), "low": range(0, n),
         "close": range(1, n + 1), "volume": [5.0] * n, "close_time": idx,
         "quote_volume": [1.0] * n, "trades": [1] * n,
         "taker_buy_base": [1.0] * n, "taker_buy_quote": [1.0] * n},
        index=idx.rename("timestamp"),
    )


def test_gap_is_filled_and_flagged():
    out = clean(make_df(), "1h")
    assert len(out) == 6
    row = out.loc["2021-01-01 03:00"]
    assert row["is_filled"] and row["volume"] == 0
    assert row["open"] == row["high"] == row["low"] == row["close"]


def test_no_nans_and_regular_index():
    out = clean(make_df(), "1h")
    assert not out.isna().any().any()
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_non_utc_rejected():
    df = make_df()
    df.index = df.index.tz_convert("Asia/Kolkata")
    with pytest.raises(AssertionError):
        clean(df, "1h")