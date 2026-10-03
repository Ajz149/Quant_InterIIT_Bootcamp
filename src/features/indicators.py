"""Hand-written indicators (no TA library). All are causal: value at t uses data <= t."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (used by RSI and ATR)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def log_return(close: pd.Series, n: int = 1) -> pd.Series:
    return np.log(close).diff(n)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of 1-bar log returns (not annualized)."""
    return log_return(close, 1).rolling(window).std()


def zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = wilder(gain, n) / wilder(loss, n)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + k * std, mid - k * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return pct_b, width

    

def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio: 1 = straight-line trend, 0 = pure chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Wilder ADX with +DI / -DI. Returns (adx, plus_di, minus_di)."""
    up, dn = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di