"""Download Binance spot klines into data/raw (raw, unmodified OHLCV).

Primary source : data.binance.vision monthly zip files (fast, clean, no API limits)
Fallback source: REST API (--source api), for when bulk files are unreachable

Usage:
    python -m src.data.acquire                      # uses configs/base.yaml
    python -m src.data.acquire --intervals 1h 1d    # also pull daily
    python -m src.data.acquire --source api
"""
import argparse
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

BULK_URL = "https://data.binance.vision/data/spot/monthly/klines"
API_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
NUM_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]


def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_ms(s: pd.Series) -> pd.Series:
    """Normalize timestamps to milliseconds (2025+ spot files use microseconds)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s < 1e14, s // 1000)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shared formatting for both sources. No value edits, only typing/indexing."""
    df = df.copy()
    df["open_time"] = _to_ms(df["open_time"])
    df["close_time"] = _to_ms(df["close_time"])
    df = df.dropna(subset=["open_time"])          # drops stray header rows
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    return df.drop(columns=["open_time", "ignore"]).set_index("timestamp")


# ----------------------------- bulk (primary) ------------------------------
def fetch_month(symbol: str, interval: str, period: pd.Period, cache: Path):
    name = f"{symbol}-{interval}-{period.year}-{period.month:02d}"
    zpath = cache / f"{name}.zip"
    if not zpath.exists():
        r = requests.get(f"{BULK_URL}/{symbol}/{interval}/{name}.zip", timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, header=None, names=COLS)
    return _clean_frame(df)


def download_bulk(symbol, interval, start, end, raw_dir: Path) -> pd.DataFrame:
    cache = raw_dir / "zips"
    cache.mkdir(parents=True, exist_ok=True)
    periods = pd.period_range(start, end, freq="M")
    frames = []
    for p in tqdm(periods, desc=f"{symbol} {interval}"):
        df = fetch_month(symbol, interval, p, cache)
        if df is None:
            print(f"  ! missing month {p} for {symbol} {interval}")
            continue
        frames.append(df)
    return pd.concat(frames)


# ------------------------------ API (fallback) -----------------------------
def download_api(symbol, interval, start, end) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    rows, cursor = [], start_ms
    base = None
    for b in API_URLS:                              # pick the first reachable host
        try:
            if requests.get(f"{b}/api/v3/ping", timeout=10).ok:
                base = b
                break
        except requests.RequestException:
            continue
    if base is None:
        raise RuntimeError("No Binance API host reachable from this network.")
    with tqdm(desc=f"{symbol} {interval} (api)") as bar:
        while cursor < end_ms:
            r = requests.get(
                f"{base}/api/v3/klines",
                params=dict(symbol=symbol, interval=interval,
                            startTime=cursor, endTime=end_ms, limit=1000),
                timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1][0] + 1
            bar.update(len(batch))
            time.sleep(0.2)                          # be polite to rate limits
    return _clean_frame(pd.DataFrame(rows, columns=COLS))


# ---------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--intervals", nargs="+", default=None)
    ap.add_argument("--source", choices=["bulk", "api"], default="bulk")
    args = ap.parse_args()

    cfg = load_config(args.config)["data"]
    intervals = args.intervals or [cfg["interval"]]
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in cfg["symbols"]:
        for interval in intervals:
            if args.source == "bulk":
                df = download_bulk(symbol, interval, cfg["start"], cfg["end"], raw_dir)
            else:
                df = download_api(symbol, interval, cfg["start"], cfg["end"])

            df = df[~df.index.duplicated(keep="first")].sort_index()
            lo = pd.Timestamp(cfg["start"], tz="UTC")
            hi = pd.Timestamp(cfg["end"], tz="UTC") + pd.Timedelta(days=1)
            df = df[(df.index >= lo) & (df.index < hi)]

            out = raw_dir / f"{symbol}_{interval}.parquet"
            df.to_parquet(out)
            print(f"saved {out}  rows={len(df)}  {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()