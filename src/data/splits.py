"""Dataset loader + chronological splits with purging.

Usage:  python -m src.data.splits     (prints a summary for both coins, A and B)
"""
import pandas as pd
import yaml

from src.features.labels import forward_return, triple_barrier

BASE = ["open", "high", "low", "close", "volume", "is_filled", "is_zero_vol"]
LABELS = ["fwd_ret", "tb_label", "tb_ret", "tb_hold"]


def load_cfg(path: str = "configs/base.yaml") -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def load_dataset(symbol: str, branch: str = "B", cfg_path: str = "configs/base.yaml"):
    """Return (frame with features + labels, list of feature column names).

    Both branches are trimmed to Branch B's first row so A and B compare fairly.
    """
    cfg = load_cfg(cfg_path)
    lab = cfg["labels"]
    feats = pd.read_parquet(f"data/processed/{symbol}_1h_{branch}.parquet")
    start = pd.read_parquet(f"data/processed/{symbol}_1h_B.parquet").index.min()
    feats = feats.loc[start:]

    clean = pd.read_parquet(f"data/interim/{symbol}_1h_clean.parquet")
    labels = triple_barrier(clean, lab["horizon"], lab["k_up"], lab["k_dn"], lab["atr_n"])
    labels["fwd_ret"] = forward_return(clean, lab["horizon"])

    df = feats.join(labels)
    feature_cols = [c for c in feats.columns if c not in BASE]
    return df, feature_cols


def get_split(df: pd.DataFrame, cfg: dict, name: str, purge: int) -> pd.DataFrame:
    """Rows of one split ('train' | 'validation' | 'test').

    train/validation: drop the last `purge` bars (their labels reach into the next
    period) and rows with no label. test: untouched, since we only backtest on it.
    """
    lo, hi = cfg["splits"][name]
    lo = pd.Timestamp(lo, tz="UTC")
    hi = pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)   # end date inclusive
    part = df[(df.index >= lo) & (df.index < hi)]
    if name != "test":
        part = part[part.index < hi - pd.Timedelta(hours=purge)]
        part = part.dropna(subset=["tb_label"])
    return part


def main() -> None:
    cfg = load_cfg()
    purge = cfg["labels"]["horizon"] + 1
    for sym in cfg["data"]["symbols"]:
        for branch in ("A", "B"):
            df, feats = load_dataset(sym, branch)
            print(f"\n=== {sym} branch {branch}: {len(feats)} features, {len(df)} rows ===")
            for name in ("train", "validation", "test"):
                p = get_split(df, cfg, name, purge)
                line = f"{name:<10} rows={len(p):>6}  {p.index.min():%Y-%m-%d} -> {p.index.max():%Y-%m-%d %H:%M}"
                if name != "test":
                    d = p["tb_label"].value_counts(normalize=True).sort_index().round(3).to_dict()
                    line += f"  labels={d}"
                print(line)


if __name__ == "__main__":
    main()