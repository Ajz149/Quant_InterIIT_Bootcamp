# Project: Inter IIT Quant PS (BTC/USDT, ETH/USDT)

## Hard constraints
- NEVER import or suggest backtrader, backtesting.py, vectorbt, zipline, bt, or any external backtesting/trading library. The backtester is custom-built in src/backtest/.
- Transaction cost + slippage = 0.15% per transaction, read from configs/base.yaml.
- No lookahead: signals from bar t close are filled at bar t+1 open. Features may only use data available at time t. Labels use future data ONLY in training, never as features.
- Data range: 2021-01-01 to 2025-12-31. 2025 is a locked test set; do not tune on it.
- All 15 required metrics must be produced (see src/backtest/metrics.py).

## Conventions
- Python 3.10+, type hints, docstrings, small pure functions.
- Config-driven: no hardcoded paths, fees or dates in code.
- Raw data in data/raw is immutable. Write cleaned data to data/interim or data/processed.
- Every strategy inherits from src/strategies/base.py and runs through the same engine.
- Write pytest tests for engine, fees and lookahead.
- Random seeds come from the config.

## Workflow
- Work on a feature branch, never directly on main.
- Small commits, message style: "feat: ...", "fix: ...", "exp: ...", "docs: ...".
