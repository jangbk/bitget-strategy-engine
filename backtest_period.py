"""
22B Strategy Engine — Period Backtest Runner

Usage:
    python backtest_period.py --start 2025-01-01 --end 2025-08-31 --label "Period1"
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

from bot.config import get_config
from bot.data.store import DataStore
from bot.data.validation_dataset_loader import ValidationDatasetLoader
from bot.data.validation_replay import ValidationReplaySession
from bot.data.replay_account import ReplayAccount
from bot.regime.detector import RegimeDetector
from bot.strategies.manager import StrategyManager
from bot.strategies.params_store import StrategyParamsStore
from db.schema import init_db

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)


def fetch_ohlcv(symbol: str, interval: str, start_dt: datetime, end_dt: datetime):
    """Fetch OHLCV from Bitget public API (no auth)."""
    exchange = ccxt.bitget({"options": {"defaultType": "swap"}})
    exchange.load_markets()

    base = symbol.replace("USDT", "")
    ccxt_symbol = f"{base}/USDT:USDT"

    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_candles = []
    prev_since = -1
    while since < end_ms:
        candles = exchange.fetch_ohlcv(ccxt_symbol, interval, since=since, limit=200)
        if not candles:
            break
        # Prevent infinite loop
        if candles[-1][0] <= prev_since:
            break
        prev_since = candles[-1][0]
        # Filter out candles beyond end date
        filtered = [c for c in candles if c[0] <= end_ms]
        all_candles.extend(filtered)
        since = candles[-1][0] + 1
        sys.stdout.write(f"\r  {symbol} {interval}: {len(all_candles)} candles...")
        sys.stdout.flush()
        if len(filtered) < len(candles):
            break  # We've passed end date
        time.sleep(0.15)  # Rate limit

    print(f"\r  {symbol} {interval}: {len(all_candles)} candles")
    return all_candles


def save_datasets(symbols, intervals, start_dt, end_dt, output_dir: Path):
    """Fetch and save all datasets as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        sym_dir = output_dir / symbol
        sym_dir.mkdir(exist_ok=True)
        for interval in intervals:
            print(f"Fetching {symbol} {interval}...")
            candles = fetch_ohlcv(symbol, interval, start_dt, end_dt)

            bars = []
            for c in candles:
                dt = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
                bars.append({
                    "open_time": dt.isoformat(),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })

            payload = {"symbol": symbol, "interval": interval, "bars": bars}
            out_file = sym_dir / f"{interval}.json"
            with open(out_file, "w") as f:
                json.dump(payload, f)

    return len(symbols) * len(intervals)


async def run_backtest(data_dir: Path, db_path: str, label: str):
    """Run replay backtest on prepared datasets."""
    conn = init_db(db_path)
    store = DataStore(conn)
    store.set_system_mode("OBSERVE")

    loader = ValidationDatasetLoader(store, str(data_dir))
    summary = await loader.load(warmup_bars=52)

    if summary.replay_bars_remaining == 0:
        logger.error("[%s] No replay bars after warmup!", label)
        return None

    replay = ValidationReplaySession(
        store=store,
        datasets=loader.get_replay_datasets(),
        warmup_bars=52,
        step_delay_ms=0,
        max_steps=0,
    )

    account = ReplayAccount(
        initial_balance=10_000.0,
        position_size_pct=0.10,
        fee_rate=0.0004,
        slippage_pct=0.0005,
    )

    StrategyParamsStore.get_instance()
    detector = RegimeDetector(store)
    manager = StrategyManager(store)
    manager.initialize()
    manager.recorder._replay_account = account

    total = replay.total_steps()
    logger.info("[%s] Running %d steps (%d candles loaded)...", label, total, summary.candles_loaded)

    step = 0
    signal_count = 0
    regime_counts = {}
    trade_results = {"wins": 0, "losses": 0, "total_pnl": 0.0}
    peak_balance = account.initial_balance
    max_drawdown = 0.0

    while True:
        bar = await replay.next_bar()
        if bar is None:
            break
        step += 1

        regime_result = detector.detect()
        if regime_result:
            r = regime_result.get("regime", "UNKNOWN")
            regime_counts[r] = regime_counts.get(r, 0) + 1
            signals = manager.run_all(regime_result)
            for sig in signals:
                if sig.action != "SKIP":
                    signal_count += 1

        # Track drawdown
        if account.balance > peak_balance:
            peak_balance = account.balance
        dd = (peak_balance - account.balance) / peak_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd

        if step % 1000 == 0:
            pct = step / total * 100
            logger.info("[%s] Step %d/%d (%.0f%%) — Balance: $%.2f", label, step, total, pct, account.balance)

    # Gather trade stats from DB
    cursor = conn.cursor()
    closed = cursor.execute(
        "SELECT pnl_pct, close_reason FROM paper_positions WHERE status='CLOSED'"
    ).fetchall()

    wins = sum(1 for r in closed if (r[0] or 0) > 0)
    losses = sum(1 for r in closed if (r[0] or 0) <= 0)
    total_trades = len(closed)
    avg_win = 0.0
    avg_loss = 0.0
    if wins > 0:
        avg_win = sum(r[0] for r in closed if (r[0] or 0) > 0) / wins
    if losses > 0:
        avg_loss = abs(sum(r[0] for r in closed if (r[0] or 0) <= 0) / losses)

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss) if total_trades > 0 else 0

    # Strategy breakdown
    strat_stats = cursor.execute(
        "SELECT strategy, COUNT(*), "
        "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END), "
        "AVG(pnl_pct) "
        "FROM paper_positions WHERE status='CLOSED' GROUP BY strategy"
    ).fetchall()

    # Close reason breakdown
    reason_stats = cursor.execute(
        "SELECT close_reason, COUNT(*) FROM paper_positions WHERE status='CLOSED' GROUP BY close_reason"
    ).fetchall()

    pnl_pct = (account.balance / account.initial_balance - 1) * 100

    results = {
        "label": label,
        "steps": step,
        "candles": summary.candles_loaded,
        "signals": signal_count,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "start_balance": account.initial_balance,
        "end_balance": account.balance,
        "pnl_pct": pnl_pct,
        "max_drawdown": max_drawdown,
        "regime_distribution": regime_counts,
        "strategy_stats": strat_stats,
        "reason_stats": reason_stats,
    }

    return results


def print_results(r):
    """Pretty-print backtest results."""
    if r is None:
        return

    print(f"\n{'='*60}")
    print(f"  {r['label']}")
    print(f"{'='*60}")
    print(f"  Steps:          {r['steps']:,}")
    print(f"  Candles:        {r['candles']:,}")
    print(f"  Signals:        {r['signals']:,}")
    print(f"  Trades:         {r['total_trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"  Win Rate:       {r['win_rate']:.1f}%")
    print(f"  Avg Win:        +{r['avg_win']:.2f}%")
    print(f"  Avg Loss:       -{r['avg_loss']:.2f}%")
    print(f"  EXPECTANCY:     {r['expectancy']:+.3f}% per trade")
    print(f"  ---")
    print(f"  Start Balance:  ${r['start_balance']:,.2f}")
    print(f"  End Balance:    ${r['end_balance']:,.2f}")
    print(f"  P&L:            {r['pnl_pct']:+.2f}%")
    print(f"  Max Drawdown:   {r['max_drawdown']:.2f}%")

    print(f"\n  Regime Distribution:")
    total_steps = r["steps"]
    for regime, cnt in sorted(r["regime_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / total_steps * 100
        print(f"    {regime:20s} {cnt:5d} ({pct:.1f}%)")

    print(f"\n  Strategy Breakdown:")
    for row in r["strategy_stats"]:
        name, count, wins, avg_pnl = row
        wr = (wins or 0) / count * 100 if count > 0 else 0
        print(f"    {name:35s} {count:3d} trades  WR={wr:.0f}%  avgPnL={avg_pnl or 0:+.2f}%")

    print(f"\n  Close Reasons:")
    for row in r["reason_stats"]:
        reason, count = row
        print(f"    {reason or 'unknown':20s} {count}")

    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--label", default="Backtest", help="Label for this run")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT")
    parser.add_argument("--intervals", default="1h,4h")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch if already downloaded")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    symbols = [s.strip() for s in args.symbols.split(",")]
    intervals = [i.strip() for i in args.intervals.split(",")]
    label = args.label

    safe_label = label.replace(" ", "_").replace("/", "-")
    data_dir = Path(f"data/backtest/{safe_label}")
    db_path = f"data/backtest/{safe_label}.db"

    # 1. Fetch data
    if not args.skip_fetch:
        logger.info("[%s] Fetching data: %s to %s", label, args.start, args.end)
        save_datasets(symbols, intervals, start_dt, end_dt, data_dir)
    else:
        logger.info("[%s] Skipping fetch — using existing data", label)

    # 2. Run backtest
    results = await run_backtest(data_dir, db_path, label)
    print_results(results)

    return results


if __name__ == "__main__":
    asyncio.run(main())
