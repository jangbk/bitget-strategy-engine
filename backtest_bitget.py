"""
Backtest data fetcher for Bitget — fetches historical OHLCV via ccxt,
saves to validation_datasets/ for the 22B Engine replay mode.

Usage:
    python backtest_bitget.py [--days 90] [--symbols BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT]
"""

import asyncio
import sys
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd


async def fetch_historical(symbol: str, interval: str, days: int = 90):
    """Fetch historical OHLCV from Bitget public API (no auth needed)."""
    exchange = ccxt.bitget({"options": {"defaultType": "swap"}})
    exchange.load_markets()

    base = symbol.replace("USDT", "")
    ccxt_symbol = f"{base}/USDT:USDT"

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(ccxt_symbol, interval, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        sys.stdout.write(f"\r  {symbol} {interval}: {len(all_candles)} candles...")
        sys.stdout.flush()
        if len(candles) < 1000:
            break

    print(f"\n  {symbol} {interval}: {len(all_candles)} total")

    df = pd.DataFrame(all_candles, columns=["ts", "o", "h", "l", "c", "v"])
    return df


async def main():
    parser = argparse.ArgumentParser(description="Fetch Bitget historical data for backtesting")
    parser.add_argument("--days", type=int, default=90, help="Number of days (default: 90)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--intervals", type=str, default="1h,4h",
                        help="Comma-separated intervals")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    intervals = [i.strip() for i in args.intervals.split(",")]

    output_dir = Path("data/import_staging/validation_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(symbols) * len(intervals)
    done = 0

    for symbol in symbols:
        for interval in intervals:
            done += 1
            print(f"[{done}/{total}] Fetching {symbol} {interval} ({args.days}d)...")
            try:
                df = await fetch_historical(symbol, interval, args.days)
                filename = f"{symbol}_{interval}.csv"
                df.to_csv(output_dir / filename, index=False)
                print(f"  Saved: {filename} ({len(df)} rows)")
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\n{'='*50}")
    print(f"✅ All data saved to: {output_dir}/")
    print(f"\nTo run backtest, set in .env:")
    print(f"  VALIDATION_DATASET_ENABLED=true")
    print(f"  VALIDATION_REPLAY_ENABLED=true")
    print(f"  VALIDATION_REPLAY_WARMUP_BARS=52")
    print(f"Then run: python -m bot.main")


if __name__ == "__main__":
    asyncio.run(main())
