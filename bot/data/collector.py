"""
BitgetCollector — fetches candles, tickers, funding rates, open interest via ccxt (Bitget).

All ccxt calls are synchronous and wrapped with asyncio.to_thread() to keep
the async interface compatible with the rest of the engine.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from bot.config import Config, create_exchange
from bot.data.store import DataStore

logger = logging.getLogger(__name__)


class BitgetCollector:
    """Collects market data from Bitget Futures (via ccxt) and writes it to DataStore."""

    def __init__(self, config: Config, store: DataStore) -> None:
        self._config = config
        self._store = store
        self._running = False
        self._exchange = None
        self._tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------ #
    # Symbol conversion
    # ------------------------------------------------------------------ #

    def _to_ccxt_symbol(self, symbol: str) -> str:
        """Convert e.g. 'BTCUSDT' → 'BTC/USDT:USDT' for Bitget swap."""
        base = symbol.replace("USDT", "")
        return f"{base}/USDT:USDT"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        logger.info("BitgetCollector starting …")

        # Create exchange and load markets
        self._exchange = create_exchange(self._config)
        await asyncio.to_thread(self._exchange.load_markets)
        logger.info("Bitget markets loaded (%d symbols).", len(self._exchange.markets))

        # Fetch historical candles for all symbols/intervals on startup
        await self._fetch_all_history()

        # Mark exchange as reachable
        self._store.set_exchange_status(True)

        # Launch background polling tasks
        self._tasks = [
            asyncio.create_task(self._rest_poller(), name="rest_poller"),
            asyncio.create_task(self._ticker_poller(), name="ticker_poller"),
        ]
        logger.info("BitgetCollector started.")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("BitgetCollector stopped.")

    # ------------------------------------------------------------------ #
    # Historical REST fetch
    # ------------------------------------------------------------------ #

    async def _fetch_all_history(self) -> None:
        """Fetch candle history for every tracked symbol x interval."""
        coros = [
            self._fetch_candles(symbol, interval)
            for symbol in self._config.tracked_symbols
            for interval in self._config.candle_intervals
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error("History fetch error: %s", res)

    async def _fetch_candles(self, symbol: str, interval: str) -> None:
        """Fetch historical OHLCV candles and write to store."""
        ccxt_sym = self._to_ccxt_symbol(symbol)
        try:
            raw = await asyncio.to_thread(
                self._exchange.fetch_ohlcv,
                ccxt_sym,
                interval,
                None,  # since
                self._config.candle_limit,
            )
        except Exception as exc:
            logger.error("Candle fetch failed %s/%s: %s", symbol, interval, exc)
            return

        for row in raw:
            candle = {
                "ts": int(row[0]),
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
                "v": float(row[5]),
            }
            await self._store.upsert_candle(symbol, interval, candle)

        logger.info("Fetched %d %s/%s candles from Bitget", len(raw), symbol, interval)

    # ------------------------------------------------------------------ #
    # Ticker fetch
    # ------------------------------------------------------------------ #

    async def _fetch_tickers(self) -> None:
        """Fetch latest ticker for each tracked symbol."""
        for symbol in self._config.tracked_symbols:
            ccxt_sym = self._to_ccxt_symbol(symbol)
            try:
                data = await asyncio.to_thread(self._exchange.fetch_ticker, ccxt_sym)
                ticker = {
                    "symbol": symbol,
                    "ts": int(data.get("timestamp") or time.time() * 1000),
                    "price": float(data.get("last") or 0),
                    "volume_24h": float(data.get("quoteVolume") or 0),
                    "change_pct": float(data.get("percentage") or 0),
                }
                await self._store.update_ticker(symbol, ticker)
            except Exception as exc:
                logger.warning("Ticker fetch %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    # Funding rate fetch
    # ------------------------------------------------------------------ #

    async def _fetch_funding_rates(self) -> None:
        """Fetch current funding rate for each tracked symbol."""
        for symbol in self._config.tracked_symbols:
            ccxt_sym = self._to_ccxt_symbol(symbol)
            try:
                data = await asyncio.to_thread(self._exchange.fetch_funding_rate, ccxt_sym)
                rate = float(data.get("fundingRate") or 0)
                await self._store.update_funding(symbol, rate)
            except Exception as exc:
                logger.warning("Funding rate fetch %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    # Open interest fetch
    # ------------------------------------------------------------------ #

    async def _fetch_open_interest(self) -> None:
        """Fetch open interest for each tracked symbol."""
        for symbol in self._config.tracked_symbols:
            ccxt_sym = self._to_ccxt_symbol(symbol)
            try:
                data = await asyncio.to_thread(self._exchange.fetch_open_interest, ccxt_sym)
                oi = float(data.get("openInterestAmount") or data.get("openInterest") or 0)
                await self._store.update_open_interest(symbol, oi)
            except Exception as exc:
                logger.warning("OI fetch %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    # Background pollers
    # ------------------------------------------------------------------ #

    async def _rest_poller(self) -> None:
        """Periodically poll candles, funding rates, and open interest (every 60s)."""
        while self._running:
            try:
                await self._fetch_all_history()
                await self._fetch_funding_rates()
                await self._fetch_open_interest()
            except Exception as exc:
                logger.error("REST poll error: %s", exc)
            await asyncio.sleep(60)

    async def _ticker_poller(self) -> None:
        """Periodically poll tickers at configured interval."""
        while self._running:
            try:
                await self._fetch_tickers()
            except Exception as exc:
                logger.error("Ticker poll error: %s", exc)
            await asyncio.sleep(self._config.ticker_update_interval_sec)

    # ------------------------------------------------------------------ #
    # Public methods for Reconciler / other components
    # ------------------------------------------------------------------ #

    async def fetch_positions(self) -> List[dict]:
        """Fetch open positions from Bitget. Returns list of position dicts."""
        positions = []
        try:
            raw = await asyncio.to_thread(self._exchange.fetch_positions)
            for pos in raw:
                # Only include positions with non-zero size
                contracts = float(pos.get("contracts") or 0)
                if contracts == 0:
                    continue
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "side": pos.get("side", ""),
                    "contracts": contracts,
                    "entryPrice": float(pos.get("entryPrice") or 0),
                    "unrealizedPnl": float(pos.get("unrealizedPnl") or 0),
                    "leverage": float(pos.get("leverage") or 1),
                    "notional": float(pos.get("notional") or 0),
                })
        except Exception as exc:
            logger.warning("fetch_positions error: %s", exc)
        return positions

    async def fetch_balance(self) -> float:
        """Fetch USDT balance from Bitget. Returns float."""
        try:
            balance = await asyncio.to_thread(self._exchange.fetch_balance)
            usdt = balance.get("USDT", {})
            return float(usdt.get("total") or 0)
        except Exception as exc:
            logger.warning("fetch_balance error: %s", exc)
            return 0.0


# Keep backward-compatible alias so existing imports still work
BinanceCollector = BitgetCollector
