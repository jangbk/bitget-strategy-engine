"""
Executor -- Bitget Futures order execution via ccxt.

Handles:
  - Market entries with optional SL/TP (via ccxt stopLoss/takeProfit params)
  - Idempotency via signal_id deduplication
  - API failure counting -> 3 consecutive failures -> KillSwitch
  - Position queries, balance, leverage, cancel orders
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import ccxt

from bot.config import create_exchange

if TYPE_CHECKING:
    from bot.config import Config
    from bot.data.store import DataStore
    from bot.execution.state_machine import OrderStateMachine
    from bot.execution.kill_switch import KillSwitch
    from bot.strategies._base import Signal

logger = logging.getLogger(__name__)

# Maximum consecutive API failures before triggering kill switch
MAX_API_FAILURES = 3


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert flat symbol (e.g. BTCUSDT) to ccxt swap format (BTC/USDT:USDT)."""
    if "/" in symbol:
        return symbol  # already ccxt format
    # Strip trailing "USDT" and rebuild
    if symbol.endswith("USDT"):
        base = symbol[: -len("USDT")]
        return f"{base}/USDT:USDT"
    return symbol


class Executor:
    """
    Executes orders on Bitget Futures via ccxt.

    All ccxt calls are wrapped with asyncio.to_thread() for async compatibility.
    Signal IDs are deduplicated to prevent double-entry.
    """

    def __init__(
        self,
        config: "Config",
        store: "DataStore",
        state_machine: "OrderStateMachine",
        kill_switch: "KillSwitch",
    ) -> None:
        self._config = config
        self._store = store
        self._sm = state_machine
        self._kill_switch = kill_switch

        self._exchange: Optional[ccxt.bitget] = None

        # Failure counter -- 3 consecutive failures -> kill switch
        self._api_failure_count: int = 0

        # Idempotency set -- signal_ids already submitted this session
        self._submitted_signals: Set[str] = set()

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    async def start(self) -> None:
        """Create the ccxt exchange instance and load markets."""
        self._exchange = create_exchange(self._config)
        await asyncio.to_thread(self._exchange.load_markets)
        logger.info(
            "[Executor] Started. Bitget demo=%s, markets loaded=%d",
            self._config.bitget_demo,
            len(self._exchange.markets),
        )

    async def stop(self) -> None:
        """Cleanup."""
        self._exchange = None
        logger.info("[Executor] Stopped.")

    # ---------------------------------------------------------------------- #
    # Main: submit order from signal
    # ---------------------------------------------------------------------- #

    async def submit_order(
        self,
        signal: "Signal",
        qty: Optional[float] = None,
    ) -> dict:
        """
        Submit a market entry order to Bitget Futures.

        Steps:
          1. Check kill switch
          2. Check signal_id idempotency
          3. Submit MARKET order with optional SL/TP
          4. Record to DB immediately
          5. Register in state machine

        Returns the order result dict.
        """
        # Step 1: Kill switch check
        if self._kill_switch.is_active:
            logger.warning(
                "[Executor] KillSwitch ACTIVE -- blocking new entry for %s", signal.symbol
            )
            return {"error": "kill_switch_active", "signal_id": signal.id}

        # Step 2: Idempotency -- prevent double-entry from same signal
        if signal.id in self._submitted_signals:
            logger.warning(
                "[Executor] Signal %s already submitted -- skipping duplicate.", signal.id
            )
            return {"error": "duplicate_signal", "signal_id": signal.id}

        self._submitted_signals.add(signal.id)

        # Generate internal order ID
        internal_order_id = str(uuid.uuid4())

        # Register in state machine
        regime = self._store.get_regime() or {}
        self._sm.create(
            order_id=internal_order_id,
            signal_id=signal.id,
            strategy=signal.strategy,
            regime_snapshot=regime,
        )

        # Step 3: Determine side
        side = "buy" if signal.action == "BUY" else "sell"

        # Determine quantity
        if qty is None or qty <= 0:
            logger.error("[Executor] qty=0 for signal %s -- skipping order.", signal.id)
            self._sm.transition(internal_order_id, "REJECTED", reason="qty=0")
            return {"error": "qty_zero", "signal_id": signal.id}

        ccxt_symbol = _to_ccxt_symbol(signal.symbol)

        # Transition: SIGNAL_CREATED -> ORDER_SUBMITTED
        self._sm.transition(
            internal_order_id, "ORDER_SUBMITTED",
            reason=f"Submitting MARKET {side} {qty} {ccxt_symbol}",
        )

        logger.info(
            "[Executor] Submitting %s MARKET %s qty=%.6f signal_id=%s",
            side, ccxt_symbol, qty, signal.id,
        )

        # Build ccxt params for SL/TP
        params: Dict = {}
        if signal.sl is not None:
            params["stopLoss"] = {"triggerPrice": signal.sl, "type": "market"}
        if signal.tp is not None:
            params["takeProfit"] = {"triggerPrice": signal.tp, "type": "market"}

        # Step 4: Call Bitget via ccxt
        try:
            result = await asyncio.to_thread(
                self._exchange.create_order,
                ccxt_symbol, "market", side, qty, None, params,
            )
        except ccxt.InsufficientFunds as exc:
            logger.error("[Executor] Insufficient funds: %s", exc)
            self._handle_api_failure()
            self._sm.transition(internal_order_id, "REJECTED", reason=f"InsufficientFunds: {exc}")
            self._persist_failed_order(internal_order_id, signal, side.upper(), qty, str(exc))
            return {"error": str(exc), "signal_id": signal.id}
        except ccxt.NetworkError as exc:
            logger.error("[Executor] Network error: %s", exc)
            self._handle_api_failure()
            self._sm.transition(internal_order_id, "REJECTED", reason=f"NetworkError: {exc}")
            self._persist_failed_order(internal_order_id, signal, side.upper(), qty, str(exc))
            return {"error": str(exc), "signal_id": signal.id}
        except ccxt.ExchangeNotAvailable as exc:
            logger.error("[Executor] Exchange not available: %s", exc)
            self._handle_api_failure()
            self._sm.transition(internal_order_id, "REJECTED", reason=f"ExchangeNotAvailable: {exc}")
            self._persist_failed_order(internal_order_id, signal, side.upper(), qty, str(exc))
            return {"error": str(exc), "signal_id": signal.id}
        except Exception as exc:
            logger.error("[Executor] API error submitting order: %s", exc)
            self._handle_api_failure()
            self._sm.transition(internal_order_id, "REJECTED", reason=str(exc))
            self._persist_failed_order(internal_order_id, signal, side.upper(), qty, str(exc))
            return {"error": str(exc), "signal_id": signal.id}

        # API success -- reset failure counter
        self._api_failure_count = 0

        # Parse ccxt result
        exchange_order_id = str(result.get("id", ""))
        filled = float(result.get("filled", 0) or 0)
        amount = float(result.get("amount", qty) or qty)
        avg_price = float(result.get("average", 0) or result.get("price", 0) or 0)
        status_str = result.get("status", "open")  # ccxt: open, closed, canceled

        is_partial = filled > 0 and filled < amount
        is_filled = filled >= amount or status_str == "closed"

        if is_partial:
            sm_status = "PARTIALLY_FILLED"
        elif is_filled:
            sm_status = "FILLED"
        else:
            sm_status = "REJECTED"

        self._sm.transition(
            internal_order_id, sm_status,
            reason=f"Bitget status={status_str} filled={filled}",
            execution_result=result,
        )

        # Persist to DB immediately
        fee_cost = 0.0
        if result.get("fee") and result["fee"].get("cost"):
            fee_cost = float(result["fee"]["cost"])

        order_record = {
            "id": internal_order_id,
            "binance_order_id": exchange_order_id,  # keep field name for DB compat
            "signal_id": signal.id,
            "ts": int(time.time() * 1000),
            "symbol": signal.symbol,
            "side": side.upper(),
            "type": "MARKET",
            "qty": amount,
            "price": avg_price,
            "status": sm_status,
            "filled_qty": filled,
            "filled_price": avg_price,
            "fee": fee_cost,
            "strategy": signal.strategy,
            "regime": signal.regime,
            "partial_fill": is_partial,
        }
        self._store.save_order(order_record)

        # Broadcast to dashboard
        self._store._broadcast("order_update", order_record)

        return {
            "internal_order_id": internal_order_id,
            "exchange_order_id": exchange_order_id,
            "signal_id":         signal.id,
            "status":            sm_status,
            "filled_qty":        filled,
            "avg_price":         avg_price,
            "partial_fill":      is_partial,
        }

    # ---------------------------------------------------------------------- #
    # submit_market_order (alias with explicit sl/tp)
    # ---------------------------------------------------------------------- #

    async def submit_market_order(
        self,
        signal: "Signal",
        qty: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> dict:
        """
        Place market order with optional SL/TP.
        Delegates to submit_order after overriding signal SL/TP if provided.
        """
        # Override signal's SL/TP if explicitly provided
        if sl is not None:
            signal.sl = sl
        if tp is not None:
            signal.tp = tp
        return await self.submit_order(signal, qty)

    # ---------------------------------------------------------------------- #
    # Close position (reduce-only)
    # ---------------------------------------------------------------------- #

    async def close_position(self, symbol: str, side: str, qty: float) -> dict:
        """Opposite market order with reduceOnly to close a position."""
        ccxt_symbol = _to_ccxt_symbol(symbol)
        close_side = "sell" if side.upper() in ("BUY", "LONG") else "buy"
        params = {"reduceOnly": True}
        try:
            result = await asyncio.to_thread(
                self._exchange.create_order,
                ccxt_symbol, "market", close_side, qty, None, params,
            )
            self._api_failure_count = 0
            logger.info(
                "[Executor] Reduce-only close for %s side=%s qty=%.6f", symbol, side, qty
            )
            return result
        except Exception as exc:
            logger.error("[Executor] close_position error: %s", exc)
            self._handle_api_failure()
            return {"error": str(exc)}

    # Keep the old name as an alias for dashboard compatibility
    async def close_position_reduce_only(self, symbol: str, side: str, qty: float) -> dict:
        """Alias for close_position -- keeps dashboard/main.py compatibility."""
        return await self.close_position(symbol, side, qty)

    # ---------------------------------------------------------------------- #
    # Cancel orders
    # ---------------------------------------------------------------------- #

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> list:
        """Cancel all open orders. If symbol given, only that symbol."""
        cancelled = []
        if symbol:
            symbols = [symbol]
        else:
            # Cancel across all tracked symbols
            symbols = getattr(self._config, "tracked_symbols", [])
            if not symbols:
                # Fallback: try to get open orders without symbol filter
                try:
                    open_orders = await asyncio.to_thread(
                        self._exchange.fetch_open_orders,
                    )
                    symbols = list({o["symbol"] for o in open_orders if o.get("symbol")})
                except Exception:
                    symbols = []

        for sym in symbols:
            ccxt_sym = _to_ccxt_symbol(sym)
            try:
                result = await asyncio.to_thread(
                    self._exchange.cancel_all_orders, ccxt_sym,
                )
                self._api_failure_count = 0
                cancelled.append({"symbol": sym, "result": result})
                logger.info("[Executor] Cancelled all orders on %s", sym)
            except ccxt.OrderNotFound:
                logger.debug("[Executor] No open orders on %s", sym)
            except Exception as exc:
                logger.error("[Executor] Cancel all orders on %s failed: %s", sym, exc)
                self._handle_api_failure()

        return cancelled

    # ---------------------------------------------------------------------- #
    # Set leverage
    # ---------------------------------------------------------------------- #

    async def set_leverage(self, symbol: str, leverage: int = 3) -> dict:
        """Set leverage for a symbol on Bitget."""
        ccxt_symbol = _to_ccxt_symbol(symbol)
        try:
            result = await asyncio.to_thread(
                self._exchange.set_leverage, leverage, ccxt_symbol,
            )
            self._api_failure_count = 0
            logger.info("[Executor] Leverage set to %dx for %s", leverage, symbol)
            return result
        except Exception as exc:
            logger.error("[Executor] set_leverage error for %s: %s", symbol, exc)
            self._handle_api_failure()
            return {"error": str(exc)}

    # ---------------------------------------------------------------------- #
    # Position / order queries
    # ---------------------------------------------------------------------- #

    async def fetch_positions(self) -> List[dict]:
        """
        Fetch open positions. Returns list of dicts with:
        symbol, side, size, entry_price, unrealised_pnl.
        """
        try:
            raw = await asyncio.to_thread(self._exchange.fetch_positions)
            self._api_failure_count = 0
            positions = []
            for p in raw:
                size = float(p.get("contracts", 0) or 0)
                if size == 0:
                    continue
                positions.append({
                    "symbol": p.get("symbol", ""),
                    "side": p.get("side", ""),
                    "size": size,
                    "entry_price": float(p.get("entryPrice", 0) or 0),
                    "unrealised_pnl": float(p.get("unrealizedPnl", 0) or 0),
                })
            return positions
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "event loop" in str(exc):
                logger.debug("[Executor] fetch_positions skipped -- cross-loop call")
                return []
            logger.error("[Executor] fetch_positions error: %s", exc)
            self._handle_api_failure()
            return []

    async def get_open_positions(self) -> list:
        """
        Compatibility method for reconciler/dashboard.

        Returns positions in the old Binance-like format:
        [{"symbol": ..., "positionAmt": ..., "entryPrice": ..., "unRealizedProfit": ...}]
        """
        try:
            raw = await asyncio.to_thread(self._exchange.fetch_positions)
            self._api_failure_count = 0
            positions = []
            for p in raw:
                size = float(p.get("contracts", 0) or 0)
                if size == 0:
                    continue
                side = p.get("side", "long")
                pos_amt = size if side == "long" else -size
                positions.append({
                    "symbol": p.get("symbol", ""),
                    "positionAmt": pos_amt,
                    "entryPrice": float(p.get("entryPrice", 0) or 0),
                    "unRealizedProfit": float(p.get("unrealizedPnl", 0) or 0),
                })
            return positions
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "event loop" in str(exc):
                logger.debug("[Executor] get_open_positions skipped -- cross-loop call")
                return []
            logger.error("[Executor] get_open_positions error: %s", exc)
            self._handle_api_failure()
            return []

    async def get_open_orders(self) -> list:
        """Fetch all open orders."""
        try:
            result = await asyncio.to_thread(self._exchange.fetch_open_orders)
            self._api_failure_count = 0
            return result if isinstance(result, list) else []
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "event loop" in str(exc):
                logger.debug("[Executor] get_open_orders skipped -- cross-loop call")
                return []
            logger.error("[Executor] get_open_orders error: %s", exc)
            self._handle_api_failure()
            return []

    # ---------------------------------------------------------------------- #
    # Account info
    # ---------------------------------------------------------------------- #

    async def fetch_balance(self) -> float:
        """Return USDT total balance."""
        try:
            balance = await asyncio.to_thread(self._exchange.fetch_balance)
            self._api_failure_count = 0
            usdt = balance.get("USDT", {})
            return float(usdt.get("total", 0) or 0)
        except Exception as exc:
            logger.error("[Executor] fetch_balance error: %s", exc)
            self._handle_api_failure()
            return 0.0

    async def get_account_balance(self) -> float:
        """Compatibility alias for main.py -- returns USDT total balance."""
        try:
            result = await self.fetch_balance()
            if result > 0:
                return result
        except Exception as exc:
            logger.error("[Executor] get_account_balance error: %s", exc)
        return self._store.get_account_balance()

    # ---------------------------------------------------------------------- #
    # API failure handling
    # ---------------------------------------------------------------------- #

    def _handle_api_failure(self) -> None:
        """Increment failure counter; trigger kill switch at threshold."""
        self._api_failure_count += 1
        logger.warning(
            "[Executor] API failure count: %d/%d",
            self._api_failure_count, MAX_API_FAILURES,
        )
        if self._api_failure_count >= MAX_API_FAILURES:
            logger.critical(
                "[Executor] %d consecutive API failures -- scheduling kill switch.",
                self._api_failure_count,
            )
            asyncio.create_task(
                self._kill_switch.trigger(
                    reason=f"{self._api_failure_count} consecutive API failures",
                    triggered_by="executor",
                )
            )

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _persist_failed_order(
        self,
        internal_order_id: str,
        signal: "Signal",
        side: str,
        qty: float,
        error: str,
    ) -> None:
        """Save a failed order record to DB."""
        self._store.save_order({
            "id": internal_order_id,
            "signal_id": signal.id,
            "ts": int(time.time() * 1000),
            "symbol": signal.symbol,
            "side": side,
            "type": "MARKET",
            "qty": qty,
            "status": "FAILED",
            "strategy": signal.strategy,
            "error": error,
        })
