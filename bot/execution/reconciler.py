"""
Reconciler — Position state reconciliation every 5 minutes (Part 3.3).

Compares:
  - Bitget actual open positions (via ccxt)
  - Local DB open live positions (from orders / positions table)

Discrepancies handled:
  - In DB but not on exchange → mark CLOSED + warn
  - On exchange but not in DB → create ORPHANED record + urgent alert
  - Quantity mismatch → log + warn
  - On kill switch trigger if unexpected position found
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from bot.data.store import DataStore
    from bot.execution.executor import Executor
    from bot.execution.kill_switch import KillSwitch
    from bot.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# How often to reconcile (seconds)
RECONCILE_INTERVAL_SEC = 300  # 5 minutes


# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class ReconcileResult:
    """Result of one reconciliation run."""
    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    checked: int = 0
    matched: int = 0
    in_db_not_exchange: List[dict] = field(default_factory=list)
    in_exchange_not_db: List[dict] = field(default_factory=list)
    qty_mismatches: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        return bool(
            self.in_db_not_exchange
            or self.in_exchange_not_db
            or self.qty_mismatches
        )

    def to_dict(self) -> dict:
        return {
            "ts":                   self.ts,
            "checked":              self.checked,
            "matched":              self.matched,
            "in_db_not_exchange":   self.in_db_not_exchange,
            "in_exchange_not_db":   self.in_exchange_not_db,
            "qty_mismatches":       self.qty_mismatches,
            "errors":               self.errors,
            "has_discrepancies":    self.has_discrepancies,
        }


# --------------------------------------------------------------------------- #
# Reconciler
# --------------------------------------------------------------------------- #

class Reconciler:
    """
    Compares local DB positions against Bitget API every 5 minutes.

    Discrepancy handling:
      - DB-only positions    → mark CLOSED (exchange already closed them)
      - Exchange-only        → create ORPHANED record, send urgent alert,
                               trigger kill switch
      - Qty mismatch         → log warning, send Telegram alert
    """

    def __init__(
        self,
        store: "DataStore",
        executor: "Executor",
        kill_switch: "KillSwitch",
        telegram: Optional["TelegramNotifier"] = None,
    ) -> None:
        self._store     = store
        self._executor  = executor
        self._kill_switch = kill_switch
        self._telegram  = telegram

        self._last_result: Optional[ReconcileResult] = None
        self._run_count:   int = 0
        self._task:        Optional[asyncio.Task] = None

    # ---------------------------------------------------------------------- #
    # Symbol normalization
    # ---------------------------------------------------------------------- #

    def _normalize_symbol(self, raw_symbol: str) -> str:
        """Normalize ccxt symbol to internal format. 'BTC/USDT:USDT' → 'BTCUSDT'"""
        if "/" in raw_symbol:
            return raw_symbol.split("/")[0] + "USDT"
        return raw_symbol

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def start(self) -> None:
        """Start the background reconciliation loop."""
        self._task = asyncio.create_task(
            self.reconcile_loop(), name="reconciler"
        )
        logger.info("[Reconciler] Background loop started (interval=%ds).", RECONCILE_INTERVAL_SEC)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    # ---------------------------------------------------------------------- #
    # Main reconciliation loop
    # ---------------------------------------------------------------------- #

    async def reconcile_loop(self) -> None:
        """Run reconcile() every RECONCILE_INTERVAL_SEC seconds."""
        while True:
            try:
                await asyncio.sleep(RECONCILE_INTERVAL_SEC)
                result = await self.run()
                self._last_result = result
                self._run_count += 1

                if result.has_discrepancies:
                    logger.warning(
                        "[Reconciler] Discrepancies found: db_only=%d exchange_only=%d qty_mismatch=%d",
                        len(result.in_db_not_exchange),
                        len(result.in_exchange_not_db),
                        len(result.qty_mismatches),
                    )
                else:
                    logger.debug(
                        "[Reconciler] Run #%d — %d positions matched, no discrepancies.",
                        self._run_count, result.matched,
                    )

            except asyncio.CancelledError:
                logger.info("[Reconciler] Loop cancelled.")
                break
            except Exception as exc:
                logger.error("[Reconciler] Loop error: %s", exc)

    # ---------------------------------------------------------------------- #
    # Core reconciliation
    # ---------------------------------------------------------------------- #

    async def run(self) -> ReconcileResult:
        """
        Perform one reconciliation cycle.

        Returns ReconcileResult with full discrepancy details.
        """
        result = ReconcileResult()

        # --- Step 1: Get Binance actual open positions ---
        try:
            exchange_positions = await self._executor.get_open_positions()
        except Exception as exc:
            msg = f"Failed to fetch exchange positions: {exc}"
            logger.error("[Reconciler] %s", msg)
            result.errors.append(msg)
            return result

        # Build exchange position map: symbol → position dict
        # Normalize ccxt symbols (e.g. 'BTC/USDT:USDT' → 'BTCUSDT')
        exchange_map: Dict[str, dict] = {}
        for pos in exchange_positions:
            raw_sym = pos.get("symbol", "")
            if raw_sym:
                sym = self._normalize_symbol(raw_sym)
                exchange_map[sym] = pos

        # --- Step 2: Get local DB open live positions ---
        try:
            db_positions = self._store.get_open_live_positions()
        except Exception as exc:
            msg = f"Failed to fetch DB positions: {exc}"
            logger.error("[Reconciler] %s", msg)
            result.errors.append(msg)
            return result

        # Build DB position map: symbol → position dict
        db_map: Dict[str, dict] = {}
        for pos in db_positions:
            sym = pos.get("symbol", "")
            if sym:
                db_map[sym] = pos

        result.checked = len(db_map) + len(exchange_map)

        # --- Step 3: Compare ---

        # 3a: In DB but NOT on exchange → likely closed externally
        for sym, db_pos in db_map.items():
            if sym not in exchange_map:
                result.in_db_not_exchange.append({
                    "symbol": sym,
                    "db_position": db_pos,
                    "action": "marked_closed",
                })
                logger.warning(
                    "[Reconciler] Position %s is in DB (OPEN) but NOT on exchange — marking CLOSED.",
                    sym,
                )
                # Mark as closed in DB
                try:
                    pos_id = db_pos.get("id")
                    if pos_id:
                        self._store.update_order(
                            str(pos_id),
                            {
                                "status": "CLOSED",
                                "close_reason": "reconciler:not_on_exchange",
                                "closed_at": int(time.time() * 1000),
                            },
                        )
                except Exception as exc:
                    logger.error("[Reconciler] Failed to mark %s CLOSED: %s", sym, exc)
                    result.errors.append(str(exc))

                # Telegram warning
                if self._telegram:
                    self._telegram._enqueue(
                        f"*Reconciler Warning*\n"
                        f"Position {sym} in DB but NOT on exchange.\n"
                        f"Marked CLOSED in DB.\n"
                        f"Time: {_ts()}"
                    )

        # 3b: On exchange but NOT in DB → orphaned position (urgent!)
        for sym, ex_pos in exchange_map.items():
            if sym not in db_map:
                pos_amt = float(ex_pos.get("positionAmt", 0))
                entry_price = float(ex_pos.get("entryPrice", 0))
                unrealised_pnl = float(ex_pos.get("unRealizedProfit", 0))

                orphan_record = {
                    "symbol":          sym,
                    "exchange_qty":    pos_amt,
                    "entry_price":     entry_price,
                    "unrealised_pnl":  unrealised_pnl,
                    "action":          "orphaned",
                }
                result.in_exchange_not_db.append(orphan_record)

                logger.critical(
                    "[Reconciler] ORPHANED POSITION on exchange: %s qty=%.6f entry=%.4f pnl=%.4f",
                    sym, pos_amt, entry_price, unrealised_pnl,
                )

                # Save orphaned position to DB
                try:
                    self._store.save_order({
                        "id":           f"orphan-{sym}-{int(time.time())}",
                        "signal_id":    None,
                        "ts":           int(time.time() * 1000),
                        "symbol":       sym,
                        "side":         "BUY" if pos_amt > 0 else "SELL",
                        "type":         "UNKNOWN",
                        "qty":          abs(pos_amt),
                        "price":        entry_price,
                        "status":       "ORPHANED",
                        "filled_qty":   abs(pos_amt),
                        "filled_price": entry_price,
                        "fee":          0,
                        "strategy":     "ORPHANED",
                    })
                except Exception as exc:
                    logger.error("[Reconciler] Failed to save orphaned position: %s", exc)

                # Urgent Telegram alert
                if self._telegram:
                    self._telegram._enqueue(
                        f"*URGENT — Orphaned Position Detected*\n"
                        f"Symbol: `{sym}`\n"
                        f"Qty: `{pos_amt:.6f}`\n"
                        f"Entry: `{entry_price:.4f}`\n"
                        f"Unrealised PnL: `{unrealised_pnl:.4f}`\n"
                        f"NOT in local DB — manual review required!\n"
                        f"Time: {_ts()}"
                    )

                # Trigger kill switch for unexpected position
                await self._kill_switch.trigger(
                    reason=f"Orphaned position found: {sym} qty={pos_amt:.6f}",
                    triggered_by="reconciler",
                )

        # 3c: On both — check quantity mismatch
        for sym in set(db_map.keys()) & set(exchange_map.keys()):
            db_pos = db_map[sym]
            ex_pos = exchange_map[sym]

            db_qty = float(db_pos.get("qty", 0) or 0)
            ex_qty = abs(float(ex_pos.get("positionAmt", 0)))

            # Tolerance: 1% or 0.0001 units
            tolerance = max(0.0001, db_qty * 0.01)
            if abs(db_qty - ex_qty) > tolerance:
                mismatch = {
                    "symbol":   sym,
                    "db_qty":   db_qty,
                    "ex_qty":   ex_qty,
                    "diff":     ex_qty - db_qty,
                }
                result.qty_mismatches.append(mismatch)
                logger.warning(
                    "[Reconciler] QTY MISMATCH %s: DB=%.6f Exchange=%.6f diff=%.6f",
                    sym, db_qty, ex_qty, ex_qty - db_qty,
                )

                if self._telegram:
                    self._telegram._enqueue(
                        f"*Reconciler Qty Mismatch*\n"
                        f"Symbol: `{sym}`\n"
                        f"DB qty: `{db_qty:.6f}`\n"
                        f"Exchange qty: `{ex_qty:.6f}`\n"
                        f"Diff: `{ex_qty - db_qty:.6f}`\n"
                        f"Time: {_ts()}"
                    )
            else:
                result.matched += 1

        # Broadcast reconcile result to dashboard
        self._store._broadcast("reconcile", result.to_dict())

        # Update last reconcile status in store
        self._store.set_last_reconcile(result.to_dict())

        return result

    # ---------------------------------------------------------------------- #
    # Status accessors
    # ---------------------------------------------------------------------- #

    @property
    def last_result(self) -> Optional[ReconcileResult]:
        return self._last_result

    @property
    def run_count(self) -> int:
        return self._run_count

    def get_status(self) -> dict:
        if self._last_result is None:
            return {
                "last_run": None,
                "run_count": self._run_count,
                "status": "never_run",
            }
        lr = self._last_result
        age_sec = (int(time.time() * 1000) - lr.ts) // 1000
        return {
            "last_run":     lr.ts,
            "age_sec":      age_sec,
            "run_count":    self._run_count,
            "matched":      lr.matched,
            "discrepancies": lr.has_discrepancies,
            "errors":        lr.errors,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ts() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
