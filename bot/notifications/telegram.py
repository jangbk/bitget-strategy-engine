"""
TelegramNotifier — sends alerts to a Telegram chat via the Bot API.

Phase 1 alerts:
  - System started / stopped
  - Regime change
  - Kill switch triggered
  - Daily summary

Uses httpx async HTTP (no dependency on python-telegram-bot library for Phase 1).
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

from bot.config import Config

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
SEND_TIMEOUT = 10


class TelegramNotifier:
    """Sends Telegram messages via the Bot API."""

    def __init__(self, config: Config) -> None:
        self._token = config.telegram_bot_token
        self._chat_id = config.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._http: Optional[httpx.AsyncClient] = None
        self._last_regime: Optional[str] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

        if not self._enabled:
            logger.warning(
                "Telegram notifier disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set."
            )

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    async def start(self) -> None:
        if not self._enabled:
            return
        self._http = httpx.AsyncClient(
            base_url=TELEGRAM_API_BASE,
            timeout=SEND_TIMEOUT,
        )
        self._task = asyncio.create_task(self._sender_loop(), name="telegram_sender")
        logger.info("TelegramNotifier started.")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._http:
            await self._http.aclose()
        logger.info("TelegramNotifier stopped.")

    # ---------------------------------------------------------------------- #
    # Public alert methods
    # ---------------------------------------------------------------------- #

    def notify_system_started(self, mode: str) -> None:
        msg = (
            f"*22B Strategy Engine Started*\n"
            f"Mode: `{mode}`\n"
            f"Time: {_ts()}"
        )
        self._enqueue(msg)

    def notify_system_stopped(self, reason: str = "normal shutdown") -> None:
        msg = (
            f"*22B Strategy Engine Stopped*\n"
            f"Reason: {reason}\n"
            f"Time: {_ts()}"
        )
        self._enqueue(msg)

    def notify_regime_change(self, old_regime: str, new_regime: str, indicators: dict) -> None:
        if old_regime == new_regime:
            return
        funding = indicators.get("funding", "N/A")
        atr_pct = indicators.get("btc_atr_pct", "N/A")
        ret_24h = indicators.get("btc_ret_24h", "N/A")
        msg = (
            f"*Regime Changed*\n"
            f"{old_regime} → `{new_regime}`\n"
            f"BTC 24H ret: {ret_24h}%\n"
            f"ATR/price: {atr_pct}%\n"
            f"Funding: {funding}%\n"
            f"Time: {_ts()}"
        )
        self._enqueue(msg)
        self._last_regime = new_regime

    def notify_kill_switch(self, reason: str) -> None:
        msg = (
            f"*KILL SWITCH TRIGGERED*\n"
            f"Reason: {reason}\n"
            f"All new entries BLOCKED\n"
            f"Time: {_ts()}"
        )
        self._enqueue(msg)

    def notify_daily_summary(self, summary: dict) -> None:
        pnl = summary.get("pnl", 0.0)
        pnl_pct = summary.get("pnl_pct", 0.0)
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        regime = summary.get("regime", "N/A")
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"*Daily Summary*\n"
            f"P&L: `{sign}{pnl:.2f} USDT ({sign}{pnl_pct:.2f}%)`\n"
            f"Trades: {trades} | Wins: {wins}\n"
            f"Regime: {regime}\n"
            f"Date: {_ts()}"
        )
        self._enqueue(msg)

    async def send_message(self, text: str) -> bool:
        """Directly send a message (awaitable). Returns True on success."""
        if not self._enabled or not self._http:
            return False
        return await self._do_send(text)

    # ---------------------------------------------------------------------- #
    # Internal sender
    # ---------------------------------------------------------------------- #

    def _enqueue(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Telegram queue full — dropping message.")

    async def _sender_loop(self) -> None:
        """Background task that drains the message queue."""
        while True:
            try:
                text = await self._queue.get()
                await self._do_send(text)
                self._queue.task_done()
                await asyncio.sleep(0.5)   # avoid Telegram rate limits (30 msg/s)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Telegram sender loop error: %s", exc)

    async def _do_send(self, text: str) -> bool:
        if not self._http or not self._enabled:
            return False
        # 마크다운 테이블(| |) → 리스트 형식 변환
        text = _sanitize_md_tables(text)
        url = f"/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            resp = await self._http.post(url, json=payload)
            if resp.status_code == 404:
                logger.warning("Telegram: invalid token (404) — disabling notifier.")
                self._enabled = False
                return False
            if resp.status_code != 200:
                logger.warning("Telegram send failed %d: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:
            logger.error("Telegram HTTP error: %s", exc)
            return False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ts() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _sanitize_md_tables(text: str) -> str:
    """마크다운 테이블(| col |)을 Telegram 호환 리스트로 변환."""
    import re
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("|") and s.endswith("|") and s.count("|") >= 3:
            tbl: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                tbl.append(lines[i].strip())
                i += 1
            if len(tbl) >= 2:
                headers = [c.strip() for c in tbl[0].split("|")[1:-1]]
                start = 2 if len(tbl) > 1 and all(c in "-|: " for c in tbl[1]) else 1
                for row in tbl[start:]:
                    cells = [c.strip() for c in row.split("|")[1:-1]]
                    parts = [f"{headers[j]}: {cells[j]}" for j in range(min(len(headers), len(cells))) if cells[j]]
                    result.append("▸ " + " | ".join(parts))
            else:
                result.extend(tbl)
            continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)
