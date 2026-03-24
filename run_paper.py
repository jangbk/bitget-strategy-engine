"""
22B Strategy Engine — Paper Trading Runner (No API Key Required)

Public API만 사용해서 실시간 데이터 수신 + 레짐 감지 + 전략 시그널 생성.
API 키 없이 동작 (주문 실행 안 함, Paper 모드).

Usage: python run_paper.py
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone

from bot.config import get_config
from bot.data.store import DataStore
from bot.data.collector import BitgetCollector
from bot.regime.detector import RegimeDetector, Regime
from bot.strategies.manager import StrategyManager
from bot.strategies.params_store import StrategyParamsStore
from bot.data.replay_account import ReplayAccount
from db.schema import init_db

# ── Logging ────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("paper_runner")

# Suppress noisy loggers
for name in ["httpx", "httpcore", "ccxt", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# ── Regime emoji ───────────────────────────────────────
REGIME_EMOJI = {
    "BTC_BULLISH": "🟢",
    "BTC_BEARISH": "🔴",
    "BTC_SIDEWAYS": "🟡",
    "ALT_ROTATION": "🔄",
    "HIGH_VOLATILITY": "⚡",
    "LOW_VOLATILITY": "😴",
    "EVENT_RISK": "🚨",
    "UNKNOWN": "❓",
}


async def main():
    config = get_config()
    logger.info("=" * 55)
    logger.info("22B Strategy Engine — PAPER MODE (No API Key)")
    logger.info("Symbols: %s", config.tracked_symbols)
    logger.info("Intervals: %s", config.candle_intervals)
    logger.info("=" * 55)

    # 1. DB
    conn = init_db(config.db_path)
    store = DataStore(conn)
    store.set_system_mode("OBSERVE")

    # 2. Collector (public API only — no auth needed)
    collector = BitgetCollector(config, store)
    await collector.start()

    # 3. Regime Detector
    detector = RegimeDetector(store)

    # 4. Strategy Manager
    StrategyParamsStore.get_instance()
    manager = StrategyManager(store)
    manager.initialize()

    # 5. Paper Account (virtual)
    account = ReplayAccount(
        initial_balance=10_000.0,
        position_size_pct=0.10,
        fee_rate=0.0004,
        slippage_pct=0.0005,
    )
    manager.recorder._replay_account = account

    strategies = [s.name for s in manager._strategies]
    logger.info("Strategies: %s", strategies)
    logger.info("Paper account: $%.2f", account.balance)
    logger.info("Waiting for first data cycle (60s)...")

    # ── Main loop ──────────────────────────────────────
    cycle = 0
    total_signals = 0
    last_regime = None

    try:
        while True:
            await asyncio.sleep(65)  # Wait for data collection
            cycle += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

            # Regime detection
            regime_result = detector.detect()
            if not regime_result:
                logger.info("[Cycle %d] %s — Waiting for enough data...", cycle, now)
                continue

            regime = regime_result.get("regime", "UNKNOWN")
            emoji = REGIME_EMOJI.get(regime, "")

            if regime != last_regime:
                logger.info(
                    "[Cycle %d] %s REGIME CHANGED: %s %s",
                    cycle, now, emoji, regime,
                )
                last_regime = regime

            # Run strategies
            signals = manager.run_all(regime_result)
            active_signals = [s for s in signals if s.action != "SKIP"]
            total_signals += len(active_signals)

            # Log signals
            for sig in active_signals:
                side_emoji = "🟢" if sig.action == "BUY" else "🔴"
                logger.info(
                    "  %s %s %s %s conf=%.2f [%s]",
                    side_emoji, sig.action, sig.symbol,
                    sig.strategy, sig.confidence, sig.reason[:60],
                )

            # Status line
            # Check paper positions
            cursor = conn.cursor()
            open_count = cursor.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE status='OPEN'"
            ).fetchone()[0]
            closed_count = cursor.execute(
                "SELECT COUNT(*) FROM paper_positions WHERE status='CLOSED'"
            ).fetchone()[0]

            logger.info(
                "[Cycle %d] %s %s %s | Signals: %d (total %d) | "
                "Paper: %d open, %d closed | Account: $%.2f",
                cycle, now, emoji, regime,
                len(active_signals), total_signals,
                open_count, closed_count, account.balance,
            )

    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping...")
    finally:
        await collector.stop()
        logger.info("Paper runner stopped. Total cycles: %d, Total signals: %d", cycle, total_signals)

        # Final summary
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT strategy, symbol, side, entry_price, pnl_pct, close_reason "
            "FROM paper_positions WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 10"
        ).fetchall()
        if rows:
            logger.info("\nRecent closed trades:")
            for r in rows:
                pnl = r[4] or 0
                emoji = "✅" if pnl > 0 else "❌"
                logger.info(
                    "  %s %s %s %s entry=%.4f pnl=%+.2f%% (%s)",
                    emoji, r[0], r[1], r[2], r[3], pnl, r[5] or "?"
                )


if __name__ == "__main__":
    asyncio.run(main())
