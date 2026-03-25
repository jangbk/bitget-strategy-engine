"""
22B Strategy Engine — Parameter Optimizer

Grid search over strategy parameters to find optimal expectancy.
Uses 2025-01~08 as training, 2025-09~26-03 as validation.
"""

import asyncio
import itertools
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from bot.config import get_config
from bot.data.store import DataStore
from bot.data.validation_dataset_loader import ValidationDatasetLoader
from bot.data.validation_replay import ValidationReplaySession
from bot.data.replay_account import ReplayAccount
from bot.regime.detector import RegimeDetector
from bot.strategies.manager import StrategyManager
from bot.strategies.params_store import StrategyParamsStore
from db.schema import init_db

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("optimizer")
logger.setLevel(logging.INFO)


async def run_single(data_dir: str, db_path: str, params: dict) -> dict:
    """Run one backtest with given parameter overrides. Returns stats dict."""
    conn = init_db(db_path)
    store = DataStore(conn)
    store.set_system_mode("OBSERVE")

    loader = ValidationDatasetLoader(store, data_dir)
    summary = await loader.load(warmup_bars=52)
    if summary.replay_bars_remaining == 0:
        return {"trades": 0, "expectancy": -999}

    replay = ValidationReplaySession(
        store=store, datasets=loader.get_replay_datasets(),
        warmup_bars=52, step_delay_ms=0, max_steps=0,
    )
    account = ReplayAccount(
        initial_balance=10_000, position_size_pct=0.10,
        fee_rate=0.0004, slippage_pct=0.0005,
    )

    StrategyParamsStore.get_instance()
    detector = RegimeDetector(store)
    manager = StrategyManager(store)
    manager.initialize()
    manager.recorder._replay_account = account

    # Apply parameter overrides
    for s in manager._strategies:
        if s.name in params:
            for k, v in params[s.name].items():
                setattr(s, k, v)

    step = 0
    while True:
        bar = await replay.next_bar()
        if bar is None:
            break
        step += 1
        regime_result = detector.detect()
        if regime_result:
            manager.run_all(regime_result)

    cursor = conn.cursor()
    closed = cursor.execute("SELECT pnl_pct FROM paper_positions WHERE status='CLOSED'").fetchall()
    wins = sum(1 for r in closed if (r[0] or 0) > 0)
    losses = sum(1 for r in closed if (r[0] or 0) <= 0)
    total = len(closed)

    if total == 0:
        conn.close()
        return {"trades": 0, "expectancy": -999, "win_rate": 0, "pnl": 0}

    avg_win = sum(r[0] for r in closed if (r[0] or 0) > 0) / wins if wins else 0
    avg_loss = abs(sum(r[0] for r in closed if (r[0] or 0) <= 0) / losses) if losses else 0
    wr = wins / total * 100
    exp = (wr / 100 * avg_win) - ((100 - wr) / 100 * avg_loss)
    pnl = (account.balance / account.initial_balance - 1) * 100

    # Strategy breakdown
    strats = cursor.execute(
        "SELECT strategy, COUNT(*), SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END), AVG(pnl_pct) "
        "FROM paper_positions WHERE status='CLOSED' GROUP BY strategy"
    ).fetchall()

    conn.close()
    import os
    try:
        os.remove(db_path)
    except:
        pass

    return {
        "trades": total, "wins": wins, "losses": losses,
        "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "expectancy": exp, "pnl": pnl,
        "strategies": {r[0]: {"trades": r[1], "wr": ((r[2] or 0) / r[1] * 100), "avg": r[3] or 0} for r in strats},
    }


async def main():
    train_dir = "data/backtest/Period1_2025-01~08"
    val_dir = "data/backtest/Period2_2025-09~2026-03"

    # Verify data exists
    for d in [train_dir, val_dir]:
        p = Path(d)
        if not p.exists() or not list(p.glob("*/*.json")):
            logger.error(f"Data not found: {d}. Run backtest_period.py first.")
            return

    # ── Define parameter grid ──────────────────────────────────────
    # VEB: main strategy (most trades)
    veb_grid = {
        "TP_RATIO": [0.5, 0.8, 1.0, 1.5, 2.0],
        "SL_OFFSET": [0.003, 0.005, 0.008, 0.012, 0.015],
        "VOL_MULT": [1.5, 2.0, 2.5, 3.0],
        "SQUEEZE_RATIO": [1.1, 1.15, 1.2],
        "EXPAND_RATIO": [1.1, 1.2, 1.3],
    }

    # Overreaction Reversal
    or_grid = {
        "TP_PCT": [0.015, 0.025, 0.035, 0.05],
        "SL_PCT": [0.008, 0.012, 0.018, 0.025],
        "RSI_OVERSOLD": [25, 28, 32],
        "RSI_OVERBOUGHT": [68, 72, 75],
        "OVERREACTION_PCT": [0.02, 0.03, 0.04, 0.05],
    }

    # Early Trend Capture
    etc_grid = {
        "TP_PCT": [0.015, 0.02, 0.03, 0.04],
        "SL_PCT": [0.008, 0.01, 0.015],
        "RSI_LOW": [35, 40, 45],
        "RSI_HIGH": [55, 60, 65],
    }

    # ── Phase 1: Optimize VEB (most impactful — 502 trades) ──────
    logger.info("=" * 60)
    logger.info("PHASE 1: Optimizing volatility_expansion_breakout")
    logger.info("=" * 60)

    # Sample subset to keep runtime manageable
    # Focus on TP_RATIO and SL_OFFSET (most impact on expectancy)
    veb_combos = list(itertools.product(
        veb_grid["TP_RATIO"],
        veb_grid["SL_OFFSET"],
        [2.0, 2.5],  # VOL_MULT subset
        [1.15],       # SQUEEZE_RATIO default
        [1.2],        # EXPAND_RATIO default
    ))
    logger.info(f"VEB combinations: {len(veb_combos)}")

    best_veb = {"expectancy": -999}
    results_veb = []

    for i, (tp, sl, vol, sq, ex) in enumerate(veb_combos):
        params = {"volatility_expansion_breakout": {
            "TP_RATIO": tp, "SL_OFFSET": sl, "VOL_MULT": vol,
            "SQUEEZE_RATIO": sq, "EXPAND_RATIO": ex,
        }}
        db = f"data/backtest/opt_veb_{i}.db"
        r = await run_single(train_dir, db, params)
        r["params"] = params["volatility_expansion_breakout"]

        if r["trades"] >= 10 and r["expectancy"] > best_veb["expectancy"]:
            best_veb = r

        if r["trades"] >= 10:
            results_veb.append(r)

        if (i + 1) % 10 == 0:
            logger.info(f"  VEB {i+1}/{len(veb_combos)} — best exp: {best_veb.get('expectancy', -999):+.3f}%")

    # Show top 5 VEB
    results_veb.sort(key=lambda x: x["expectancy"], reverse=True)
    logger.info("\nTop 5 VEB parameter sets:")
    for j, r in enumerate(results_veb[:5]):
        p = r["params"]
        logger.info(
            f"  #{j+1} EXP={r['expectancy']:+.3f}% WR={r['win_rate']:.0f}% "
            f"trades={r['trades']} TP={p['TP_RATIO']} SL={p['SL_OFFSET']} VOL={p['VOL_MULT']}"
        )

    # ── Phase 2: Optimize Overreaction Reversal ──────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Optimizing overreaction_reversal")
    logger.info("=" * 60)

    or_combos = list(itertools.product(
        or_grid["TP_PCT"],
        or_grid["SL_PCT"],
        [28],  # RSI_OVERSOLD default
        [72],  # RSI_OVERBOUGHT default
        or_grid["OVERREACTION_PCT"],
    ))
    logger.info(f"OR combinations: {len(or_combos)}")

    best_or = {"expectancy": -999}
    results_or = []

    for i, (tp, sl, rsi_lo, rsi_hi, over) in enumerate(or_combos):
        params = {"overreaction_reversal": {
            "TP_PCT": tp, "SL_PCT": sl, "RSI_OVERSOLD": rsi_lo,
            "RSI_OVERBOUGHT": rsi_hi, "OVERREACTION_PCT": over,
        }}
        db = f"data/backtest/opt_or_{i}.db"
        r = await run_single(train_dir, db, params)
        r["params"] = params["overreaction_reversal"]

        if r["trades"] >= 5 and r["expectancy"] > best_or["expectancy"]:
            best_or = r

        if r["trades"] >= 5:
            results_or.append(r)

        if (i + 1) % 10 == 0:
            logger.info(f"  OR {i+1}/{len(or_combos)} — best exp: {best_or.get('expectancy', -999):+.3f}%")

    results_or.sort(key=lambda x: x["expectancy"], reverse=True)
    logger.info("\nTop 5 OR parameter sets:")
    for j, r in enumerate(results_or[:5]):
        p = r["params"]
        logger.info(
            f"  #{j+1} EXP={r['expectancy']:+.3f}% WR={r['win_rate']:.0f}% "
            f"trades={r['trades']} TP={p['TP_PCT']} SL={p['SL_PCT']} OVER={p['OVERREACTION_PCT']}"
        )

    # ── Phase 3: Validate best combo on Period 2 ─────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: VALIDATION — Best params on Period 2 (out-of-sample)")
    logger.info("=" * 60)

    best_params = {}
    if best_veb.get("trades", 0) >= 10:
        best_params["volatility_expansion_breakout"] = best_veb["params"]
    if best_or.get("trades", 0) >= 5:
        best_params["overreaction_reversal"] = best_or["params"]

    logger.info(f"Best VEB params: {best_params.get('volatility_expansion_breakout', 'N/A')}")
    logger.info(f"Best OR params: {best_params.get('overreaction_reversal', 'N/A')}")

    # Run on training (in-sample)
    r_train = await run_single(train_dir, "data/backtest/opt_best_train.db", best_params)
    logger.info(f"\nTRAINING (in-sample): trades={r_train['trades']} WR={r_train['win_rate']:.1f}% "
                f"EXP={r_train['expectancy']:+.3f}% PnL={r_train['pnl']:+.1f}%")

    # Run on validation (out-of-sample)
    r_val = await run_single(val_dir, "data/backtest/opt_best_val.db", best_params)
    logger.info(f"VALIDATION (out-of-sample): trades={r_val['trades']} WR={r_val['win_rate']:.1f}% "
                f"EXP={r_val['expectancy']:+.3f}% PnL={r_val['pnl']:+.1f}%")

    # Also run original (default params) on validation for comparison
    r_orig = await run_single(val_dir, "data/backtest/opt_orig_val.db", {})
    logger.info(f"ORIGINAL (validation):      trades={r_orig['trades']} WR={r_orig['win_rate']:.1f}% "
                f"EXP={r_orig['expectancy']:+.3f}% PnL={r_orig['pnl']:+.1f}%")

    # ── Final Summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Best Parameters Found:")
    print(json.dumps(best_params, indent=4))

    print(f"\n  {'Metric':20s} {'Original':>12s} {'Optimized':>12s} {'Change':>10s}")
    print(f"  {'-'*54}")
    for metric in ["trades", "win_rate", "expectancy", "pnl"]:
        o = r_orig.get(metric, 0)
        n = r_val.get(metric, 0)
        fmt = ".1f" if metric in ["win_rate", "pnl"] else ".3f" if metric == "expectancy" else "d"
        chg = n - o
        print(f"  {metric:20s} {o:>12{fmt}} {n:>12{fmt}} {chg:>+10{fmt}}")

    print(f"\n  Expectancy improved: {r_orig['expectancy']:+.3f}% → {r_val['expectancy']:+.3f}%")
    if r_val["expectancy"] > 0:
        print(f"  ✅ POSITIVE EXPECTANCY achieved on out-of-sample data!")
    else:
        print(f"  ❌ Still negative — more strategy work needed")
    print(f"{'='*60}")

    # Save best params
    with open("data/best_params.json", "w") as f:
        json.dump({"optimized_params": best_params, "train_result": r_train, "val_result": r_val}, f, indent=2)
    logger.info("Saved to data/best_params.json")


if __name__ == "__main__":
    asyncio.run(main())
