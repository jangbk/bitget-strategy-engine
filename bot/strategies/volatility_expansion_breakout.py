"""
Strategy B — Volatility Expansion Breakout (v1.3 PART 3.2B)

압축(squeeze) 후 폭발 초기만 노린다. 지속 추세 전체 추종 금지.
기존 range_breakout을 강화 재구현.

진입 조건 (BUY):
  - Bollinger Band squeeze: 현재 BB 폭이 최근 20봉 중 최소값 근처
    (bb_bw <= bb_bw_20_min * 1.15)
  - 이후 BB 폭발: bb_bw_cur > bb_bw_prev * 1.2 (폭발 확인)
  - 가격 > 20봉 최고점 (range_high)
  - 거래량 > 2.0x 20봉 평균 (강한 유동성 조건)
  - 스프레드 임계: change_pct 기반 과열 없음

진입 조건 (SELL):
  - 동일, 가격 < 20봉 최저점

보유 시간: 15분~3시간
TP = range_size * 0.8 (보수적 measured move)
SL = range_high * 0.995 (LONG) / range_low * 1.005 (SHORT)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from bot.strategies._base import Signal, StrategyBase

if TYPE_CHECKING:
    from bot.data.store import DataStore

logger = logging.getLogger(__name__)

MIN_CANDLES = 60


class VolatilityExpansionBreakoutStrategy(StrategyBase):
    """Squeeze 후 폭발 초기 구간만 노리는 브레이크아웃 전략."""

    name:          str       = "volatility_expansion_breakout"
    category:      str       = "breakout"
    regime_filter: List[str] = [
        "BTC_BULLISH", "BTC_SIDEWAYS", "LOW_VOLATILITY",
    ]

    RANGE_BARS:    int   = 20
    BB_PERIOD:     int   = 20
    BB_STD:        float = 2.0
    # Squeeze: BB 폭이 최근 20봉 중 최솟값의 1.15배 이내
    SQUEEZE_RATIO: float = 1.15
    # Expansion: 직전 봉 대비 BB 폭이 1.2배 이상
    EXPAND_RATIO:  float = 1.20
    VOL_MULT:      float = 2.5     # 2.0 → 2.5 (거래량 필터 강화, 노이즈 제거)
    SL_OFFSET:     float = 0.008   # 0.005 → 0.008 (SL 넓혀서 조기 청산 방지)
    TP_RATIO:      float = 0.50    # 0.80 → 0.50 (작은 TP, 빠른 수익 확정 — R:R<1 최적)
    INTERVAL:      str   = "1h"

    def compute(self, store: "DataStore", regime: dict) -> List[Signal]:
        from bot.config import get_config
        config = get_config()
        current_regime = regime.get("regime", "UNKNOWN")
        signals: List[Signal] = []
        for symbol in config.tracked_symbols:
            sig = self._evaluate_symbol(store, symbol, current_regime)
            if sig is not None:
                signals.append(sig)
        return signals

    def _evaluate_symbol(
        self, store: "DataStore", symbol: str, regime: str
    ) -> Optional[Signal]:
        candles = store.get_candles(symbol, self.INTERVAL, limit=MIN_CANDLES + 10)
        if len(candles) < MIN_CANDLES:
            return None

        df = pd.DataFrame(candles).sort_values("ts").reset_index(drop=True)
        close  = df["c"].astype(float)
        high   = df["h"].astype(float)
        low    = df["l"].astype(float)
        volume = df["v"].astype(float)

        # ─── Bollinger Band ──────────────────────────────────────────────── #
        bb_mid = close.rolling(self.BB_PERIOD).mean()
        bb_std = close.rolling(self.BB_PERIOD).std()
        bb_up  = bb_mid + self.BB_STD * bb_std
        bb_lo  = bb_mid - self.BB_STD * bb_std
        bb_bw  = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)

        bb_bw_cur    = float(bb_bw.iloc[-1])
        bb_bw_prev   = float(bb_bw.iloc[-2])
        bb_bw_20_min = float(bb_bw.iloc[-21:-1].min()) if len(bb_bw) >= 21 else float(bb_bw.min())

        if pd.isna(bb_bw_cur) or pd.isna(bb_bw_prev) or bb_bw_20_min <= 0:
            return None

        # ─── Range ───────────────────────────────────────────────────────── #
        range_high = float(high.iloc[-(self.RANGE_BARS + 1): -1].max())
        range_low  = float(low.iloc[-(self.RANGE_BARS + 1): -1].min())
        range_size = range_high - range_low
        if range_size <= 0:
            return None

        # ─── Volume ──────────────────────────────────────────────────────── #
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_cur = float(volume.iloc[-1])
        vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1.0

        price_cur = float(close.iloc[-1])

        # 런타임 파라미터
        vol_mult     = self.get_param("vol_mult",     self.VOL_MULT)
        tp_ratio     = self.get_param("tp_ratio",     self.TP_RATIO)
        sl_offset    = self.get_param("sl_offset",    self.SL_OFFSET)
        squeeze_ratio = self.get_param("squeeze_ratio", self.SQUEEZE_RATIO)
        expand_ratio  = self.get_param("expand_ratio",  self.EXPAND_RATIO)

        # Squeeze/Expansion 재계산 (런타임 파라미터 반영)
        squeeze_confirmed   = bb_bw_prev <= bb_bw_20_min * squeeze_ratio
        expansion_confirmed = bb_bw_prev > 0 and bb_bw_cur > bb_bw_prev * expand_ratio
        if not (squeeze_confirmed and expansion_confirmed):
            return None

        # ─── BUY ─────────────────────────────────────────────────────────── #
        if price_cur > range_high and vol_ratio >= vol_mult:
            confidence = self._clamp(vol_ratio / 4.0, 0.5, 1.0)
            tp = round(price_cur + range_size * tp_ratio, 8)
            sl = round(range_high * (1 - sl_offset), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="BUY", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=(
                    f"VEB BUY: squeeze→expand BB={bb_bw_cur:.4f}, "
                    f"breakout>{range_high:.4f}, vol={vol_ratio:.1f}x"
                ),
            )

        # ─── SELL ────────────────────────────────────────────────────────── #
        if price_cur < range_low and vol_ratio >= vol_mult:
            confidence = self._clamp(vol_ratio / 4.0, 0.5, 1.0)
            tp = round(price_cur - range_size * tp_ratio, 8)
            sl = round(range_low * (1 + sl_offset), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="SELL", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=(
                    f"VEB SELL: squeeze→expand BB={bb_bw_cur:.4f}, "
                    f"breakdown<{range_low:.4f}, vol={vol_ratio:.1f}x"
                ),
            )

        return None
