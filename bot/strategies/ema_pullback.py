"""
Strategy: EMA Pullback (고승률 전략 #3)

핵심 원리:
  강한 추세에서 EMA로의 되돌림(pullback)에 진입.
  추세 방향으로만 진입 — 역추세 진입 금지.
  EMA + RSI 조합으로 65~70% 승률 (리서치 기반).

진입 조건 (BUY — 상승 추세):
  - EMA(20) > EMA(50) (상승 추세 확인)
  - 가격이 EMA(20)까지 하락 (pullback)
  - 가격이 EMA(20) 터치 후 반등 (현재봉 close > EMA20, 직전봉 low <= EMA20)
  - RSI(14) 35~55 (과매도에서 반등 시작)
  - ADX > 20 (추세 존재 확인)

진입 조건 (SELL — 하락 추세):
  - EMA(20) < EMA(50)
  - 가격이 EMA(20)까지 상승 후 하락
  - RSI(14) 45~65 (과매수에서 하락 시작)
  - ADX > 20

R:R < 1.0:
  TP = 1.2% (작은 TP)
  SL = 1.5% (EMA 넘어서면 추세 깨진 것)

레짐: BTC_BULLISH, BTC_BEARISH (추세장에서만)
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

MIN_CANDLES = 55


class EMAPullbackStrategy(StrategyBase):
    """EMA 되돌림 진입 전략 — 추세 방향으로만."""

    name:          str       = "ema_pullback"
    category:      str       = "trend"
    regime_filter: List[str] = [
        "BTC_BULLISH", "BTC_BEARISH",
    ]

    EMA_FAST:       int   = 20
    EMA_SLOW:       int   = 50
    RSI_PERIOD:     int   = 14
    ADX_PERIOD:     int   = 14
    ADX_THRESHOLD:  float = 20.0
    RSI_BUY_LOW:    float = 35.0
    RSI_BUY_HIGH:   float = 55.0
    RSI_SELL_LOW:   float = 45.0
    RSI_SELL_HIGH:  float = 65.0
    TP_PCT:         float = 0.012  # 1.2%
    SL_PCT:         float = 0.015  # 1.5%
    INTERVAL:       str   = "1h"

    def compute(self, store: "DataStore", regime: dict) -> List[Signal]:
        from bot.config import get_config
        config = get_config()
        current_regime = regime.get("regime", "UNKNOWN")
        signals: List[Signal] = []

        for symbol in config.tracked_symbols:
            sig = self._evaluate(store, symbol, current_regime)
            if sig is not None:
                signals.append(sig)
        return signals

    def _evaluate(self, store: "DataStore", symbol: str, regime: str) -> Optional[Signal]:
        candles = store.get_candles(symbol, self.INTERVAL, limit=MIN_CANDLES + 10)
        if len(candles) < MIN_CANDLES:
            return None

        df = pd.DataFrame(candles).sort_values("ts").reset_index(drop=True)
        close = df["c"].astype(float)
        high = df["h"].astype(float)
        low = df["l"].astype(float)

        # EMA
        ema_fast = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=self.EMA_SLOW, adjust=False).mean()

        ema_f = float(ema_fast.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        price = float(close.iloc[-1])
        low_cur = float(low.iloc[-1])
        low_prev = float(low.iloc[-2])
        high_cur = float(high.iloc[-1])
        high_prev = float(high.iloc[-2])

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss_s = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.RSI_PERIOD, adjust=False).mean()
        avg_loss = loss_s.ewm(span=self.RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_cur = float(rsi.iloc[-1])
        if pd.isna(rsi_cur):
            return None

        # ADX (simplified using DI difference as proxy)
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.ADX_PERIOD, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=self.ADX_PERIOD, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=self.ADX_PERIOD, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=self.ADX_PERIOD, adjust=False).mean()
        adx_cur = float(adx.iloc[-1])
        if pd.isna(adx_cur):
            return None

        # ADX filter
        if adx_cur < self.ADX_THRESHOLD:
            return None

        # 런타임 파라미터
        tp_pct = self.get_param("tp_pct", self.TP_PCT)
        sl_pct = self.get_param("sl_pct", self.SL_PCT)

        # BUY: 상승 추세 + EMA20 pullback 후 반등
        if ema_f > ema_s:
            # 직전봉 low가 EMA20 터치 (또는 아래)하고, 현재봉 close > EMA20
            ema_f_prev = float(ema_fast.iloc[-2])
            touched = low_prev <= ema_f_prev * 1.002  # 0.2% 마진
            bounced = price > ema_f

            if touched and bounced and self.RSI_BUY_LOW <= rsi_cur <= self.RSI_BUY_HIGH:
                confidence = self._clamp(0.5 + (adx_cur - self.ADX_THRESHOLD) / 60, 0.5, 0.9)
                tp = round(price * (1 + tp_pct), 8)
                sl = round(price * (1 - sl_pct), 8)
                return Signal(
                    strategy=self.name, symbol=symbol,
                    action="BUY", mode=self._PHASE2_MODE,
                    confidence=round(confidence, 4), regime=regime,
                    tp=tp, sl=sl,
                    reason=f"EMA pullback BUY: EMA20={ema_f:.0f}>EMA50={ema_s:.0f}, RSI={rsi_cur:.1f}, ADX={adx_cur:.1f}",
                )

        # SELL: 하락 추세 + EMA20 pullback 후 하락
        if ema_f < ema_s:
            ema_f_prev = float(ema_fast.iloc[-2])
            touched = high_prev >= ema_f_prev * 0.998
            dropped = price < ema_f

            if touched and dropped and self.RSI_SELL_LOW <= rsi_cur <= self.RSI_SELL_HIGH:
                confidence = self._clamp(0.5 + (adx_cur - self.ADX_THRESHOLD) / 60, 0.5, 0.9)
                tp = round(price * (1 - tp_pct), 8)
                sl = round(price * (1 + sl_pct), 8)
                return Signal(
                    strategy=self.name, symbol=symbol,
                    action="SELL", mode=self._PHASE2_MODE,
                    confidence=round(confidence, 4), regime=regime,
                    tp=tp, sl=sl,
                    reason=f"EMA pullback SELL: EMA20={ema_f:.0f}<EMA50={ema_s:.0f}, RSI={rsi_cur:.1f}, ADX={adx_cur:.1f}",
                )

        return None
