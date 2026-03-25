"""
Strategy: VWAP Bounce (고승률 전략 #2)

핵심 원리:
  VWAP(거래량가중평균가)는 기관 트레이더의 기준 가격.
  가격이 VWAP에서 이탈 후 되돌아올 때 진입.
  VWAP + RSI 조합 = 65~72% 승률 (리서치 기반).

1h 봉에서 VWAP 근사:
  rolling VWAP = sum(close * volume) / sum(volume) (20봉)

진입 조건 (BUY):
  - 가격이 VWAP 아래에서 VWAP로 바운스 (현재봉 close > VWAP)
  - 직전봉 close < VWAP (방금 돌파)
  - RSI(14) 40~60 (과열 아님, 중립)
  - 거래량 > 평균의 1.2배 (유동성 확인)

진입 조건 (SELL):
  - 가격이 VWAP 위에서 VWAP로 하락 (현재봉 close < VWAP)
  - 직전봉 close > VWAP
  - RSI(14) 40~60
  - 거래량 > 평균의 1.2배

R:R < 1.0:
  TP = 0.8% (매우 작은 TP, 높은 WR 추구)
  SL = 1.2% (넓은 SL)

레짐: BTC_BULLISH, BTC_SIDEWAYS (추세+횡보)
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

MIN_CANDLES = 25


class VWAPBounceStrategy(StrategyBase):
    """VWAP 바운스 전략 — 기관 기준가 되돌림."""

    name:          str       = "vwap_bounce"
    category:      str       = "mean_reversion"
    regime_filter: List[str] = [
        "BTC_BULLISH", "BTC_SIDEWAYS", "LOW_VOLATILITY", "ALT_ROTATION",
    ]

    VWAP_PERIOD:    int   = 20
    RSI_PERIOD:     int   = 14
    RSI_LOW:        float = 40.0
    RSI_HIGH:       float = 60.0
    VOL_MULT:       float = 1.2
    TP_PCT:         float = 0.008  # 0.8%
    SL_PCT:         float = 0.012  # 1.2%
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
        candles = store.get_candles(symbol, self.INTERVAL, limit=MIN_CANDLES + 5)
        if len(candles) < MIN_CANDLES:
            return None

        df = pd.DataFrame(candles).sort_values("ts").reset_index(drop=True)
        close = df["c"].astype(float)
        volume = df["v"].astype(float)
        open_p = df["o"].astype(float)

        # Rolling VWAP (20봉)
        typical_price = close  # simplified: use close as typical price
        vwap = (typical_price * volume).rolling(self.VWAP_PERIOD).sum() / volume.rolling(self.VWAP_PERIOD).sum()

        vwap_cur = float(vwap.iloc[-1])
        vwap_prev = float(vwap.iloc[-2])
        if pd.isna(vwap_cur) or pd.isna(vwap_prev):
            return None

        price_cur = float(close.iloc[-1])
        price_prev = float(close.iloc[-2])

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(span=self.RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_cur = float(rsi.iloc[-1])
        if pd.isna(rsi_cur):
            return None

        # Volume check
        vol_avg = float(volume.rolling(self.VWAP_PERIOD).mean().iloc[-1])
        vol_cur = float(volume.iloc[-1])
        vol_ok = vol_cur >= vol_avg * self.VOL_MULT if vol_avg > 0 else False

        # RSI neutral zone
        rsi_neutral = self.RSI_LOW <= rsi_cur <= self.RSI_HIGH

        # 런타임 파라미터
        tp_pct = self.get_param("tp_pct", self.TP_PCT)
        sl_pct = self.get_param("sl_pct", self.SL_PCT)

        # BUY: 아래에서 VWAP 돌파
        if price_prev < vwap_prev and price_cur > vwap_cur and rsi_neutral and vol_ok:
            # VWAP 대비 거리로 confidence
            dist = abs(price_cur - vwap_cur) / vwap_cur
            confidence = self._clamp(0.7 - dist * 10, 0.5, 0.9)
            tp = round(price_cur * (1 + tp_pct), 8)
            sl = round(price_cur * (1 - sl_pct), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="BUY", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=f"VWAP bounce UP: price crossed VWAP={vwap_cur:.2f}, RSI={rsi_cur:.1f}, vol={vol_cur/vol_avg:.1f}x",
            )

        # SELL: 위에서 VWAP 하방 돌파
        if price_prev > vwap_prev and price_cur < vwap_cur and rsi_neutral and vol_ok:
            dist = abs(price_cur - vwap_cur) / vwap_cur
            confidence = self._clamp(0.7 - dist * 10, 0.5, 0.9)
            tp = round(price_cur * (1 - tp_pct), 8)
            sl = round(price_cur * (1 + sl_pct), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="SELL", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=f"VWAP bounce DOWN: price crossed VWAP={vwap_cur:.2f}, RSI={rsi_cur:.1f}, vol={vol_cur/vol_avg:.1f}x",
            )

        return None
