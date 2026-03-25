"""
Strategy: RSI-3 Mean Reversion (고승률 전략 #1)

핵심 원리:
  RSI(3) 극단값에서 평균 회귀를 노린다.
  단기 RSI(3)는 가격의 과매수/과매도를 매우 민감하게 포착.
  참고: QuantifiedStrategies.com 연구에서 RSI-3 전략 91% 승률 보고.

진입 조건 (BUY):
  - RSI(3) < 10 (극심한 과매도)
  - 가격이 EMA(50) 위 (상승 추세 확인)
  - 직전 2봉 연속 음봉 (되돌림 확인)

진입 조건 (SELL):
  - RSI(3) > 90 (극심한 과매수)
  - 가격이 EMA(50) 아래 (하락 추세 확인)
  - 직전 2봉 연속 양봉 (되돌림 확인)

R:R < 1.0 설계:
  TP = 1.0% (작게, 빠르게 수익 확정)
  SL = 1.5% (넓게, 숨쉴 공간)

레짐: BTC_BULLISH, BTC_BEARISH, BTC_SIDEWAYS (전 시장 조건)
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


class RSI3ReversalStrategy(StrategyBase):
    """RSI(3) 극단값 평균회귀 전략 — 고승률, 작은 수익."""

    name:          str       = "rsi3_reversal"
    category:      str       = "mean_reversion"
    regime_filter: List[str] = [
        "BTC_BULLISH", "BTC_BEARISH", "BTC_SIDEWAYS", "LOW_VOLATILITY",
    ]

    RSI_PERIOD:     int   = 3
    RSI_OVERSOLD:   float = 10.0
    RSI_OVERBOUGHT: float = 90.0
    EMA_PERIOD:     int   = 50
    CONSEC_BARS:    int   = 2      # 연속 음봉/양봉 수
    TP_PCT:         float = 0.010  # 1.0% (작은 TP)
    SL_PCT:         float = 0.015  # 1.5% (넓은 SL)
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

        # RSI(3)
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

        # EMA(50)
        ema = close.ewm(span=self.EMA_PERIOD, adjust=False).mean()
        ema_cur = float(ema.iloc[-1])
        price = float(close.iloc[-1])

        # 연속 봉 체크
        consec_red = all(float(close.iloc[-(i+1)]) < float(df["o"].astype(float).iloc[-(i+1)])
                         for i in range(self.CONSEC_BARS))
        consec_green = all(float(close.iloc[-(i+1)]) > float(df["o"].astype(float).iloc[-(i+1)])
                           for i in range(self.CONSEC_BARS))

        # 런타임 파라미터
        tp_pct = self.get_param("tp_pct", self.TP_PCT)
        sl_pct = self.get_param("sl_pct", self.SL_PCT)

        # BUY: RSI(3) < 10 + 가격 > EMA50 + 연속 음봉
        if rsi_cur < self.RSI_OVERSOLD and price > ema_cur and consec_red:
            confidence = self._clamp((self.RSI_OVERSOLD - rsi_cur) / self.RSI_OVERSOLD, 0.5, 1.0)
            tp = round(price * (1 + tp_pct), 8)
            sl = round(price * (1 - sl_pct), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="BUY", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=f"RSI3={rsi_cur:.1f}<{self.RSI_OVERSOLD} + price>{ema_cur:.0f}(EMA50) + {self.CONSEC_BARS}연속음봉",
            )

        # SELL: RSI(3) > 90 + 가격 < EMA50 + 연속 양봉
        if rsi_cur > self.RSI_OVERBOUGHT and price < ema_cur and consec_green:
            confidence = self._clamp((rsi_cur - self.RSI_OVERBOUGHT) / (100 - self.RSI_OVERBOUGHT), 0.5, 1.0)
            tp = round(price * (1 - tp_pct), 8)
            sl = round(price * (1 + sl_pct), 8)
            return Signal(
                strategy=self.name, symbol=symbol,
                action="SELL", mode=self._PHASE2_MODE,
                confidence=round(confidence, 4), regime=regime,
                tp=tp, sl=sl,
                reason=f"RSI3={rsi_cur:.1f}>{self.RSI_OVERBOUGHT} + price<{ema_cur:.0f}(EMA50) + {self.CONSEC_BARS}연속양봉",
            )

        return None
