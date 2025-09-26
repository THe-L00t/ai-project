#!/usr/bin/env python3
"""
기술적 지표 계산 모듈
RSI(9), MACD(3-10-16), 볼린저 밴드, 거래량 분석
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""

    def __init__(self):
        self.data_cache = {}

    def calculate_rsi(self, prices, period=9):
        """
        RSI(9) 계산

        Args:
            prices: 가격 리스트 (최신순)
            period: RSI 기간 (기본 9)

        Returns:
            RSI 값 (0~100)
        """
        try:
            if len(prices) < period + 1:
                return 50.0  # 기본값

            # pandas Series로 변환
            price_series = pd.Series(prices)

            # 가격 변화 계산
            delta = price_series.diff()

            # 상승/하락 분리
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 평균 상승/하락 계산
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # RS 계산
            rs = avg_gain / avg_loss

            # RSI 계산
            rsi = 100 - (100 / (1 + rs))

            # 최신 RSI 값 반환
            latest_rsi = rsi.iloc[-1]
            return latest_rsi if not np.isnan(latest_rsi) else 50.0

        except Exception as e:
            logger.error(f"RSI 계산 오류: {e}")
            return 50.0

    def calculate_macd(self, prices, fast=3, slow=10, signal=16):
        """
        MACD(3-10-16) 계산

        Args:
            prices: 가격 리스트
            fast: 빠른 이동평균선 (기본 3)
            slow: 느린 이동평균선 (기본 10)
            signal: 시그널선 (기본 16)

        Returns:
            dict: {'macd': MACD값, 'signal': 시그널값, 'histogram': 히스토그램값}
        """
        try:
            if len(prices) < max(slow, signal) + 1:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

            # pandas Series로 변환
            price_series = pd.Series(prices)

            # 지수이동평균 계산
            ema_fast = price_series.ewm(span=fast).mean()
            ema_slow = price_series.ewm(span=slow).mean()

            # MACD 계산
            macd = ema_fast - ema_slow

            # 시그널선 계산
            signal_line = macd.ewm(span=signal).mean()

            # 히스토그램 계산
            histogram = macd - signal_line

            return {
                'macd': macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0,
                'signal': signal_line.iloc[-1] if not np.isnan(signal_line.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not np.isnan(histogram.iloc[-1]) else 0.0
            }

        except Exception as e:
            logger.error(f"MACD 계산 오류: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        볼린저 밴드 계산

        Args:
            prices: 가격 리스트
            period: 이동평균 기간 (기본 20)
            std_dev: 표준편차 배수 (기본 2)

        Returns:
            dict: {'upper': 상단밴드, 'middle': 중간선, 'lower': 하단밴드, 'position': 위치%}
        """
        try:
            if len(prices) < period:
                current_price = prices[-1] if prices else 0
                return {
                    'upper': current_price * 1.02,
                    'middle': current_price,
                    'lower': current_price * 0.98,
                    'position': 50.0
                }

            # pandas Series로 변환
            price_series = pd.Series(prices)

            # 이동평균 계산
            middle = price_series.rolling(window=period).mean()

            # 표준편차 계산
            std = price_series.rolling(window=period).std()

            # 상단/하단 밴드 계산
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)

            # 현재 가격의 볼린저 밴드 내 위치 계산 (0~100%)
            current_price = prices[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            current_middle = middle.iloc[-1]

            if not np.isnan(current_upper) and not np.isnan(current_lower):
                position = ((current_price - current_lower) / (current_upper - current_lower)) * 100
                position = max(0, min(100, position))  # 0~100% 범위로 제한
            else:
                position = 50.0

            return {
                'upper': current_upper if not np.isnan(current_upper) else current_price * 1.02,
                'middle': current_middle if not np.isnan(current_middle) else current_price,
                'lower': current_lower if not np.isnan(current_lower) else current_price * 0.98,
                'position': position
            }

        except Exception as e:
            logger.error(f"볼린저 밴드 계산 오류: {e}")
            current_price = prices[-1] if prices else 0
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position': 50.0
            }

    def analyze_volume(self, volumes, prices):
        """
        거래량 분석

        Args:
            volumes: 거래량 리스트
            prices: 가격 리스트

        Returns:
            dict: {'avg_volume': 평균거래량, 'volume_ratio': 비율, 'volume_price_trend': 추세}
        """
        try:
            if not volumes or len(volumes) < 2:
                return {'avg_volume': 1.0, 'volume_ratio': 1.0, 'volume_price_trend': 'neutral'}

            # 평균 거래량 계산 (최근 10개)
            recent_volumes = volumes[-10:]
            avg_volume = np.mean(recent_volumes)

            # 현재 거래량 비율
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # 거래량-가격 추세 분석
            if len(prices) >= 2 and len(volumes) >= 2:
                price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
                volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if volumes[-2] > 0 else 0

                # 거래량과 가격 방향성 분석
                if price_change > 0 and volume_change > 20:
                    trend = 'bullish_confirmed'  # 상승 + 거래량 증가
                elif price_change < 0 and volume_change > 20:
                    trend = 'bearish_confirmed'  # 하락 + 거래량 증가
                elif price_change > 0 and volume_change < -10:
                    trend = 'bullish_weak'  # 상승 + 거래량 감소
                elif price_change < 0 and volume_change < -10:
                    trend = 'bearish_weak'  # 하락 + 거래량 감소
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'

            return {
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_price_trend': trend
            }

        except Exception as e:
            logger.error(f"거래량 분석 오류: {e}")
            return {'avg_volume': 1.0, 'volume_ratio': 1.0, 'volume_price_trend': 'neutral'}

    def generate_technical_signal(self, rsi, macd_data, bollinger_data, volume_data):
        """
        기술적 지표 종합 신호 생성

        Returns:
            dict: {'signal': 'BUY'/'SELL'/'HOLD', 'confidence': 0.0~1.0, 'reasons': []}
        """
        try:
            signals = []
            reasons = []

            # RSI 신호
            if rsi < 30:
                signals.append(('BUY', 0.7))
                reasons.append(f"RSI 과매도 {rsi:.1f}")
            elif rsi > 70:
                signals.append(('SELL', 0.7))
                reasons.append(f"RSI 과매수 {rsi:.1f}")
            elif rsi < 40:
                signals.append(('BUY', 0.3))
                reasons.append(f"RSI 매수권 진입 {rsi:.1f}")
            elif rsi > 60:
                signals.append(('SELL', 0.3))
                reasons.append(f"RSI 매도권 진입 {rsi:.1f}")

            # MACD 신호
            macd_val = macd_data['macd']
            signal_val = macd_data['signal']
            histogram = macd_data['histogram']

            if macd_val > signal_val and histogram > 0:
                signals.append(('BUY', 0.6))
                reasons.append("MACD 골든크로스")
            elif macd_val < signal_val and histogram < 0:
                signals.append(('SELL', 0.6))
                reasons.append("MACD 데드크로스")

            # 볼린저 밴드 신호
            bb_position = bollinger_data['position']
            if bb_position < 20:
                signals.append(('BUY', 0.5))
                reasons.append(f"볼린저 하단 근접 {bb_position:.1f}%")
            elif bb_position > 80:
                signals.append(('SELL', 0.5))
                reasons.append(f"볼린저 상단 근접 {bb_position:.1f}%")

            # 거래량 신호
            volume_trend = volume_data['volume_price_trend']
            volume_ratio = volume_data['volume_ratio']

            if volume_trend == 'bullish_confirmed' and volume_ratio > 1.5:
                signals.append(('BUY', 0.4))
                reasons.append(f"거래량 급증 상승 확인 {volume_ratio:.1f}x")
            elif volume_trend == 'bearish_confirmed' and volume_ratio > 1.5:
                signals.append(('SELL', 0.4))
                reasons.append(f"거래량 급증 하락 확인 {volume_ratio:.1f}x")

            # 신호 통합
            if not signals:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reasons': ['기술적 신호 없음']}

            # 가중 투표
            buy_weight = sum(w for s, w in signals if s == 'BUY')
            sell_weight = sum(w for s, w in signals if s == 'SELL')

            if buy_weight > sell_weight:
                return {'signal': 'BUY', 'confidence': min(0.9, buy_weight), 'reasons': reasons}
            elif sell_weight > buy_weight:
                return {'signal': 'SELL', 'confidence': min(0.9, sell_weight), 'reasons': reasons}
            else:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reasons': reasons}

        except Exception as e:
            logger.error(f"기술적 신호 생성 오류: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reasons': ['신호 계산 오류']}