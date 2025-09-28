#!/usr/bin/env python3
"""
고도화된 매수 타이밍 결정 알고리즘
- 다중 시간대 RSI 분석
- MACD 다이버전스 감지
- 볼린저 밴드 스퀴즈 및 확장 감지
- 거래량 급증 및 가격 돌파 분석
- 지지선 반등 확인
- 시장 구조 변화 감지
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SignalResult:
    """신호 결과 데이터 클래스"""
    action: str  # 'BUY', 'SELL', 'WAIT', 'ANALYZE_MORE'
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: List[str]
    timeframe: str
    risk_reward_ratio: float
    signal_strength: float

class AdvancedEntrySignalEngine:
    """기존 자동매매 AI에 통합할 고도화된 매수 신호 엔진"""

    def __init__(self, upbit_api, config=None):
        self.upbit = upbit_api
        self.config = config or {}

        # 신호 가중치 설정
        self.signal_weights = {
            'multi_timeframe_rsi': 0.25,      # 다중 시간대 RSI 분석
            'macd_divergence': 0.20,          # MACD 다이버전스 감지
            'bollinger_squeeze': 0.15,        # 볼린저 밴드 스퀴즈 후 확장
            'volume_breakout': 0.15,          # 거래량 급증 + 가격 돌파
            'support_bounce': 0.15,           # 지지선 반등 확인
            'market_structure': 0.10          # 시장 구조 변화 감지
        }

        # 임계값 설정 (AI_SETTINGS.md에서 로드)
        self.thresholds = {
            'min_confidence': self.config.get('ADVANCED_ENTRY_MIN_CONFIDENCE', 70),
            'rsi_oversold': self.config.get('RSI_OVERSOLD_THRESHOLD', 25),
            'rsi_overbought': self.config.get('RSI_OVERBOUGHT_THRESHOLD', 75),
            'volume_spike_multiplier': self.config.get('VOLUME_SPIKE_MULTIPLIER', 2.5),
            'bollinger_squeeze_ratio': self.config.get('BOLLINGER_SQUEEZE_RATIO', 0.8),
            'divergence_min_strength': 60   # 다이버전스 최소 강도
        }

        # 시간대별 설정
        self.timeframes = {
            'primary': '15m',     # 주 분석 시간대
            'secondary': ['1m', '5m', '1h', '4h'],  # 보조 분석 시간대
            'weights': {          # 시간대별 가중치
                '1m': 0.1,
                '5m': 0.15,
                '15m': 0.3,
                '1h': 0.25,
                '4h': 0.2
            }
        }

        # 캐시
        self.signal_cache = {}
        self.candle_cache = {}
        self.cache_ttl = 30  # 30초

        logger.info("✅ 고도화된 매수 타이밍 알고리즘 초기화 완료")

    def generate_buy_signal_sync(self, market: str, market_data: Dict) -> SignalResult:
        """
        동기 버전의 매수 신호 생성 함수 (기존 시스템 호환성)

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            market_data: 현재 시장 데이터

        Returns:
            SignalResult: 종합 매수 신호 결과
        """
        try:
            # 캐시 확인
            cache_key = f"{market}_{int(time.time() / self.cache_ttl)}"
            if cache_key in self.signal_cache:
                return self.signal_cache[cache_key]

            logger.debug(f"🔍 {market} 고도화된 매수 신호 분석 시작 (동기)")

            # 1. 다중 시간대 분석 (동기 버전)
            multi_tf_analysis = self.analyze_multiple_timeframes_sync(market)

            # 2. 기술적 지표 종합 분석 (동기 버전)
            technical_signals = self.calculate_technical_indicators_sync(market, market_data)

            # 3. 시장 구조 분석 (동기 버전)
            market_structure = self.analyze_market_structure_sync(market)

            # 4. 종합 신호 생성
            final_signal = self.synthesize_signals(
                market, multi_tf_analysis, technical_signals, market_structure, market_data
            )

            # 캐시 저장
            self.signal_cache[cache_key] = final_signal

            # 오래된 캐시 정리
            self.cleanup_cache()

            return final_signal

        except Exception as e:
            logger.error(f"❌ 매수 신호 생성 실패 ({market}): {e}")
            return SignalResult(
                action='WAIT',
                confidence=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                reasoning=[f"신호 생성 오류: {str(e)}"],
                timeframe='15m',
                risk_reward_ratio=0,
                signal_strength=0
            )

    async def generate_buy_signal(self, market: str, market_data: Dict) -> SignalResult:
        """
        기존 매매 시스템에서 호출할 통합 매수 신호 생성 함수

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            market_data: 현재 시장 데이터

        Returns:
            SignalResult: 종합 매수 신호 결과
        """
        try:
            # 캐시 확인
            cache_key = f"{market}_{int(time.time() / self.cache_ttl)}"
            if cache_key in self.signal_cache:
                return self.signal_cache[cache_key]

            logger.debug(f"🔍 {market} 고도화된 매수 신호 분석 시작")

            # 1. 다중 시간대 분석
            multi_tf_analysis = await self.analyze_multiple_timeframes(market)

            # 2. 기술적 지표 종합 분석
            technical_signals = await self.calculate_technical_indicators(market, market_data)

            # 3. 시장 구조 분석
            market_structure = await self.analyze_market_structure(market)

            # 4. 종합 신호 생성
            final_signal = self.synthesize_signals(
                market, multi_tf_analysis, technical_signals, market_structure, market_data
            )

            # 캐시 저장
            self.signal_cache[cache_key] = final_signal

            # 오래된 캐시 정리
            self.cleanup_cache()

            return final_signal

        except Exception as e:
            logger.error(f"❌ 매수 신호 생성 실패 ({market}): {e}")
            return SignalResult(
                action='WAIT',
                confidence=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                reasoning=[f"신호 생성 오류: {str(e)}"],
                timeframe='15m',
                risk_reward_ratio=0,
                signal_strength=0
            )

    async def analyze_multiple_timeframes(self, market: str) -> Dict:
        """다중 시간대 RSI 분석"""
        try:
            timeframes = ['1m', '5m', '15m', '1h', '4h']
            rsi_signals = {}

            for tf in timeframes:
                # 시간대별 캔들 데이터 조회
                candle_data = await self.get_candle_data(market, tf, 50)
                if not candle_data or len(candle_data) < 14:
                    continue

                # RSI 계산
                closes = [float(candle['trade_price']) for candle in candle_data]
                rsi = self.calculate_rsi(closes, period=14)

                if len(rsi) == 0:
                    continue

                current_rsi = rsi[-1]
                rsi_trend = rsi[-3:] if len(rsi) >= 3 else [current_rsi]

                # RSI 신호 분석
                rsi_signals[tf] = {
                    'value': current_rsi,
                    'signal': self.interpret_rsi_signal(current_rsi, rsi_trend),
                    'divergence': self.detect_rsi_divergence(closes, rsi),
                    'weight': self.timeframes['weights'].get(tf, 0.1)
                }

            # 다중 시간대 신호 종합
            consolidated = self.consolidate_timeframe_signals(rsi_signals)

            logger.debug(f"📊 {market} 다중 시간대 RSI 분석: {len(rsi_signals)}개 시간대")
            return consolidated

        except Exception as e:
            logger.error(f"다중 시간대 분석 실패 ({market}): {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reasoning': []}

    def analyze_multiple_timeframes_sync(self, market: str) -> Dict:
        """다중 시간대 RSI 분석 (동기 버전)"""
        try:
            timeframes = ['1m', '5m', '15m', '1h', '4h']
            rsi_signals = {}

            for tf in timeframes:
                # 시간대별 캔들 데이터 조회 (동기)
                candle_data = self.get_candle_data_sync(market, tf, 50)
                if not candle_data or len(candle_data) < 14:
                    continue

                # RSI 계산
                closes = [float(candle['trade_price']) for candle in candle_data]
                rsi = self.calculate_rsi(closes, period=14)

                if len(rsi) == 0:
                    continue

                current_rsi = rsi[-1]
                rsi_trend = rsi[-3:] if len(rsi) >= 3 else [current_rsi]

                # RSI 신호 분석
                rsi_signals[tf] = {
                    'value': current_rsi,
                    'signal': self.interpret_rsi_signal(current_rsi, rsi_trend),
                    'divergence': self.detect_rsi_divergence(closes, rsi),
                    'weight': self.timeframes['weights'].get(tf, 0.1)
                }

            # 다중 시간대 신호 종합
            consolidated = self.consolidate_timeframe_signals(rsi_signals)

            logger.debug(f"📊 {market} 다중 시간대 RSI 분석: {len(rsi_signals)}개 시간대")
            return consolidated

        except Exception as e:
            logger.error(f"다중 시간대 분석 실패 ({market}): {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reasoning': []}

    def interpret_rsi_signal(self, rsi_value: float, rsi_trend: List[float]) -> Dict:
        """RSI 값과 추세 기반 신호 해석"""
        try:
            trend_direction = 'UP' if len(rsi_trend) >= 2 and rsi_trend[-1] > rsi_trend[-2] else 'DOWN'

            if rsi_value <= self.thresholds['rsi_oversold']:
                if trend_direction == 'UP':
                    return {'signal': 'STRONG_BUY', 'score': 90}
                return {'signal': 'BUY', 'score': 75}
            elif rsi_value <= 35:
                if trend_direction == 'UP':
                    return {'signal': 'BUY', 'score': 70}
                return {'signal': 'CAUTIOUS_BUY', 'score': 55}
            elif 35 < rsi_value < 65:
                return {'signal': 'NEUTRAL', 'score': 50}
            elif rsi_value >= self.thresholds['rsi_overbought']:
                return {'signal': 'AVOID', 'score': 20}
            else:
                return {'signal': 'CAUTIOUS', 'score': 40}

        except Exception as e:
            logger.error(f"RSI 신호 해석 실패: {e}")
            return {'signal': 'NEUTRAL', 'score': 50}

    def detect_rsi_divergence(self, prices: List[float], rsi_values: List[float]) -> Dict:
        """RSI 다이버전스 감지"""
        try:
            if len(prices) < 20 or len(rsi_values) < 20:
                return {'type': 'NONE', 'strength': 0}

            # 최근 20개 데이터에서 저점 찾기
            price_lows = self.find_local_minima(prices[-20:])
            rsi_lows = self.find_local_minima(rsi_values[-20:])

            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                # 강세 다이버전스 확인 (가격은 하락, RSI는 상승)
                if (price_lows[-1]['value'] < price_lows[-2]['value'] and
                    rsi_lows[-1]['value'] > rsi_lows[-2]['value']):

                    strength = abs(rsi_lows[-1]['value'] - rsi_lows[-2]['value']) * 2
                    return {
                        'type': 'BULLISH_DIVERGENCE',
                        'strength': min(strength, 100),
                        'price_trend': 'DOWN',
                        'rsi_trend': 'UP'
                    }

            return {'type': 'NONE', 'strength': 0}

        except Exception as e:
            logger.error(f"RSI 다이버전스 감지 실패: {e}")
            return {'type': 'NONE', 'strength': 0}

    async def calculate_technical_indicators(self, market: str, market_data: Dict) -> Dict:
        """기술적 지표 종합 분석"""
        try:
            # 15분봉 데이터 조회 (주 분석 시간대)
            candle_data = await self.get_candle_data(market, '15m', 100)
            if not candle_data or len(candle_data) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            closes = [float(candle['trade_price']) for candle in candle_data]
            volumes = [float(candle['candle_acc_trade_volume']) for candle in candle_data]

            # 1. MACD 다이버전스 분석
            macd_analysis = await self.detect_macd_divergence(market, closes)

            # 2. 볼린저 밴드 스퀴즈 및 돌파 분석
            bollinger_analysis = await self.detect_bollinger_squeeze_breakout(market, closes)

            # 3. 거래량 급증 및 가격 돌파 분석
            volume_analysis = await self.detect_volume_breakout(market, closes, volumes)

            # 4. 종합 신호 계산
            technical_signal = self.combine_technical_signals(
                macd_analysis, bollinger_analysis, volume_analysis
            )

            return technical_signal

        except Exception as e:
            logger.error(f"기술적 지표 분석 실패 ({market}): {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    def calculate_technical_indicators_sync(self, market: str, market_data: Dict) -> Dict:
        """기술적 지표 종합 분석 (동기 버전)"""
        try:
            # 15분봉 데이터 조회 (주 분석 시간대)
            candle_data = self.get_candle_data_sync(market, '15m', 100)
            if not candle_data or len(candle_data) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            closes = [float(candle['trade_price']) for candle in candle_data]
            volumes = [float(candle['candle_acc_trade_volume']) for candle in candle_data]

            # 1. MACD 다이버전스 분석
            macd_analysis = self.detect_macd_divergence_sync(market, closes)

            # 2. 볼린저 밴드 스퀴즈 및 돌파 분석
            bollinger_analysis = self.detect_bollinger_squeeze_breakout_sync(market, closes)

            # 3. 거래량 급증 및 가격 돌파 분석
            volume_analysis = self.detect_volume_breakout_sync(market, closes, volumes)

            # 4. 종합 신호 계산
            technical_signal = self.combine_technical_signals(
                macd_analysis, bollinger_analysis, volume_analysis
            )

            return technical_signal

        except Exception as e:
            logger.error(f"기술적 지표 분석 실패 ({market}): {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    async def detect_macd_divergence(self, market: str, closes: List[float]) -> Dict:
        """MACD 다이버전스 감지"""
        try:
            if len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # MACD 계산
            macd_line, signal_line, histogram = self.calculate_macd(closes)

            if len(macd_line) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 가격 저점과 MACD 저점 비교 (최근 20개)
            price_lows = self.find_local_minima(closes[-20:])
            macd_lows = self.find_local_minima(macd_line[-20:])

            # 강세 다이버전스 확인
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if (price_lows[-1]['value'] < price_lows[-2]['value'] and
                    macd_lows[-1]['value'] > macd_lows[-2]['value']):

                    strength = abs(macd_lows[-1]['value'] - macd_lows[-2]['value']) * 100
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'BULLISH_DIVERGENCE',
                        'confidence': min(85, 60 + strength),
                        'reasoning': 'MACD 강세 다이버전스 감지'
                    }

            # MACD 라인과 시그널 라인 교차 확인
            if len(macd_line) >= 2 and len(signal_line) >= 2:
                if (macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]):
                    return {
                        'signal': 'BUY',
                        'type': 'MACD_CROSSOVER',
                        'confidence': 65,
                        'reasoning': 'MACD 골든 크로스'
                    }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"MACD 다이버전스 감지 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    async def detect_bollinger_squeeze_breakout(self, market: str, closes: List[float]) -> Dict:
        """볼린저 밴드 스퀴즈 및 확장 감지"""
        try:
            if len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 볼린저 밴드 계산
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes, period=20)

            if len(bb_upper) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 밴드 폭 계산
            band_width = [(upper - lower) / middle * 100
                         for upper, middle, lower in zip(bb_upper, bb_middle, bb_lower)]

            current_width = band_width[-1]
            avg_width = np.mean(band_width[-20:])

            # 스퀴즈 조건 확인
            is_squeeze = current_width < avg_width * self.thresholds['bollinger_squeeze_ratio']

            current_price = closes[-1]

            if is_squeeze:
                # 상단 밴드 돌파 확인
                if current_price > bb_upper[-1]:
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'BOLLINGER_BREAKOUT',
                        'confidence': 80,
                        'entry_price': current_price,
                        'stop_loss': bb_middle[-1],
                        'take_profit': current_price + (current_price - bb_middle[-1]) * 2,
                        'reasoning': '볼린저 밴드 스퀴즈 후 상단 돌파'
                    }

            # 하단 밴드 근처에서 반등 신호
            if current_price <= bb_lower[-1] * 1.02:  # 하단 밴드 근처 (2% 여유)
                return {
                    'signal': 'BUY',
                    'type': 'BOLLINGER_BOUNCE',
                    'confidence': 70,
                    'reasoning': '볼린저 밴드 하단에서 반등 예상'
                }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"볼린저 밴드 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    async def detect_volume_breakout(self, market: str, closes: List[float], volumes: List[float]) -> Dict:
        """거래량 급증과 함께 발생하는 가격 돌파 신호"""
        try:
            if len(volumes) < 20 or len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            current_volume = volumes[-1]
            avg_volume_20 = np.mean(volumes[-20:-1])  # 최근 20개 평균 (현재 제외)

            # 거래량 급증 확인
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_spike = volume_ratio > self.thresholds['volume_spike_multiplier']

            if volume_spike:
                # 저항선 레벨 찾기
                resistance_level = self.find_resistance_level(closes[-50:])
                current_price = closes[-1]

                # 가격 돌파 확인
                if current_price > resistance_level * 1.005:  # 0.5% 돌파
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'VOLUME_BREAKOUT',
                        'confidence': 85,
                        'volume_ratio': volume_ratio,
                        'breakout_level': resistance_level,
                        'entry_price': current_price,
                        'stop_loss': resistance_level * 0.98,
                        'reasoning': f'거래량 {volume_ratio:.1f}배 급증과 함께 저항선 돌파'
                    }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"거래량 돌파 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    async def analyze_market_structure(self, market: str) -> Dict:
        """시장 구조 변화 감지"""
        try:
            # 4시간봉으로 전체적인 시장 구조 파악
            candle_data = await self.get_candle_data(market, '4h', 50)
            if not candle_data or len(candle_data) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 50}

            closes = [float(candle['trade_price']) for candle in candle_data]
            highs = [float(candle['high_price']) for candle in candle_data]
            lows = [float(candle['low_price']) for candle in candle_data]

            # 고점과 저점 분석
            higher_highs = self.check_higher_highs(highs[-10:])
            higher_lows = self.check_higher_lows(lows[-10:])

            # 상승 추세 확인
            if higher_highs and higher_lows:
                return {
                    'signal': 'BUY',
                    'trend': 'UPTREND',
                    'confidence': 75,
                    'reasoning': '상승 추세 지속: 고점과 저점 모두 상승'
                }

            # 지지선 테스트 확인
            support_level = min(lows[-10:])
            current_price = closes[-1]

            if abs(current_price - support_level) / support_level < 0.02:  # 2% 이내
                return {
                    'signal': 'BUY',
                    'type': 'SUPPORT_TEST',
                    'confidence': 70,
                    'support_level': support_level,
                    'reasoning': '주요 지지선 근처에서 반등 기대'
                }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"시장 구조 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 50}

    def synthesize_signals(self, market: str, multi_tf: Dict, technical: Dict,
                          market_structure: Dict, market_data: Dict) -> SignalResult:
        """모든 신호를 종합하여 최종 매수 결정"""
        try:
            current_price = float(market_data.get('trade_price', 0))
            if current_price == 0:
                return SignalResult(
                    action='WAIT', confidence=0, entry_price=0, stop_loss=0,
                    take_profit=0, position_size=0, reasoning=['가격 정보 없음'],
                    timeframe='15m', risk_reward_ratio=0, signal_strength=0
                )

            # 신호 점수 계산
            total_score = 0
            total_weight = 0
            reasoning = []

            # 1. 다중 시간대 RSI 신호
            if multi_tf.get('signal') in ['BUY', 'STRONG_BUY']:
                score = multi_tf.get('confidence', 0)
                weight = self.signal_weights['multi_timeframe_rsi']
                total_score += score * weight
                total_weight += weight
                reasoning.append(f"다중 시간대 RSI: {multi_tf.get('signal', 'UNKNOWN')}")

            # 2. 기술적 지표 신호
            if technical.get('signal') in ['BUY', 'STRONG_BUY']:
                score = technical.get('confidence', 0)
                weight = self.signal_weights['macd_divergence']
                total_score += score * weight
                total_weight += weight
                reasoning.append(f"기술적 지표: {technical.get('reasoning', 'N/A')}")

            # 3. 시장 구조 신호
            if market_structure.get('signal') == 'BUY':
                score = market_structure.get('confidence', 0)
                weight = self.signal_weights['market_structure']
                total_score += score * weight
                total_weight += weight
                reasoning.append(f"시장 구조: {market_structure.get('reasoning', 'N/A')}")

            # 최종 신뢰도 계산
            final_confidence = total_score / total_weight if total_weight > 0 else 0

            # 매수 결정
            if final_confidence >= self.thresholds['min_confidence'] and len(reasoning) >= 2:
                # 포지션 크기 계산 (신뢰도에 따라 조절)
                base_position_size = 0.15  # 기본 15%
                confidence_multiplier = min(final_confidence / 100, 1.0)
                position_size = base_position_size * confidence_multiplier

                # 손절가와 목표가 계산
                stop_loss = current_price * 0.97  # 3% 손절
                take_profit = current_price * 1.06  # 6% 익절
                risk_reward = (take_profit - current_price) / (current_price - stop_loss)

                return SignalResult(
                    action='BUY',
                    confidence=final_confidence,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    reasoning=reasoning,
                    timeframe='15m',
                    risk_reward_ratio=risk_reward,
                    signal_strength=final_confidence
                )

            return SignalResult(
                action='WAIT',
                confidence=final_confidence,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                reasoning=reasoning or ['신호 강도 부족'],
                timeframe='15m',
                risk_reward_ratio=0,
                signal_strength=final_confidence
            )

        except Exception as e:
            logger.error(f"신호 종합 실패: {e}")
            return SignalResult(
                action='WAIT', confidence=0, entry_price=0, stop_loss=0,
                take_profit=0, position_size=0, reasoning=[f'오류: {str(e)}'],
                timeframe='15m', risk_reward_ratio=0, signal_strength=0
            )

    # ==========================================================================
    # 보조 함수들
    # ==========================================================================

    async def get_candle_data(self, market: str, timeframe: str, count: int) -> List[Dict]:
        """캔들 데이터 조회 (캐시 포함)"""
        try:
            cache_key = f"{market}_{timeframe}_{int(time.time() / 60)}"  # 1분 캐시

            if cache_key in self.candle_cache:
                return self.candle_cache[cache_key]

            # 시간대별 캔들 데이터 조회
            if timeframe == '1m':
                data = self.upbit.GetMinuteCandles(market, count)
            elif timeframe == '5m':
                data = self.upbit.GetCandles(market, 'minutes', 5, count)
            elif timeframe == '15m':
                data = self.upbit.GetCandles(market, 'minutes', 15, count)
            elif timeframe == '1h':
                data = self.upbit.GetCandles(market, 'minutes', 60, count)
            elif timeframe == '4h':
                data = self.upbit.GetCandles(market, 'minutes', 240, count)
            else:
                data = self.upbit.GetMinuteCandles(market, count)

            if data:
                self.candle_cache[cache_key] = data

            return data or []

        except Exception as e:
            logger.error(f"캔들 데이터 조회 실패: {e}")
            return []

    def get_candle_data_sync(self, market: str, timeframe: str, count: int) -> List[Dict]:
        """캔들 데이터 조회 동기 버전 (캐시 포함)"""
        try:
            cache_key = f"{market}_{timeframe}_{int(time.time() / 60)}"  # 1분 캐시

            if cache_key in self.candle_cache:
                return self.candle_cache[cache_key]

            # 시간대별 캔들 데이터 조회
            if timeframe == '1m':
                data = self.upbit.GetMinuteCandles(market, count)
            elif timeframe == '5m':
                data = self.upbit.GetCandles(market, 'minutes', 5, count)
            elif timeframe == '15m':
                data = self.upbit.GetCandles(market, 'minutes', 15, count)
            elif timeframe == '1h':
                data = self.upbit.GetCandles(market, 'minutes', 60, count)
            elif timeframe == '4h':
                data = self.upbit.GetCandles(market, 'minutes', 240, count)
            else:
                data = self.upbit.GetMinuteCandles(market, count)

            if data:
                self.candle_cache[cache_key] = data

            return data or []

        except Exception as e:
            logger.error(f"캔들 데이터 조회 실패: {e}")
            return []

    def detect_macd_divergence_sync(self, market: str, closes: List[float]) -> Dict:
        """MACD 다이버전스 감지 (동기 버전)"""
        return self.detect_macd_divergence_internal(market, closes)

    def detect_bollinger_squeeze_breakout_sync(self, market: str, closes: List[float]) -> Dict:
        """볼린저 밴드 스퀴즈 및 확장 감지 (동기 버전)"""
        return self.detect_bollinger_squeeze_breakout_internal(market, closes)

    def detect_volume_breakout_sync(self, market: str, closes: List[float], volumes: List[float]) -> Dict:
        """거래량 급증과 함께 발생하는 가격 돌파 신호 (동기 버전)"""
        return self.detect_volume_breakout_internal(market, closes, volumes)

    def analyze_market_structure_sync(self, market: str) -> Dict:
        """시장 구조 변화 감지 (동기 버전)"""
        try:
            # 4시간봉으로 전체적인 시장 구조 파악
            candle_data = self.get_candle_data_sync(market, '4h', 50)
            if not candle_data or len(candle_data) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 50}

            closes = [float(candle['trade_price']) for candle in candle_data]
            highs = [float(candle['high_price']) for candle in candle_data]
            lows = [float(candle['low_price']) for candle in candle_data]

            # 고점과 저점 분석
            higher_highs = self.check_higher_highs(highs[-10:])
            higher_lows = self.check_higher_lows(lows[-10:])

            # 상승 추세 확인
            if higher_highs and higher_lows:
                return {
                    'signal': 'BUY',
                    'trend': 'UPTREND',
                    'confidence': 75,
                    'reasoning': '상승 추세 지속: 고점과 저점 모두 상승'
                }

            # 지지선 테스트 확인
            support_level = min(lows[-10:])
            current_price = closes[-1]

            if abs(current_price - support_level) / support_level < 0.02:  # 2% 이내
                return {
                    'signal': 'BUY',
                    'type': 'SUPPORT_TEST',
                    'confidence': 70,
                    'support_level': support_level,
                    'reasoning': '주요 지지선 근처에서 반등 기대'
                }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"시장 구조 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 50}

    def detect_macd_divergence_internal(self, market: str, closes: List[float]) -> Dict:
        """MACD 다이버전스 감지 내부 로직"""
        try:
            if len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # MACD 계산
            macd_line, signal_line, histogram = self.calculate_macd(closes)

            if len(macd_line) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 가격 저점과 MACD 저점 비교 (최근 20개)
            price_lows = self.find_local_minima(closes[-20:])
            macd_lows = self.find_local_minima(macd_line[-20:])

            # 강세 다이버전스 확인
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if (price_lows[-1]['value'] < price_lows[-2]['value'] and
                    macd_lows[-1]['value'] > macd_lows[-2]['value']):

                    strength = abs(macd_lows[-1]['value'] - macd_lows[-2]['value']) * 100
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'BULLISH_DIVERGENCE',
                        'confidence': min(85, 60 + strength),
                        'reasoning': 'MACD 강세 다이버전스 감지'
                    }

            # MACD 라인과 시그널 라인 교차 확인
            if len(macd_line) >= 2 and len(signal_line) >= 2:
                if (macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]):
                    return {
                        'signal': 'BUY',
                        'type': 'MACD_CROSSOVER',
                        'confidence': 65,
                        'reasoning': 'MACD 골든 크로스'
                    }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"MACD 다이버전스 감지 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    def detect_bollinger_squeeze_breakout_internal(self, market: str, closes: List[float]) -> Dict:
        """볼린저 밴드 스퀴즈 및 확장 감지 내부 로직"""
        try:
            if len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 볼린저 밴드 계산
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes, period=20)

            if len(bb_upper) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # 밴드 폭 계산
            band_width = [(upper - lower) / middle * 100
                         for upper, middle, lower in zip(bb_upper, bb_middle, bb_lower)]

            current_width = band_width[-1]
            avg_width = np.mean(band_width[-20:])

            # 스퀴즈 조건 확인
            is_squeeze = current_width < avg_width * self.thresholds['bollinger_squeeze_ratio']

            current_price = closes[-1]

            if is_squeeze:
                # 상단 밴드 돌파 확인
                if current_price > bb_upper[-1]:
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'BOLLINGER_BREAKOUT',
                        'confidence': 80,
                        'entry_price': current_price,
                        'stop_loss': bb_middle[-1],
                        'take_profit': current_price + (current_price - bb_middle[-1]) * 2,
                        'reasoning': '볼린저 밴드 스퀴즈 후 상단 돌파'
                    }

            # 하단 밴드 근처에서 반등 신호
            if current_price <= bb_lower[-1] * 1.02:  # 하단 밴드 근처 (2% 여유)
                return {
                    'signal': 'BUY',
                    'type': 'BOLLINGER_BOUNCE',
                    'confidence': 70,
                    'reasoning': '볼린저 밴드 하단에서 반등 예상'
                }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"볼린저 밴드 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    def detect_volume_breakout_internal(self, market: str, closes: List[float], volumes: List[float]) -> Dict:
        """거래량 급증과 함께 발생하는 가격 돌파 신호 내부 로직"""
        try:
            if len(volumes) < 20 or len(closes) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            current_volume = volumes[-1]
            avg_volume_20 = np.mean(volumes[-20:-1])  # 최근 20개 평균 (현재 제외)

            # 거래량 급증 확인
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_spike = volume_ratio > self.thresholds['volume_spike_multiplier']

            if volume_spike:
                # 저항선 레벨 찾기
                resistance_level = self.find_resistance_level(closes[-50:])
                current_price = closes[-1]

                # 가격 돌파 확인
                if current_price > resistance_level * 1.005:  # 0.5% 돌파
                    return {
                        'signal': 'STRONG_BUY',
                        'type': 'VOLUME_BREAKOUT',
                        'confidence': 85,
                        'volume_ratio': volume_ratio,
                        'breakout_level': resistance_level,
                        'entry_price': current_price,
                        'stop_loss': resistance_level * 0.98,
                        'reasoning': f'거래량 {volume_ratio:.1f}배 급증과 함께 저항선 돌파'
                    }

            return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"거래량 돌파 분석 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return []

            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            rsi_values = []

            for i in range(period, len(deltas)):
                if avg_loss == 0:
                    rsi_values.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_values.append(rsi)

                # 다음 계산을 위한 평균 업데이트
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            return rsi_values

        except Exception as e:
            logger.error(f"RSI 계산 실패: {e}")
            return []

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD 계산"""
        try:
            if len(prices) < slow + signal:
                return [], [], []

            # EMA 계산
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)

            # MACD 라인
            macd_line = [fast_val - slow_val for fast_val, slow_val in
                        zip(ema_fast[slow-fast:], ema_slow)]

            # 시그널 라인
            signal_line = self.calculate_ema(macd_line, signal)

            # 히스토그램
            histogram = [macd - sig for macd, sig in
                        zip(macd_line[signal-1:], signal_line)]

            return macd_line[signal-1:], signal_line, histogram

        except Exception as e:
            logger.error(f"MACD 계산 실패: {e}")
            return [], [], []

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """EMA 계산"""
        try:
            if len(prices) < period:
                return []

            multiplier = 2 / (period + 1)
            ema_values = [np.mean(prices[:period])]  # 첫 번째 EMA는 SMA

            for price in prices[period:]:
                ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)

            return ema_values

        except Exception as e:
            logger.error(f"EMA 계산 실패: {e}")
            return []

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """볼린저 밴드 계산"""
        try:
            if len(prices) < period:
                return [], [], []

            upper_band = []
            middle_band = []
            lower_band = []

            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                sma = np.mean(window)
                std = np.std(window)

                middle_band.append(sma)
                upper_band.append(sma + (std * std_dev))
                lower_band.append(sma - (std * std_dev))

            return upper_band, middle_band, lower_band

        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {e}")
            return [], [], []

    def find_local_minima(self, data: List[float]) -> List[Dict]:
        """지역 최솟값 찾기"""
        try:
            minima = []
            for i in range(1, len(data) - 1):
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    minima.append({'index': i, 'value': data[i]})
            return minima
        except:
            return []

    def find_resistance_level(self, prices: List[float]) -> float:
        """저항선 레벨 찾기"""
        try:
            if len(prices) < 10:
                return max(prices) if prices else 0

            # 최근 고점들 찾기
            highs = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    highs.append(prices[i])

            # 가장 많이 테스트된 저항선 레벨 찾기
            if highs:
                return max(highs)
            else:
                return max(prices)

        except Exception as e:
            logger.error(f"저항선 레벨 계산 실패: {e}")
            return max(prices) if prices else 0

    def consolidate_timeframe_signals(self, rsi_signals: Dict) -> Dict:
        """다중 시간대 신호 통합"""
        try:
            if not rsi_signals:
                return {'signal': 'NEUTRAL', 'confidence': 0, 'reasoning': []}

            total_score = 0
            total_weight = 0
            buy_signals = 0
            strong_buy_signals = 0

            for tf, data in rsi_signals.items():
                signal = data['signal']
                weight = data['weight']

                if signal['signal'] == 'STRONG_BUY':
                    total_score += signal['score'] * weight
                    strong_buy_signals += 1
                elif signal['signal'] in ['BUY', 'CAUTIOUS_BUY']:
                    total_score += signal['score'] * weight
                    buy_signals += 1
                else:
                    total_score += 50 * weight  # 중성 신호

                total_weight += weight

            avg_score = total_score / total_weight if total_weight > 0 else 50

            # 다수의 시간대에서 매수 신호가 나오면 신뢰도 증가
            if strong_buy_signals >= 2:
                return {
                    'signal': 'STRONG_BUY',
                    'confidence': min(avg_score + 15, 95),
                    'reasoning': [f'{strong_buy_signals}개 시간대에서 강매수 신호']
                }
            elif buy_signals + strong_buy_signals >= 3:
                return {
                    'signal': 'BUY',
                    'confidence': min(avg_score + 10, 85),
                    'reasoning': [f'{buy_signals + strong_buy_signals}개 시간대에서 매수 신호']
                }
            else:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': avg_score,
                    'reasoning': ['시간대별 신호 일치도 부족']
                }

        except Exception as e:
            logger.error(f"시간대 신호 통합 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reasoning': []}

    def combine_technical_signals(self, macd: Dict, bollinger: Dict, volume: Dict) -> Dict:
        """기술적 지표 신호 통합"""
        try:
            signals = [macd, bollinger, volume]
            buy_signals = [s for s in signals if s.get('signal') in ['BUY', 'STRONG_BUY']]

            if len(buy_signals) >= 2:
                avg_confidence = np.mean([s.get('confidence', 50) for s in buy_signals])
                reasoning = [s.get('reasoning', 'N/A') for s in buy_signals if s.get('reasoning')]

                return {
                    'signal': 'BUY',
                    'confidence': avg_confidence,
                    'reasoning': '; '.join(reasoning)
                }
            else:
                return {'signal': 'NEUTRAL', 'confidence': 50}

        except Exception as e:
            logger.error(f"기술적 신호 통합 실패: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 50}

    def check_higher_highs(self, highs: List[float]) -> bool:
        """고점 상승 확인"""
        try:
            if len(highs) < 4:
                return False
            return highs[-1] > highs[-3] and highs[-3] > highs[-5] if len(highs) >= 5 else highs[-1] > highs[-2]
        except:
            return False

    def check_higher_lows(self, lows: List[float]) -> bool:
        """저점 상승 확인"""
        try:
            if len(lows) < 4:
                return False
            return lows[-1] > lows[-3] and lows[-3] > lows[-5] if len(lows) >= 5 else lows[-1] > lows[-2]
        except:
            return False

    def cleanup_cache(self):
        """오래된 캐시 정리"""
        try:
            current_time = int(time.time())

            # 신호 캐시 정리 (5분 이상된 것)
            expired_signal_keys = [
                key for key in self.signal_cache.keys()
                if current_time - int(key.split('_')[-1]) * self.cache_ttl > 300
            ]
            for key in expired_signal_keys:
                del self.signal_cache[key]

            # 캔들 캐시 정리 (10분 이상된 것)
            expired_candle_keys = [
                key for key in self.candle_cache.keys()
                if current_time - int(key.split('_')[-1]) * 60 > 600
            ]
            for key in expired_candle_keys:
                del self.candle_cache[key]

        except Exception as e:
            logger.debug(f"캐시 정리 실패: {e}")