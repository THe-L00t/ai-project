"""
리스크 관리 시스템
거래 전 리스크 평가와 실시간 리스크 모니터링을 담당합니다.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskMetrics:
    """리스크 메트릭"""
    level: RiskLevel
    score: float
    factors: List[str]
    recommendations: List[str]


class RiskManager:
    """리스크 관리자"""

    def __init__(self, max_daily_loss: float = 0.1,
                 max_drawdown: float = 0.2,
                 var_confidence: float = 0.95,
                 volatility_threshold: float = 0.3):
        """
        초기화

        Args:
            max_daily_loss: 최대 일일 손실률 (10%)
            max_drawdown: 최대 낙폭률 (20%)
            var_confidence: VaR 신뢰수준 (95%)
            volatility_threshold: 변동성 임계값 (30%)
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.volatility_threshold = volatility_threshold

        # 상태 추적
        self.daily_start_balance = 0
        self.peak_balance = 0
        self.trades_history = []
        self.risk_events = []

        self.logger = logging.getLogger(__name__)

    def assess_trade_risk(self, symbol: str, signal_strength: float,
                         position_size: float, current_price: float,
                         market_data: pd.DataFrame) -> RiskMetrics:
        """개별 거래 리스크 평가"""
        try:
            risk_factors = []
            risk_score = 0.0

            # 1. 변동성 리스크
            volatility = self._calculate_volatility(market_data)
            if volatility > self.volatility_threshold:
                risk_factors.append(f"고변동성: {volatility:.2%}")
                risk_score += 0.3

            # 2. 신호 강도 리스크
            if signal_strength < 0.7:
                risk_factors.append(f"낮은 신호 강도: {signal_strength:.2f}")
                risk_score += 0.2

            # 3. 포지션 크기 리스크
            portfolio_risk = position_size / self.daily_start_balance if self.daily_start_balance > 0 else 0
            if portfolio_risk > 0.2:
                risk_factors.append(f"큰 포지션 크기: {portfolio_risk:.2%}")
                risk_score += 0.2

            # 4. 시장 상황 리스크
            market_trend = self._assess_market_trend(market_data)
            if market_trend == 'BEARISH':
                risk_factors.append("약세 시장 상황")
                risk_score += 0.2

            # 5. 시간대 리스크
            hour = datetime.now().hour
            if hour < 6 or hour > 22:  # 새벽/늦은 밤
                risk_factors.append("저유동성 시간대")
                risk_score += 0.1

            # 리스크 레벨 결정
            if risk_score < 0.3:
                level = RiskLevel.LOW
            elif risk_score < 0.6:
                level = RiskLevel.MEDIUM
            elif risk_score < 0.9:
                level = RiskLevel.HIGH
            else:
                level = RiskLevel.CRITICAL

            # 권장사항 생성
            recommendations = self._generate_recommendations(level, risk_factors)

            return RiskMetrics(
                level=level,
                score=risk_score,
                factors=risk_factors,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"거래 리스크 평가 중 오류: {e}")
            return RiskMetrics(
                level=RiskLevel.HIGH,
                score=0.8,
                factors=["평가 오류"],
                recommendations=["수동 검토 필요"]
            )

    def assess_portfolio_risk(self, positions: Dict, current_prices: Dict[str, float],
                            total_balance: float) -> RiskMetrics:
        """포트폴리오 리스크 평가"""
        try:
            risk_factors = []
            risk_score = 0.0

            # 1. 드로다운 리스크
            if self.peak_balance > 0:
                drawdown = (self.peak_balance - total_balance) / self.peak_balance
                if drawdown > self.max_drawdown * 0.8:  # 80% 도달시 경고
                    risk_factors.append(f"높은 드로다운: {drawdown:.2%}")
                    risk_score += 0.4

            # 2. 일일 손실 리스크
            if self.daily_start_balance > 0:
                daily_loss = (self.daily_start_balance - total_balance) / self.daily_start_balance
                if daily_loss > self.max_daily_loss * 0.5:  # 50% 도달시 경고
                    risk_factors.append(f"일일 손실: {daily_loss:.2%}")
                    risk_score += 0.3

            # 3. 포지션 집중도 리스크
            concentration = self._calculate_concentration(positions, current_prices)
            if concentration > 0.4:
                risk_factors.append(f"포지션 집중도: {concentration:.2%}")
                risk_score += 0.2

            # 4. 상관관계 리스크
            correlation = self._calculate_correlation_risk(positions)
            if correlation > 0.8:
                risk_factors.append(f"높은 상관관계: {correlation:.2f}")
                risk_score += 0.2

            # 5. 레버리지 리스크 (현재는 현물만이므로 0)
            # 향후 마진 거래 지원시 추가

            # 리스크 레벨 결정
            if risk_score < 0.25:
                level = RiskLevel.LOW
            elif risk_score < 0.5:
                level = RiskLevel.MEDIUM
            elif risk_score < 0.8:
                level = RiskLevel.HIGH
            else:
                level = RiskLevel.CRITICAL

            recommendations = self._generate_portfolio_recommendations(level, risk_factors)

            return RiskMetrics(
                level=level,
                score=risk_score,
                factors=risk_factors,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 평가 중 오류: {e}")
            return RiskMetrics(
                level=RiskLevel.HIGH,
                score=0.8,
                factors=["평가 오류"],
                recommendations=["즉시 검토 필요"]
            )

    def check_circuit_breaker(self, current_balance: float) -> bool:
        """서킷 브레이커 체크"""
        try:
            # 1. 일일 손실 한도
            if self.daily_start_balance > 0:
                daily_loss_rate = (self.daily_start_balance - current_balance) / self.daily_start_balance
                if daily_loss_rate >= self.max_daily_loss:
                    self.logger.critical(f"일일 손실 한도 도달: {daily_loss_rate:.2%}")
                    self._log_risk_event("DAILY_LOSS_LIMIT", daily_loss_rate)
                    return True

            # 2. 최대 드로다운
            if self.peak_balance > 0:
                drawdown = (self.peak_balance - current_balance) / self.peak_balance
                if drawdown >= self.max_drawdown:
                    self.logger.critical(f"최대 드로다운 도달: {drawdown:.2%}")
                    self._log_risk_event("MAX_DRAWDOWN", drawdown)
                    return True

            # 3. 급격한 잔고 변화 (10분 내 20% 이상)
            recent_balance_changes = self._get_recent_balance_changes()
            if recent_balance_changes:
                max_change = max(abs(change) for change in recent_balance_changes)
                if max_change > 0.2:
                    self.logger.critical(f"급격한 잔고 변화: {max_change:.2%}")
                    self._log_risk_event("RAPID_BALANCE_CHANGE", max_change)
                    return True

            return False

        except Exception as e:
            self.logger.error(f"서킷 브레이커 체크 중 오류: {e}")
            return True  # 오류 시 안전하게 거래 중단

    def calculate_position_size(self, signal_strength: float,
                              available_balance: float,
                              volatility: float) -> float:
        """리스크 기반 포지션 사이즈 계산"""
        try:
            # Kelly Criterion 기반 계산
            win_prob = signal_strength
            avg_win = 0.1  # 평균 수익률 10%
            avg_loss = 0.05  # 평균 손실률 5%

            if avg_loss > 0:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            else:
                kelly_fraction = 0

            # Kelly 비율 제한 (최대 25%)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))

            # 변동성 조정
            volatility_adjustment = max(0.1, 1 - volatility)
            adjusted_fraction = kelly_fraction * volatility_adjustment

            # 최종 포지션 크기
            position_size = available_balance * adjusted_fraction

            # 최소/최대 제한
            min_position = 5000  # 최소 5천원
            max_position = available_balance * 0.3  # 최대 30%

            position_size = max(min_position, min(position_size, max_position))

            return position_size

        except Exception as e:
            self.logger.error(f"포지션 사이즈 계산 중 오류: {e}")
            return available_balance * 0.1  # 안전한 기본값

    def update_balance_tracking(self, current_balance: float):
        """잔고 추적 업데이트"""
        try:
            now = datetime.now()

            # 일일 시작 잔고 설정 (자정 이후 첫 업데이트)
            if self.daily_start_balance == 0 or now.hour == 0:
                self.daily_start_balance = current_balance

            # 최고점 업데이트
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance

            # 잔고 변화 기록 (최근 1시간만 보관)
            cutoff_time = now - timedelta(hours=1)
            self.balance_history = [
                (timestamp, balance) for timestamp, balance in getattr(self, 'balance_history', [])
                if timestamp > cutoff_time
            ]
            self.balance_history.append((now, current_balance))

        except Exception as e:
            self.logger.error(f"잔고 추적 업데이트 중 오류: {e}")

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """변동성 계산"""
        if len(data) < window:
            return 0.5  # 기본값

        try:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=window).std().iloc[-1]
            return volatility * np.sqrt(24)  # 일일 변동성으로 환산

        except Exception:
            return 0.5

    def _assess_market_trend(self, data: pd.DataFrame) -> str:
        """시장 트렌드 평가"""
        if len(data) < 50:
            return 'NEUTRAL'

        try:
            # 단순 이동평균 기반 트렌드 판단
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            current_price = data['close'].iloc[-1]

            if current_price > sma_20 > sma_50:
                return 'BULLISH'
            elif current_price < sma_20 < sma_50:
                return 'BEARISH'
            else:
                return 'NEUTRAL'

        except Exception:
            return 'NEUTRAL'

    def _calculate_concentration(self, positions: Dict, current_prices: Dict[str, float]) -> float:
        """포지션 집중도 계산"""
        if not positions:
            return 0.0

        try:
            total_value = sum(
                pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
                for pos in positions.values()
            )

            max_position_value = max(
                pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
                for pos in positions.values()
            )

            return max_position_value / total_value if total_value > 0 else 0

        except Exception:
            return 1.0  # 안전하게 최대값 반환

    def _calculate_correlation_risk(self, positions: Dict) -> float:
        """상관관계 리스크 계산 (단순화)"""
        if len(positions) < 2:
            return 0.0

        try:
            symbols = [pos.symbol for pos in positions.values()]
            correlations = []

            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    # 간단한 휴리스틱 기반 상관관계
                    coin1 = symbol1.split('-')[1]
                    coin2 = symbol2.split('-')[1]

                    if coin1 in ['BTC', 'ETH'] and coin2 in ['BTC', 'ETH']:
                        correlations.append(0.9)  # 메이저 코인간 높은 상관관계
                    elif coin1 not in ['BTC', 'ETH'] and coin2 not in ['BTC', 'ETH']:
                        correlations.append(0.7)  # 알트코인간 중간 상관관계
                    else:
                        correlations.append(0.5)  # 메이저-알트간 중간 상관관계

            return np.mean(correlations) if correlations else 0.0

        except Exception:
            return 0.8  # 안전하게 높은 값 반환

    def _get_recent_balance_changes(self) -> List[float]:
        """최근 잔고 변화율 계산"""
        try:
            if not hasattr(self, 'balance_history') or len(self.balance_history) < 2:
                return []

            changes = []
            for i in range(1, len(self.balance_history)):
                prev_balance = self.balance_history[i-1][1]
                curr_balance = self.balance_history[i][1]
                change_rate = (curr_balance - prev_balance) / prev_balance
                changes.append(change_rate)

            return changes

        except Exception:
            return []

    def _generate_recommendations(self, level: RiskLevel, factors: List[str]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        if level == RiskLevel.LOW:
            recommendations.append("거래 진행 가능")

        elif level == RiskLevel.MEDIUM:
            recommendations.append("포지션 크기 축소 고려")
            if "고변동성" in str(factors):
                recommendations.append("스탑로스 타이트하게 설정")

        elif level == RiskLevel.HIGH:
            recommendations.append("거래 보류 고려")
            recommendations.append("추가 분석 필요")
            if "낮은 신호 강도" in str(factors):
                recommendations.append("신호 재검증 필요")

        else:  # CRITICAL
            recommendations.append("거래 중단")
            recommendations.append("포지션 정리 고려")

        return recommendations

    def _generate_portfolio_recommendations(self, level: RiskLevel, factors: List[str]) -> List[str]:
        """포트폴리오 권장사항 생성"""
        recommendations = []

        if level == RiskLevel.LOW:
            recommendations.append("포트폴리오 양호")

        elif level == RiskLevel.MEDIUM:
            recommendations.append("리스크 모니터링 강화")
            if "포지션 집중도" in str(factors):
                recommendations.append("포지션 분산 고려")

        elif level == RiskLevel.HIGH:
            recommendations.append("일부 포지션 정리 고려")
            recommendations.append("신규 포지션 제한")

        else:  # CRITICAL
            recommendations.append("즉시 포지션 정리")
            recommendations.append("거래 일시 중단")

        return recommendations

    def _log_risk_event(self, event_type: str, value: float):
        """리스크 이벤트 기록"""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'value': value,
            'description': f"{event_type}: {value:.2%}"
        }

        self.risk_events.append(event)

        # 최근 24시간 이벤트만 보관
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.risk_events = [
            event for event in self.risk_events
            if event['timestamp'] > cutoff_time
        ]

    def get_risk_summary(self) -> Dict:
        """리스크 요약 반환"""
        return {
            'daily_loss_used': (self.daily_start_balance - self.peak_balance) / self.daily_start_balance if self.daily_start_balance > 0 else 0,
            'max_daily_loss': self.max_daily_loss,
            'current_drawdown': (self.peak_balance - self.daily_start_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'recent_risk_events': self.risk_events[-5:],  # 최근 5개
            'risk_events_24h': len(self.risk_events)
        }