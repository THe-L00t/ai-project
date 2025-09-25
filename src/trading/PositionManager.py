"""
포지션 관리 시스템
리스크 관리와 포트폴리오 최적화를 담당합니다.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    signal: str
    order_uuid: Optional[str] = None


class PositionManager:
    """포지션 관리자"""

    def __init__(self, max_positions: int = 5,
                 max_position_size: float = 0.2,
                 correlation_threshold: float = 0.7):
        """
        초기화

        Args:
            max_positions: 최대 동시 포지션 수
            max_position_size: 단일 포지션 최대 비중 (20%)
            correlation_threshold: 상관관계 임계값
        """
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.correlation_threshold = correlation_threshold

        self.positions: Dict[str, Position] = {}
        self.correlation_matrix = {}

        self.logger = logging.getLogger(__name__)

    def can_add_position(self, symbol: str, current_prices: Dict[str, float]) -> bool:
        """새 포지션 추가 가능 여부 확인"""

        # 1. 최대 포지션 수 체크
        if len(self.positions) >= self.max_positions:
            self.logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.max_positions}")
            return False

        # 2. 이미 보유 중인지 체크
        if symbol in self.positions:
            self.logger.info(f"{symbol} 이미 포지션 보유 중")
            return False

        # 3. 상관관계 체크
        if self._check_correlation_risk(symbol, current_prices):
            self.logger.info(f"{symbol} 상관관계 리스크로 인한 포지션 제한")
            return False

        return True

    def add_position(self, position: Position) -> bool:
        """포지션 추가"""
        try:
            if position.symbol not in self.positions:
                self.positions[position.symbol] = position
                self.logger.info(f"포지션 추가: {position.symbol} @ {position.entry_price}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"포지션 추가 중 오류: {e}")
            return False

    def remove_position(self, symbol: str) -> Optional[Position]:
        """포지션 제거"""
        return self.positions.pop(symbol, None)

    def update_stop_loss(self, symbol: str, new_stop_loss: float):
        """트레일링 스탑 업데이트"""
        if symbol in self.positions:
            position = self.positions[symbol]

            # 매수 포지션의 경우 스탑로스를 위로만 이동
            if new_stop_loss > position.stop_loss:
                position.stop_loss = new_stop_loss
                self.logger.info(f"{symbol} 스탑로스 업데이트: {new_stop_loss}")

    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """청산 조건 확인"""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # 손절 체크
        if current_price <= position.stop_loss:
            return 'STOP_LOSS'

        # 익절 체크
        if current_price >= position.take_profit:
            return 'TAKE_PROFIT'

        # 시간 기반 청산 (24시간 초과)
        if datetime.now() - position.entry_time > timedelta(hours=24):
            return 'TIME_EXIT'

        # 트레일링 스탑 업데이트
        self._update_trailing_stop(symbol, current_price)

        return None

    def _update_trailing_stop(self, symbol: str, current_price: float):
        """트레일링 스탑 업데이트"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        profit_rate = (current_price - position.entry_price) / position.entry_price

        # 10% 이상 수익시 트레일링 스탑 활성화
        if profit_rate > 0.1:
            # 현재가의 5% 아래로 스탑로스 설정
            new_stop_loss = current_price * 0.95
            self.update_stop_loss(symbol, new_stop_loss)

    def _check_correlation_risk(self, symbol: str, current_prices: Dict[str, float]) -> bool:
        """상관관계 리스크 체크"""
        if not self.positions:
            return False

        try:
            # 간단한 가격 변동률 기반 상관관계 체크
            for existing_symbol in self.positions.keys():
                if self._get_correlation(symbol, existing_symbol) > self.correlation_threshold:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"상관관계 체크 중 오류: {e}")
            return False

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """두 심볼간 상관관계 계산 (단순화)"""

        # 실제로는 과거 가격 데이터를 사용해야 하지만,
        # 여기서는 코인 쌍 기반 간단한 휴리스틱 사용

        major_coins = ['BTC', 'ETH']
        altcoins = ['ADA', 'DOT', 'LINK', 'XRP', 'MATIC', 'SOL']

        coin1 = symbol1.split('-')[1]
        coin2 = symbol2.split('-')[1]

        # 같은 그룹 내에서는 높은 상관관계 가정
        if (coin1 in major_coins and coin2 in major_coins) or \
           (coin1 in altcoins and coin2 in altcoins):
            return 0.8

        # 다른 그룹 간에는 중간 상관관계
        return 0.5

    def calculate_portfolio_risk(self, current_prices: Dict[str, float]) -> Dict:
        """포트폴리오 리스크 계산"""
        if not self.positions:
            return {'total_risk': 0, 'concentration_risk': 0, 'correlation_risk': 0}

        try:
            total_value = sum(
                pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
                for pos in self.positions.values()
            )

            # 집중도 리스크
            concentration_risk = max(
                (pos.quantity * current_prices.get(pos.symbol, pos.entry_price)) / total_value
                for pos in self.positions.values()
            )

            # 상관관계 리스크 (단순화)
            symbols = list(self.positions.keys())
            correlation_risk = 0

            if len(symbols) > 1:
                correlations = []
                for i, symbol1 in enumerate(symbols):
                    for symbol2 in symbols[i+1:]:
                        corr = self._get_correlation(symbol1, symbol2)
                        correlations.append(corr)

                correlation_risk = np.mean(correlations) if correlations else 0

            total_risk = (concentration_risk + correlation_risk) / 2

            return {
                'total_risk': total_risk,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'position_count': len(self.positions),
                'max_position_size': max_position_size if self.positions else 0
            }

        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 계산 중 오류: {e}")
            return {'total_risk': 1.0, 'concentration_risk': 1.0, 'correlation_risk': 1.0}

    def get_position_summary(self, current_prices: Dict[str, float]) -> Dict:
        """포지션 요약"""
        if not self.positions:
            return {'total_positions': 0, 'total_value': 0, 'total_pnl': 0}

        try:
            total_value = 0
            total_pnl = 0
            position_details = []

            for symbol, position in self.positions.items():
                current_price = current_prices.get(symbol, position.entry_price)
                position_value = position.quantity * current_price
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_rate = (current_price - position.entry_price) / position.entry_price * 100

                total_value += position_value
                total_pnl += pnl

                position_details.append({
                    'symbol': symbol,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'quantity': position.quantity,
                    'value': position_value,
                    'pnl': pnl,
                    'pnl_rate': pnl_rate,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'duration': (datetime.now() - position.entry_time).total_seconds() / 3600
                })

            return {
                'total_positions': len(self.positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_rate': (total_pnl / sum(pos.entry_price * pos.quantity for pos in self.positions.values())) * 100,
                'positions': position_details
            }

        except Exception as e:
            self.logger.error(f"포지션 요약 중 오류: {e}")
            return {'total_positions': 0, 'total_value': 0, 'total_pnl': 0}

    def optimize_position_sizes(self, signals: Dict[str, Dict],
                              available_balance: float) -> Dict[str, float]:
        """포지션 사이즈 최적화"""
        if not signals:
            return {}

        try:
            # Kelly Criterion 기반 포지션 사이징 (단순화)
            optimized_sizes = {}

            # 신뢰도와 기대수익률 기반 가중치 계산
            total_weight = 0
            weights = {}

            for symbol, signal_data in signals.items():
                confidence = signal_data.get('confidence', 0.7)
                expected_return = signal_data.get('expected_return', 0.05)

                # Kelly fraction 계산 (단순화)
                win_prob = confidence
                win_loss_ratio = 2.0  # 2:1 리스크 리워드

                kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
                kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))  # 제한

                weights[symbol] = kelly_fraction
                total_weight += kelly_fraction

            # 가용 자금 배분
            if total_weight > 0:
                for symbol, weight in weights.items():
                    normalized_weight = weight / total_weight
                    position_size = available_balance * normalized_weight

                    # 최소/최대 제한
                    position_size = min(position_size, available_balance * self.max_position_size)
                    if position_size >= 5000:  # 최소 거래 금액
                        optimized_sizes[symbol] = position_size

            return optimized_sizes

        except Exception as e:
            self.logger.error(f"포지션 사이즈 최적화 중 오류: {e}")
            return {}

    def should_close_position(self, symbol: str, current_price: float,
                            market_condition: str = 'NORMAL') -> bool:
        """포지션 청산 여부 판단"""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        # 기본 청산 조건
        exit_condition = self.check_exit_conditions(symbol, current_price)
        if exit_condition:
            return True

        # 시장 상황 기반 청산
        if market_condition == 'PANIC':
            # 패닉 상황에서는 손해를 보더라도 청산
            return True
        elif market_condition == 'HIGH_VOLATILITY':
            # 고변동성 상황에서는 작은 수익이라도 청산
            profit_rate = (current_price - position.entry_price) / position.entry_price
            if profit_rate > 0.03:  # 3% 이상 수익시 청산
                return True

        return False