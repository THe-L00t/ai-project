#!/usr/bin/env python3
"""
백테스팅 엔진
- 과거 데이터를 이용한 전략 성능 검증
- 리스크 지표 계산 및 분석
- 포트폴리오 시뮬레이션
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns


class OrderType(Enum):
    """주문 타입"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    market: str
    order_type: OrderType
    price: float
    quantity: float
    amount: float
    fee: float

    # AI 결정 정보
    signal_type: str
    confidence: float
    reasoning: str

    # 성과 정보 (나중에 계산)
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    holding_period: Optional[int] = None


@dataclass
class PortfolioSnapshot:
    """포트폴리오 스냅샷"""
    timestamp: datetime
    cash: float
    positions: Dict[str, float]  # market -> quantity
    position_values: Dict[str, float]  # market -> value
    total_value: float
    daily_return: float
    cumulative_return: float


@dataclass
class BacktestResult:
    """백테스트 결과"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float

    # 수익성 지표
    total_return: float
    annualized_return: float
    daily_returns: List[float]

    # 리스크 지표
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)

    # 거래 지표
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float

    # 포트폴리오 기록
    portfolio_history: List[PortfolioSnapshot]
    trade_history: List[Trade]

    # 추가 메트릭
    calmar_ratio: float
    information_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None


class PortfolioManager:
    """
    포트폴리오 관리자
    백테스팅 중 포트폴리오 상태 관리
    """

    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        """
        초기화

        Args:
            initial_capital: 초기 자본
            commission_rate: 수수료율
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        # 포트폴리오 상태
        self.cash = initial_capital
        self.positions = {}  # market -> quantity
        self.position_entry_prices = {}  # market -> entry_price

        # 기록
        self.portfolio_history = []
        self.trade_history = []

        self.logger = logging.getLogger(__name__)

    def execute_trade(self, timestamp: datetime, market: str, order_type: OrderType,
                     price: float, amount: float, signal_info: Dict = None) -> Optional[Trade]:
        """
        거래 실행

        Args:
            timestamp: 거래 시간
            market: 마켓 코드
            order_type: 주문 타입
            price: 가격
            amount: 거래 금액 (매수시) 또는 수량 (매도시)
            signal_info: 신호 정보

        Returns:
            거래 기록
        """
        try:
            if order_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
                return self._execute_buy(timestamp, market, order_type, price, amount, signal_info)
            else:
                return self._execute_sell(timestamp, market, order_type, price, amount, signal_info)

        except Exception as e:
            self.logger.error(f"거래 실행 오류: {e}")
            return None

    def _execute_buy(self, timestamp: datetime, market: str, order_type: OrderType,
                    price: float, amount: float, signal_info: Dict) -> Optional[Trade]:
        """매수 실행"""
        if amount <= 0 or price <= 0:
            return None

        # 수수료 포함 총 비용 계산
        quantity = amount / price
        fee = amount * self.commission_rate
        total_cost = amount + fee

        # 현금 부족 체크
        if self.cash < total_cost:
            available_amount = self.cash / (1 + self.commission_rate)
            if available_amount < price:  # 최소 1코인도 못 사는 경우
                return None

            # 가용한 현금으로 최대한 매수
            quantity = available_amount / price
            amount = quantity * price
            fee = amount * self.commission_rate
            total_cost = amount + fee

        # 포트폴리오 업데이트
        self.cash -= total_cost
        current_position = self.positions.get(market, 0)

        # 가중평균 매수가격 계산
        if current_position > 0:
            old_value = current_position * self.position_entry_prices[market]
            new_value = old_value + amount
            self.position_entry_prices[market] = new_value / (current_position + quantity)
        else:
            self.position_entry_prices[market] = price

        self.positions[market] = current_position + quantity

        # 거래 기록
        trade = Trade(
            timestamp=timestamp,
            market=market,
            order_type=order_type,
            price=price,
            quantity=quantity,
            amount=amount,
            fee=fee,
            signal_type=signal_info.get('signal_type', ''),
            confidence=signal_info.get('confidence', 0.0),
            reasoning=signal_info.get('reasoning', '')
        )

        self.trade_history.append(trade)
        return trade

    def _execute_sell(self, timestamp: datetime, market: str, order_type: OrderType,
                     price: float, quantity: float, signal_info: Dict) -> Optional[Trade]:
        """매도 실행"""
        current_position = self.positions.get(market, 0)

        if current_position <= 0 or quantity <= 0:
            return None

        # 매도 가능 수량 조정
        sell_quantity = min(quantity, current_position)
        amount = sell_quantity * price
        fee = amount * self.commission_rate
        net_proceeds = amount - fee

        # 손익 계산
        entry_price = self.position_entry_prices.get(market, price)
        profit_loss = (price - entry_price) * sell_quantity
        profit_loss_pct = profit_loss / (entry_price * sell_quantity) if entry_price > 0 else 0.0

        # 포트폴리오 업데이트
        self.cash += net_proceeds
        self.positions[market] = current_position - sell_quantity

        # 포지션을 모두 정리한 경우
        if self.positions[market] <= 0:
            self.positions.pop(market, None)
            self.position_entry_prices.pop(market, None)

        # 거래 기록
        trade = Trade(
            timestamp=timestamp,
            market=market,
            order_type=order_type,
            price=price,
            quantity=sell_quantity,
            amount=amount,
            fee=fee,
            signal_type=signal_info.get('signal_type', ''),
            confidence=signal_info.get('confidence', 0.0),
            reasoning=signal_info.get('reasoning', ''),
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct
        )

        self.trade_history.append(trade)
        return trade

    def update_portfolio_value(self, timestamp: datetime, current_prices: Dict[str, float]):
        """
        포트폴리오 가치 업데이트

        Args:
            timestamp: 현재 시간
            current_prices: 현재 가격 정보 {market: price}
        """
        position_values = {}
        total_position_value = 0.0

        for market, quantity in self.positions.items():
            if market in current_prices:
                value = quantity * current_prices[market]
                position_values[market] = value
                total_position_value += value

        total_value = self.cash + total_position_value

        # 수익률 계산
        daily_return = 0.0
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].total_value
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0

        cumulative_return = (total_value - self.initial_capital) / self.initial_capital

        # 스냅샷 생성
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            position_values=position_values,
            total_value=total_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return
        )

        self.portfolio_history.append(snapshot)

    def get_current_value(self, current_prices: Dict[str, float]) -> float:
        """현재 포트폴리오 가치 계산"""
        position_value = sum(
            quantity * current_prices.get(market, 0)
            for market, quantity in self.positions.items()
        )
        return self.cash + position_value


class BacktestEngine:
    """
    백테스팅 엔진
    전략의 과거 성과를 시뮬레이션하고 분석
    """

    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001):
        """
        초기화

        Args:
            initial_capital: 초기 자본
            commission_rate: 수수료율
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        self.logger = logging.getLogger(__name__)

        # 결과 저장용
        self.results = {}

    def run_backtest(self, strategy_func, market_data: pd.DataFrame,
                    start_date: str = None, end_date: str = None,
                    benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """
        백테스트 실행

        Args:
            strategy_func: 전략 함수 (market_data -> signal)
            market_data: 시장 데이터
            start_date: 시작 날짜
            end_date: 종료 날짜
            benchmark_data: 벤치마크 데이터 (비교 대상)

        Returns:
            백테스트 결과
        """
        try:
            self.logger.info("백테스트 시작")

            # 데이터 필터링
            test_data = self._prepare_data(market_data, start_date, end_date)

            if len(test_data) == 0:
                raise ValueError("백테스트 데이터가 없습니다")

            # 포트폴리오 매니저 초기화
            portfolio = PortfolioManager(self.initial_capital, self.commission_rate)

            # 백테스트 실행
            for i in range(len(test_data)):
                current_data = test_data.iloc[:i+50]  # 충분한 히스토리 제공

                if len(current_data) < 50:  # 최소 데이터 요구사항
                    continue

                current_row = test_data.iloc[i]
                timestamp = pd.to_datetime(current_row['timestamp'])
                current_prices = {current_row['market']: current_row['close_price']}

                try:
                    # 전략 신호 생성
                    signal = strategy_func(current_data)

                    # 신호에 따른 거래 실행
                    if signal and signal.final_signal.value != 'hold':
                        self._execute_strategy_signal(
                            portfolio, timestamp, current_row, signal
                        )

                except Exception as e:
                    self.logger.error(f"전략 실행 오류 (인덱스 {i}): {e}")
                    continue

                # 포트폴리오 가치 업데이트
                portfolio.update_portfolio_value(timestamp, current_prices)

            # 결과 분석
            result = self._analyze_results(
                portfolio, test_data, benchmark_data
            )

            self.logger.info(f"백테스트 완료 - 총 수익률: {result.total_return:.2%}")
            return result

        except Exception as e:
            self.logger.error(f"백테스트 실행 오류: {e}")
            raise

    def _prepare_data(self, market_data: pd.DataFrame,
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """백테스트 데이터 준비"""
        data = market_data.copy()

        # 타임스탬프 변환
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # 날짜 필터링
        if start_date:
            data = data[data['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['timestamp'] <= pd.to_datetime(end_date)]

        # 정렬
        data = data.sort_values('timestamp').reset_index(drop=True)

        return data

    def _execute_strategy_signal(self, portfolio: PortfolioManager,
                               timestamp: datetime, current_row: pd.Series,
                               signal) -> None:
        """전략 신호에 따른 거래 실행"""
        market = current_row['market']
        price = current_row['close_price']
        signal_type = signal.final_signal.value

        # 신호 정보
        signal_info = {
            'signal_type': signal_type,
            'confidence': signal.confidence,
            'reasoning': signal.reasoning or ''
        }

        # 현재 포트폴리오 가치
        current_value = portfolio.get_current_value({market: price})

        # 신호별 거래 실행
        if signal_type in ['strong_buy', 'buy']:
            # 매수 신호
            position_ratio = 0.3 if signal_type == 'strong_buy' else 0.1
            buy_amount = current_value * position_ratio

            if buy_amount >= price:  # 최소 거래 금액 체크
                portfolio.execute_trade(
                    timestamp, market, OrderType.MARKET_BUY,
                    price, buy_amount, signal_info
                )

        elif signal_type in ['strong_sell', 'sell']:
            # 매도 신호
            current_position = portfolio.positions.get(market, 0)
            if current_position > 0:
                sell_ratio = 0.7 if signal_type == 'strong_sell' else 0.3
                sell_quantity = current_position * sell_ratio

                portfolio.execute_trade(
                    timestamp, market, OrderType.MARKET_SELL,
                    price, sell_quantity, signal_info
                )

    def _analyze_results(self, portfolio: PortfolioManager,
                        test_data: pd.DataFrame,
                        benchmark_data: pd.DataFrame = None) -> BacktestResult:
        """백테스트 결과 분석"""
        if len(portfolio.portfolio_history) == 0:
            raise ValueError("포트폴리오 히스토리가 없습니다")

        # 기본 정보
        start_date = portfolio.portfolio_history[0].timestamp
        end_date = portfolio.portfolio_history[-1].timestamp
        final_value = portfolio.portfolio_history[-1].total_value

        # 수익률 계산
        total_return = (final_value - self.initial_capital) / self.initial_capital
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 일별 수익률
        daily_returns = [snapshot.daily_return for snapshot in portfolio.portfolio_history[1:]]

        # 리스크 지표 계산
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # 2% 무위험수익률 가정

        # Sortino 비율 (하방 변동성만 고려)
        negative_returns = [r for r in daily_returns if r < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
        sortino_ratio = (annualized_return - 0.02) / downside_volatility if downside_volatility > 0 else 0

        # 최대 낙폭 (Maximum Drawdown)
        portfolio_values = [snapshot.total_value for snapshot in portfolio.portfolio_history]
        peak_values = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak_values - portfolio_values) / peak_values
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # VaR (Value at Risk) 95%
        var_95 = np.percentile(daily_returns, 5) if daily_returns else 0

        # 거래 분석
        profitable_trades = [t for t in portfolio.trade_history if t.profit_loss and t.profit_loss > 0]
        losing_trades = [t for t in portfolio.trade_history if t.profit_loss and t.profit_loss < 0]

        total_trades = len([t for t in portfolio.trade_history if t.profit_loss is not None])
        winning_trades = len(profitable_trades)
        losing_trades_count = len(losing_trades)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = np.mean([t.profit_loss for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([abs(t.profit_loss) for t in losing_trades]) if losing_trades else 0
        profit_factor = (avg_profit * winning_trades) / (avg_loss * losing_trades_count) if avg_loss > 0 and losing_trades_count > 0 else 0

        # 추가 지표
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        information_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 else 0

        # 벤치마크 대비 베타, 알파 (벤치마크 데이터가 있는 경우)
        beta, alpha = None, None
        if benchmark_data is not None:
            try:
                beta, alpha = self._calculate_beta_alpha(daily_returns, benchmark_data)
            except Exception as e:
                self.logger.warning(f"베타/알파 계산 실패: {e}")

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=daily_returns,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades_count,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            portfolio_history=portfolio.portfolio_history,
            trade_history=portfolio.trade_history,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha
        )

    def _calculate_beta_alpha(self, strategy_returns: List[float],
                            benchmark_data: pd.DataFrame) -> Tuple[float, float]:
        """베타와 알파 계산"""
        # 간단한 구현 - 실제로는 더 정교한 계산이 필요
        if len(strategy_returns) < 2:
            return 0.0, 0.0

        # 벤치마크 수익률 계산 (예시)
        benchmark_returns = benchmark_data['close_price'].pct_change().dropna().tolist()

        if len(benchmark_returns) < len(strategy_returns):
            benchmark_returns = benchmark_returns * (len(strategy_returns) // len(benchmark_returns) + 1)

        benchmark_returns = benchmark_returns[:len(strategy_returns)]

        if len(benchmark_returns) != len(strategy_returns):
            return 0.0, 0.0

        # 선형 회귀를 통한 베타 계산
        covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)

        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # 알파 계산
        strategy_mean = np.mean(strategy_returns)
        benchmark_mean = np.mean(benchmark_returns)
        alpha = strategy_mean - beta * benchmark_mean

        return beta, alpha

    def generate_report(self, result: BacktestResult) -> str:
        """백테스트 결과 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("백테스트 결과 리포트")
        report.append("=" * 60)
        report.append("")

        # 기본 정보
        report.append("📊 기본 정보")
        report.append(f"기간: {result.start_date.strftime('%Y-%m-%d')} ~ {result.end_date.strftime('%Y-%m-%d')}")
        report.append(f"초기 자본: {result.initial_capital:,.0f}원")
        report.append(f"최종 자산: {result.final_value:,.0f}원")
        report.append("")

        # 수익성 지표
        report.append("💰 수익성 지표")
        report.append(f"총 수익률: {result.total_return:.2%}")
        report.append(f"연환산 수익률: {result.annualized_return:.2%}")
        report.append(f"절대 수익: {result.final_value - result.initial_capital:,.0f}원")
        report.append("")

        # 리스크 지표
        report.append("⚠️ 리스크 지표")
        report.append(f"변동성 (연환산): {result.volatility:.2%}")
        report.append(f"샤프 비율: {result.sharpe_ratio:.3f}")
        report.append(f"소르티노 비율: {result.sortino_ratio:.3f}")
        report.append(f"최대 낙폭: {result.max_drawdown:.2%}")
        report.append(f"VaR (95%): {result.var_95:.2%}")
        report.append(f"칼마 비율: {result.calmar_ratio:.3f}")
        report.append("")

        # 거래 지표
        report.append("🔄 거래 지표")
        report.append(f"총 거래 수: {result.total_trades}")
        report.append(f"수익 거래: {result.winning_trades} ({result.win_rate:.1%})")
        report.append(f"손실 거래: {result.losing_trades}")
        report.append(f"평균 수익: {result.avg_profit:,.0f}원")
        report.append(f"평균 손실: {result.avg_loss:,.0f}원")
        report.append(f"수익 팩터: {result.profit_factor:.2f}")
        report.append("")

        # 벤치마크 비교 (있는 경우)
        if result.beta is not None and result.alpha is not None:
            report.append("📈 벤치마크 비교")
            report.append(f"베타: {result.beta:.3f}")
            report.append(f"알파: {result.alpha:.2%}")
            report.append("")

        return "\n".join(report)

    def save_results(self, result: BacktestResult, filename: str):
        """백테스트 결과 저장"""
        try:
            # JSON으로 직렬화 가능한 형태로 변환
            result_dict = {
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'initial_capital': result.initial_capital,
                'final_value': result.final_value,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'var_95': result.var_95,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'avg_profit': result.avg_profit,
                'avg_loss': result.avg_loss,
                'profit_factor': result.profit_factor,
                'calmar_ratio': result.calmar_ratio,
                'information_ratio': result.information_ratio
            }

            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"백테스트 결과 저장: {filename}")

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """백테스트 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('백테스트 결과 분석', fontsize=16)

            # 포트폴리오 가치 변화
            dates = [snapshot.timestamp for snapshot in result.portfolio_history]
            values = [snapshot.total_value for snapshot in result.portfolio_history]

            axes[0, 0].plot(dates, values, linewidth=2)
            axes[0, 0].axhline(y=result.initial_capital, color='r', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Value (KRW)')
            axes[0, 0].grid(True, alpha=0.3)

            # 누적 수익률
            cumulative_returns = [(v/result.initial_capital - 1) * 100 for v in values]
            axes[0, 1].plot(dates, cumulative_returns, linewidth=2, color='green')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('Cumulative Returns (%)')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].grid(True, alpha=0.3)

            # 드로다운
            peak_values = np.maximum.accumulate(values)
            drawdowns = [(peak - val) / peak * 100 for peak, val in zip(peak_values, values)]
            axes[1, 0].fill_between(dates, drawdowns, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown (%)')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)

            # 일별 수익률 분포
            daily_returns_pct = [r * 100 for r in result.daily_returns]
            axes[1, 1].hist(daily_returns_pct, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Return (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"차트 저장: {save_path}")

            plt.show()

        except Exception as e:
            self.logger.error(f"차트 생성 실패: {e}")